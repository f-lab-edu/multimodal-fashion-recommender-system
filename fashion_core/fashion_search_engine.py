# fashion_core/fashion_search_engine.py
# Fashion-CLIP + FAISS 인덱스를 사용해서
# 텍스트 쿼리로 패션 아이템을 검색하는 유틸

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
import faiss

from googletrans import Translator

FASHION_CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"


class ShardedFashionSearchEngine:
    """
    여러 개의 (FAISS 인덱스, 메타데이터) 샤드를 한 번에 검색하는 엔진.

    - 모델/프로세서는 1번만 로드
    - 각 샤드를 모두 검색하고, 점수를 기준으로 글로벌 top-k를 선택
    """

    def __init__(
        self,
        shard_specs: List[tuple[Path, Path]],  # (index_path, meta_path) 리스트
        model_name: str = FASHION_CLIP_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.shard_specs = shard_specs
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.translator = Translator()

        print(f"[INFO] Using device = {self.device}")
        print(f"[INFO] MODEL      = {self.model_name}")
        print(f"[INFO] NUM SHARDS = {len(self.shard_specs)}")

        # 1) 모든 샤드 인덱스 + 메타데이터 로드
        self.shards: List[Dict[str, Any]] = []
        for i, (index_path, meta_path) in enumerate(self.shard_specs):
            print(f"[INFO] Loading shard {i}:")
            print(f"       INDEX_PATH = {index_path}")
            print(f"       META_PATH  = {meta_path}")

            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            index = faiss.read_index(str(index_path))
            print(f"       -> index.ntotal = {index.ntotal}")

            metadata: List[Dict[str, Any]] = []
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    metadata.append(json.loads(line))

            print(f"       -> metadata entries = {len(metadata)}")

            if len(metadata) != index.ntotal:
                print(
                    f"[WARN] shard {i}: metadata count ({len(metadata)}) != "
                    f"index.ntotal ({index.ntotal})"
                )

            self.shards.append(
                {
                    "index_path": index_path,
                    "meta_path": meta_path,
                    "index": index,
                    "metadata": metadata,
                }
            )

        # 2) 모델/프로세서 1번만 로드
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    @classmethod
    def from_prefix(
        cls,
        base_prefix: Path,
        model_name: str = FASHION_CLIP_MODEL_NAME,
        device: Optional[str] = None,
    ) -> "ShardedFashionSearchEngine":
        """
        base_prefix 기준으로 샤드 파일들을 자동 탐색해서 엔진 생성.

        기대하는 파일 이름 패턴:
          - {base_prefix}_shard_0.faiss
          - {base_prefix}_shard_0_meta.jsonl
          - {base_prefix}_shard_1.faiss
          - {base_prefix}_shard_1_meta.jsonl
          - ...

        예: base_prefix = Path("data/fashion_index")
        """
        base_prefix = Path(base_prefix)
        directory = base_prefix.parent  # data/
        prefix_name = base_prefix.name  # "fashion_index"

        # 예: fashion_index_shard_*.faiss
        faiss_files = sorted(directory.glob(f"{prefix_name}_shard_*.faiss"))

        if not faiss_files:
            raise FileNotFoundError(
                f"No shard FAISS files found for prefix: {base_prefix}"
            )

        shard_specs: list[tuple[Path, Path]] = []

        # 예: fashion_index_shard_0 → group(1) = "0"
        shard_pattern = re.compile(rf"^{re.escape(prefix_name)}_shard_(\d+)$")

        for faiss_path in faiss_files:
            stem = faiss_path.stem  # "fashion_index_shard_0"
            m = shard_pattern.match(stem)
            if not m:
                print(f"[WARN] Skip file (name pattern mismatch): {faiss_path}")
                continue

            shard_id = int(m.group(1))
            meta_name = f"{prefix_name}_shard_{shard_id}_meta.jsonl"
            meta_path = directory / meta_name

            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Meta file for shard {shard_id} not found: {meta_path}"
                )

            shard_specs.append((faiss_path, meta_path))

        if not shard_specs:
            raise RuntimeError("No valid shard (index, meta) pairs found.")

        # shard_id 순으로 정렬
        shard_specs.sort(
            key=lambda pair: int(re.search(r"_(\d+)$", pair[0].stem).group(1))
        )

        return cls(
            shard_specs=shard_specs,
            model_name=model_name,
            device=device,
        )

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norms

    def preprocess_query(self, query: str) -> str:
        """
        - 쿼리에 한글이 섞여 있으면 ko -> en 번역
        """
        # 한글 문자 존재 여부 체크
        has_korean = any("\uac00" <= ch <= "\ud7a3" for ch in query)
        if not has_korean:
            return query

        try:
            translated = self.translator.translate(query, src="ko", dest="en").text
            print(f"[INFO] Translated query: {query!r} -> {translated!r}")
            return translated
        except Exception as e:
            print(f"[WARN] Translation failed ({e}), using original query.")
            return query

    def encode_text_query(self, query: str) -> np.ndarray:
        """
        텍스트 쿼리를 Fashion-CLIP 텍스트 임베딩으로 변환
        (모든 샤드에서 재사용)
        """
        # (한글 → 영어 번역)
        query = self.preprocess_query(query)

        inputs = self.processor(
            text=[query],
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.get_text_features(**inputs)

        emb = out.cpu().numpy().astype("float32")
        emb = self._normalize(emb)
        return emb

    def search(
        self,
        query: str,
        top_k: int = 10,
        per_shard_k_factor: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        모든 샤드를 검색해서 글로벌 top_k 결과를 반환.

        - per_shard_k_factor: 샤드마다 top_k * factor 만큼 더 넉넉하게 뽑은 뒤
                             전체에서 다시 상위 top_k를 선택
        """
        print(f"[INFO] Global query: {query!r}")
        q_emb = self.encode_text_query(query)

        all_hits: List[Dict[str, Any]] = []
        per_shard_k = max(top_k, top_k * per_shard_k_factor)

        for shard_idx, shard in enumerate(self.shards):
            index = shard["index"]
            metadata = shard["metadata"]

            print(
                f"[INFO] Searching shard {shard_idx} "
                f"(ntotal={index.ntotal}) with top_k={per_shard_k}"
            )

            scores, indices = index.search(q_emb, per_shard_k)
            scores = scores[0]
            indices = indices[0]

            for local_rank, (idx, score) in enumerate(zip(indices, scores), start=1):
                if idx < 0 or idx >= len(metadata):
                    continue
                meta = metadata[idx]
                all_hits.append(
                    {
                        "shard_id": shard_idx,
                        "local_rank": local_rank,
                        "score": float(score),
                        "index_in_shard": int(idx),
                        "asin": meta.get("asin"),
                        "text": meta.get("text"),
                        "image_path": meta.get("image_path"),
                        "meta": meta,
                    }
                )

        # 점수 기준으로 글로벌 정렬 후 상위 top_k만
        all_hits.sort(key=lambda x: x["score"], reverse=True)
        top_hits = all_hits[:top_k]
        for global_rank, hit in enumerate(top_hits, start=1):
            hit["rank"] = global_rank
        return top_hits


class FusionSearchEngine:
    """
    텍스트 + 이미지 인덱스를 동시에 검색하고
    RRF(rank fusion)로 점수 합쳐서 최종 top-k 반환
    """

    def __init__(
        self,
        text_engine: ShardedFashionSearchEngine,
        image_engine: ShardedFashionSearchEngine,
        rrf_k: int = 60,
        w_text: float = 1.0,
        w_image: float = 1.0,
    ):
        self.text_engine = text_engine
        self.image_engine = image_engine
        self.rrf_k = rrf_k
        self.w_text = w_text
        self.w_image = w_image

    @classmethod
    def from_prefixes(
        cls,
        text_base_prefix: Path,
        image_base_prefix: Path,
        model_name: str = FASHION_CLIP_MODEL_NAME,
        device: Optional[str] = None,
        rrf_k: int = 60,
        w_text: float = 1.0,
        w_image: float = 1.0,
    ) -> "FusionSearchEngine":
        """
        텍스트/이미지 인덱스 prefix로부터 Sharded 엔진 두 개를 자동 생성.
        """
        text_engine = ShardedFashionSearchEngine.from_prefix(
            base_prefix=text_base_prefix,
            model_name=model_name,
            device=device,
        )
        image_engine = ShardedFashionSearchEngine.from_prefix(
            base_prefix=image_base_prefix,
            model_name=model_name,
            device=device,
        )
        return cls(
            text_engine=text_engine,
            image_engine=image_engine,
            rrf_k=rrf_k,
            w_text=w_text,
            w_image=w_image,
        )

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def search(
        self,
        query: str,
        top_k: int = 10,
        per_shard_k_factor: int = 2,
        stage1_factor: int = 3,  # 1차 후보군 배수
    ) -> List[Dict[str, Any]]:
        """
        1) 텍스트 인덱스에서 top-K 검색
        2) 이미지 인덱스에서 top-K 검색
        3) asin 기준으로 RRF 점수 합산 후 글로벌 top-K 반환
        """

        print(f"[INFO] Fusion search query: {query!r}")
        # 1단계 후보 개수
        stage1_k = top_k * stage1_factor

        # 1) 텍스트 샤드 검색
        text_hits = self.text_engine.search(
            query=query,
            top_k=stage1_k,
            per_shard_k_factor=per_shard_k_factor,
        )
        # 2) 이미지 샤드 검색
        image_hits = self.image_engine.search(
            query=query,
            top_k=stage1_k,
            per_shard_k_factor=per_shard_k_factor,
        )

        fused: Dict[str, Dict[str, Any]] = {}

        # 텍스트 결과 반영
        for h in text_hits:
            asin = h.get("asin")
            if not asin:
                continue
            if asin not in fused:
                fused[asin] = {
                    "asin": asin,
                    "text": h.get("text"),
                    "image_path": h.get("image_path"),
                    "meta": h.get("meta"),
                    "score_text_raw": h.get("score"),
                    "score_image_raw": None,
                    "rank_text": h.get("rank"),
                    "rank_image": None,
                    "rrf_score": 0.0,
                }
            rrf = self._rrf_score(h["rank"])
            fused[asin]["rrf_score"] += self.w_text * rrf

        # 이미지 결과 반영
        for h in image_hits:
            asin = h.get("asin")
            if not asin:
                continue
            if asin not in fused:
                fused[asin] = {
                    "asin": asin,
                    "text": h.get("text"),
                    "image_path": h.get("image_path"),
                    "meta": h.get("meta"),
                    "score_text_raw": None,
                    "score_image_raw": h.get("score"),
                    "rank_text": None,
                    "rank_image": h.get("rank"),
                    "rrf_score": 0.0,
                }
            else:
                # 텍스트 쪽에서 이미 기본 meta가 들어있으면, 이미지 경로가 더 “좋은” 게 있으면 덮어쓸 수도 있음
                if h.get("image_path"):
                    fused[asin]["image_path"] = h.get("image_path")
                fused[asin]["score_image_raw"] = h.get("score")
                fused[asin]["rank_image"] = h.get("rank")
            rrf = self._rrf_score(h["rank"])
            fused[asin]["rrf_score"] += self.w_image * rrf

        # RRF 점수 기준으로 정렬 후 상위 top_k만
        all_results = list(fused.values())
        all_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        for global_rank, r in enumerate(all_results[:top_k], start=1):
            r["rank"] = global_rank

        return all_results[:top_k]
