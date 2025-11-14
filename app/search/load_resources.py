from __future__ import annotations
from pathlib import Path
import pandas as pd
import json


CATALOG_JSONL_PATH = Path("data/meta_Amazon_Fashion_v1.jsonl")


class ResourceLoader:
    def choose_main_image(self, images: list) -> str | None:
        """
        images 리스트에서 hi_res > large > thumb 순으로 하나 선택.
        """
        if not isinstance(images, list) or not images:
            return None

        # 1) variant == "MAIN" 인 것들
        main = [img for img in images if img.get("variant") == "MAIN"]
        candidates = main if main else images

        # 2) hi_res > large > thumb
        for key in ["hi_res", "large", "thumb"]:
            for img in candidates:
                if not isinstance(img, dict):
                    continue
                url = img.get(key)
                if url:
                    return url.strip()
        return None

    def join_list(self, x):
        if isinstance(x, list):
            return ", ".join(str(i).strip().lower() for i in x if str(i).strip())
        return ""

    def preprocess_catalog_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        아마존 패션 JSONL 로드 후 전처리
        """
        df = df.copy()

        if "parent_asin" in df.columns:
            df["item_id"] = df["parent_asin"].fillna("")
        else:
            df["item_id"] = df.index.astype(str)

        # 브랜드, store
        if "store" in df.columns:
            df["brand"] = df["store"]
        else:
            df["brand"] = ""

        # 이미지
        if "images" in df.columns:
            df["image_url"] = df["images"].apply(self.choose_main_image)
        else:
            df["image_url"] = None

        # 문자열 컬럼 정규화
        for col in ["title", "brand"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

        if "features" in df.columns:
            df["features_text"] = df["features"].apply(self.join_list)
        else:
            df["features_text"] = ""

        if "description" in df.columns:
            df["description_text"] = df["description"].apply(self.join_list)
        else:
            df["description_text"] = ""

        if "categories" in df.columns:
            df["categories_text"] = df["categories"].apply(self.join_list)
        else:
            df["categories_text"] = ""

        return df

    def get_catalog_df(self) -> pd.DataFrame:
        # df = pd.read_json(CATALOG_JSONL_PATH, lines=True)
        # df = self.preprocess_catalog_df(df)
        # return df
        """
        JSONL 파일을 한 줄씩 읽어서 DataFrame으로 변환.
        - 빈 줄은 건너뜀
        """
        records = []
        with CATALOG_JSONL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # 빈 줄이면 스킵
                    continue
                records.append(json.loads(line))

        df = pd.DataFrame(records)
        df = self.preprocess_catalog_df(df)
        return df
