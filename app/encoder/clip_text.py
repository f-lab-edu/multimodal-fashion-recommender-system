from __future__ import annotations
import numpy as np
import torch
import open_clip


class ClipTextEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pretrained = pretrained
        # 실제 모델/토크나이저 로드
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def tokenize(self, texts: list[str]):
        # 실제: clip tokenizer
        return self.tokenizer(texts)
        # 더미: 토큰화 없이 placeholder 반환(디버그용)
        # return texts

    def encode_text(self, texts: list[str]) -> np.ndarray:
        # 실제: 텍스트 임베딩 생성
        with torch.no_grad():
            tokenized_texts = self.tokenize(texts).to(self.device)
            text_embeddings = self.model.encode_text(tokenized_texts)
            text_embeddings = text_embeddings / (
                text_embeddings.norm(dim=-1, keepdim=True) + 1e-9
            )
        return text_embeddings.detach().cpu().numpy().astype(np.float32)
        # 더미: 랜덤 임베딩 생성(디버그용)
        # D = 512
        # x = np.random.rand(len(texts), D).astype(np.float32)
        # x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        # return x
