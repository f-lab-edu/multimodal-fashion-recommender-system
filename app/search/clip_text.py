from __future__ import annotations

from typing import List

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel


class ClipTextEncoder:
    def __init__(
        self, model_name: str = "patrickjohncyh/fashion-clip", device: str | None = None
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.max_length = 77

    @torch.no_grad()
    def encode(self, texts: List[str] | str) -> np.ndarray:
        """
        텍스트 리스트를 Fashion-CLIP 임베딩으로 변환.

        Args:
            texts: 임베딩할 문자열 리스트

        Returns:
            shape = (len(texts), hidden_dim) 의 numpy 배열 (L2 정규화)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            images=None,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}

        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()

    def __call__(self, texts: List[str] | str) -> np.ndarray:
        return self.encode(texts)
