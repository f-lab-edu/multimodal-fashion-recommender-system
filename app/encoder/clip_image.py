from __future__ import annotations
import numpy as np
import torch
import open_clip
from PIL import Image


class ClipImageEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        image_size: int = 224,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()
        self.image_size = image_size

    def encode_pil(self, imgs: list[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            batch = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
            z = self.model.encode_image(batch)
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-9)
        return z.detach().cpu().numpy().astype(np.float32)
