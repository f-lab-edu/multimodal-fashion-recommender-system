# smoke_clip.py
from clip_text import ClipTextEncoder
import numpy as np

enc = ClipTextEncoder(model_name="ViT-B-32", pretrained="openai")

q1 = "pink silk skirt summer"
q2 = "brown leather boots winter"
q3 = "graph theory lemma"

emb = enc.encode_text([q1, q2, q3])  # (3, D)
print("shape:", emb.shape)

# L2 정규화 확인 (≈1.0)
print("norms:", np.linalg.norm(emb, axis=1))

# 코사인 유사도 매트릭스
S = emb @ emb.T
print("cosine:\n", np.round(S, 3))
