"""
gender.py
----------
Classifies a product as male/female (for selecting which mannequin to use).

Priority order:
  1. If your Product/WardrobeItem DB already has a gender field -> use it
     directly (no ML needed, most reliable).
  2. Otherwise fall back to CLIP zero-shot classification (pretrained,
     no training data required to get started).
  3. Once you've logged enough (product -> gender) pairs from real usage,
     swap this for a small fine-tuned classifier (see train_gender.py).
"""

from __future__ import annotations
from PIL import Image


def gender_from_metadata(product_metadata: dict) -> str | None:
    """
    Cheapest and most reliable path: use existing DB field if present.
    Expected keys tried in order: 'gender', 'category_gender', 'target_gender'
    """
    for key in ("gender", "category_gender", "target_gender"):
        val = product_metadata.get(key)
        if val:
            val = str(val).lower()
            if val in ("male", "men", "m", "man"):
                return "male"
            if val in ("female", "women", "f", "woman"):
                return "female"
    return None


class ClipGenderClassifier:
    """
    Zero-shot fallback using CLIP. No training needed to start; accuracy
    is decent for clearly gendered product photos but not perfect -
    treat as a fallback, not the primary signal.
    """

    def __init__(self, device: str = "cpu"):
        import torch
        import clip  # pip install git+https://github.com/openai/CLIP.git
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.prompts = [
            "a photo of men's clothing",
            "a photo of women's clothing",
        ]
        import torch as _torch
        text = clip.tokenize(self.prompts).to(device)
        with _torch.no_grad():
            self.text_features = self.model.encode_text(text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image: Image.Image) -> tuple[str, float]:
        import torch
        img = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.model.encode_image(img)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            sims = (img_features @ self.text_features.T).softmax(dim=-1)[0]
        idx = int(sims.argmax())
        label = "male" if idx == 0 else "female"
        return label, float(sims[idx])


def classify_gender(product_metadata: dict, image: Image.Image, clip_classifier: ClipGenderClassifier | None = None) -> str:
    """
    Main entry point. Falls back through the priority chain described above.
    Defaults to 'unisex' mannequin set if nothing is confident enough -
    make sure you have a neutral mannequin available for that case.
    """
    meta_result = gender_from_metadata(product_metadata)
    if meta_result:
        return meta_result

    if clip_classifier is not None:
        label, confidence = clip_classifier.predict(image)
        if confidence >= 0.6:
            return label

    return "unisex"
