"""
demo.py
--------
End-to-end smoke test / usage example. Generates 3 outfit variants
from a shirt + pants image on a mannequin, using the synthetic sample
data. Replace sample_data/*.png with your real product & mannequin
photos to test with real data.
"""
from PIL import Image
from pipeline.orchestrator import generate_outfits

top = Image.open("sample_data/shirt_sample.png")
bottom = Image.open("sample_data/pants_sample.png")
mannequin = Image.open("sample_data/mannequin_unisex.png")

result = generate_outfits(
    top_image=top,
    bottom_image=bottom,
    mannequin_photos={"unisex": mannequin},
    product_metadata={},   # no gender field -> will fall back to "unisex"
    n_variants=3,
)

print(f"Gender used: {result.gender_used}, mannequin: {result.mannequin_id}")

import os
os.makedirs("outputs", exist_ok=True)
for i, img in enumerate(result.images):
    out_path = f"outputs/outfit_{i+1}.png"
    img.save(out_path)
    print(f"Saved {out_path}")
