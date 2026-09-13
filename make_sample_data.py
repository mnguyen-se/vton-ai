"""
make_sample_data.py
----------------------
Generates simple synthetic test images (mannequin silhouette + garment
shapes) purely so we can smoke-test that the pipeline runs end-to-end
without errors, BEFORE plugging in real product/mannequin photos.

Not meant to produce realistic output - just validates the code path.
"""
import os
from PIL import Image, ImageDraw

os.makedirs("sample_data", exist_ok=True)


def make_mannequin(path, w=400, h=700, color=(230, 220, 210)):
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    # crude body silhouette: head, torso, legs
    draw.ellipse((w//2 - 30, 40, w//2 + 30, 100), fill=color)          # head
    draw.polygon([(w//2-60, 100), (w//2+60, 100), (w//2+50, 380), (w//2-50, 380)], fill=color)  # torso
    draw.rectangle((w//2-45, 380, w//2-10, 650), fill=color)           # left leg
    draw.rectangle((w//2+10, 380, w//2+45, 650), fill=color)           # right leg
    img.save(path)
    return img


def make_shirt(path, w=300, h=250):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([(40, 20), (w-40, 20), (w-20, h-20), (20, h-20)], fill=(60, 120, 200, 255))
    img.save(path)


def make_pants(path, w=260, h=300):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([(20, 10), (w-20, 10), (w-10, h-10), (w//2+5, h-10),
                  (w//2, 150), (w//2-5, h-10), (10, h-10)], fill=(40, 40, 40, 255))
    img.save(path)


if __name__ == "__main__":
    mannequin_img = make_mannequin("sample_data/mannequin_unisex.png")
    make_shirt("sample_data/shirt_sample.png")
    make_pants("sample_data/pants_sample.png")
    print("Sample data created in sample_data/")

    # Also auto-generate approximate keypoints for this synthetic mannequin
    # (skips manual clicking since geometry is known/programmatic here).
    from pipeline.mannequin_pose import BodyKeypoints, save_manual_keypoints
    w, h = 400, 700
    kp = BodyKeypoints(
        left_shoulder=(w//2 - 55, 105),
        right_shoulder=(w//2 + 55, 105),
        left_hip=(w//2 - 45, 375),
        right_hip=(w//2 + 45, 375),
        left_knee=(w//2 - 30, 520),
        right_knee=(w//2 + 30, 520),
    )
    path = save_manual_keypoints("unisex_default", kp)
    print(f"Sample keypoints saved -> {path}")
