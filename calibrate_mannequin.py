"""
calibrate_mannequin.py
------------------------
One-time manual keypoint calibration tool for a mannequin base photo.
Run this once per mannequin photo you plan to reuse (e.g. once for your
male mannequin, once for your female mannequin). Click points in order
when the image appears; result is saved to
models/mannequin_keypoints/<mannequin_id>.json and reused by the
pipeline every time after that.

Local (with a display) usage:
    python calibrate_mannequin.py --image sample_data/mannequin_female.png --id female_default

Colab usage (no GUI window - matplotlib's ginput does not work reliably
against Colab's inline backend, so this module also exposes
`calibrate_colab()`, which captures clicks via an HTML5 canvas instead):

    from calibrate_mannequin import calibrate_colab
    calibrate_colab(mannequin_id="female_default")   # prompts a file upload,
                                                       # then click 6 points
"""

import argparse
import json
import io
import base64

POINT_ORDER = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]
POINT_LABELS_VI = ["Vai trái", "Vai phải", "Hông trái", "Hông phải", "Đầu gối trái", "Đầu gối phải"]


def calibrate(image_path: str, mannequin_id: str):
    """Local/desktop calibration using matplotlib's interactive ginput.
    Requires a real GUI backend (e.g. TkAgg/QtAgg) - will hang or do
    nothing under Colab's default inline backend. Use calibrate_colab()
    instead when running in a notebook."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from pipeline.mannequin_pose import BodyKeypoints, save_manual_keypoints

    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click in order: " + " -> ".join(POINT_ORDER))
    pts = plt.ginput(n=len(POINT_ORDER), timeout=0)
    plt.close(fig)

    kwargs = {name: (float(x), float(y)) for name, (x, y) in zip(POINT_ORDER, pts)}
    keypoints = BodyKeypoints(**kwargs)
    path = save_manual_keypoints(mannequin_id, keypoints)
    print(f"Saved keypoints for '{mannequin_id}' -> {path}")
    return keypoints


def calibrate_colab(mannequin_id: str, image_path: str = None):
    """Colab-native calibration: uploads (if image_path is None) or loads
    an image, draws it on an HTML5 canvas, and waits for exactly 6 clicks
    in POINT_ORDER, then saves the keypoints. Works reliably in any Colab
    session since it only relies on the browser, not a matplotlib GUI
    backend.

    Returns (keypoints, image_path) so the caller can reuse image_path
    later (e.g. as --mannequin_bare for inference).
    """
    from IPython.display import display, HTML
    from google.colab.output import eval_js
    from google.colab import files
    from PIL import Image
    from pipeline.mannequin_pose import BodyKeypoints, save_manual_keypoints

    if image_path is None:
        up = files.upload()
        assert up, "Không có ảnh nào được upload."
        image_path = list(up.keys())[0]

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    display(HTML(f'''
<p id="instr" style="font-weight:bold;">Click: {POINT_LABELS_VI[0]}</p>
<canvas id="c" width="{w}" height="{h}" style="border:1px solid #333; cursor:crosshair; max-width:100%;"></canvas>
<script>
(function() {{
  var c = document.getElementById('c'), ctx = c.getContext('2d');
  var img = new Image();
  img.onload = function() {{ ctx.drawImage(img, 0, 0); }};
  img.src = "data:image/png;base64,{b64}";
  window._vton_points = [];
  var labels = {json.dumps(POINT_LABELS_VI)};
  window._vton_done = new Promise(function(resolve) {{
    c.addEventListener('click', function(e) {{
      var r = c.getBoundingClientRect();
      var x = Math.round((e.clientX - r.left) * (c.width / r.width));
      var y = Math.round((e.clientY - r.top) * (c.height / r.height));
      window._vton_points.push([x, y]);
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 2*Math.PI); ctx.fillStyle = 'red'; ctx.fill();
      ctx.font = "16px Arial"; ctx.fillStyle = "yellow"; ctx.fillText(window._vton_points.length, x+8, y);
      if (window._vton_points.length < labels.length) {{
        document.getElementById('instr').innerText = "Click: " + labels[window._vton_points.length];
      }} else {{
        document.getElementById('instr').innerText = "Xong, đang lưu...";
        resolve(window._vton_points);
      }}
    }});
  }});
}})();
</script>
'''))
    print(f"Ảnh {w}x{h}px — click lần lượt: {' -> '.join(POINT_LABELS_VI)}")

    pts = eval_js("window._vton_done")  # blocks until 6 points are clicked

    kwargs = {name: (float(x), float(y)) for name, (x, y) in zip(POINT_ORDER, pts)}
    keypoints = BodyKeypoints(**kwargs)
    path = save_manual_keypoints(mannequin_id, keypoints)
    print(f"Đã lưu keypoints cho '{mannequin_id}' -> {path}")
    return keypoints, image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--id", required=True, help="e.g. male_default, female_default, unisex_default")
    args = parser.parse_args()
    calibrate(args.image, args.id)
