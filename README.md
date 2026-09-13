# Virtual Mannequin Outfit Generator — Baseline Pipeline

Ghép ảnh áo (top) + quần (bottom) lên mannequin, tự động chọn mannequin theo
giới tính, sinh ra 3 biến thể outfit. Đây là **baseline chạy được ngay,
không cần train gì** — dùng segmentation + TPS warp + compositing cổ điển.
Đã test end-to-end với dữ liệu giả trong `sample_data/`.

## Đã test chạy được (trong môi trường dev)
```
python make_sample_data.py   # tạo mannequin + áo/quần giả để test
python demo.py                # chạy full pipeline, xuất outputs/outfit_1.png .. 3.png
```
Kết quả: áo được đặt & warp nhẹ vào vùng thân trên, quần vào vùng thân dưới,
3 ảnh có variation nhẹ (brightness/shift) để không trùng pixel.

## Cấu trúc project
```
pipeline/
  segmentation.py      # tách nền sản phẩm (rembg, pretrained, local)
  mannequin_pose.py     # keypoints mannequin (MANUAL calibration hoặc AUTO qua MediaPipe)
  warp.py                # TPS warp garment theo body shape
  compose.py             # blend layers + tạo variation cho 3 outfit
  gender.py              # phân loại nam/nữ (metadata -> CLIP zero-shot fallback)
  orchestrator.py        # ghép toàn bộ pipeline lại, hàm chính generate_outfits()
calibrate_mannequin.py   # tool click tay để lấy keypoints cho 1 ảnh mannequin (chạy 1 lần/mannequin)
make_sample_data.py       # tạo data giả để smoke-test
demo.py                    # ví dụ chạy full pipeline
train_gender_classifier.py # (Phase 2) fine-tune classifier riêng khi có đủ data thật
```

## Chạy với ảnh thật của bạn (làm trên Colab hoặc local có GPU)

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị mannequin
- Có ảnh mannequin nam, nữ (và/hoặc unisex) ở tư thế đứng thẳng, chụp thẳng mặt
- Lấy keypoints cho từng ảnh (chỉ làm 1 lần mỗi mannequin, tái sử dụng mãi):
```bash
python calibrate_mannequin.py --image path/to/mannequin_male.jpg --id male_default
python calibrate_mannequin.py --image path/to/mannequin_female.jpg --id female_default
```
Click theo thứ tự: vai trái -> vai phải -> hông trái -> hông phải -> đầu gối trái -> đầu gối phải.

> Lưu ý quan trọng: MediaPipe Pose (chế độ AUTO trong `mannequin_pose.py`) được train trên
> ảnh người thật, nên với mannequin trừu tượng/không có mặt độ chính xác sẽ không ổn định.
> Vì số lượng mannequin nền của bạn chắc chắn cố định và ít, calibrate tay 1 lần/mannequin
> là cách đáng tin cậy hơn nhiều so với auto-detect mỗi lần.

### 3. Gọi pipeline
```python
from PIL import Image
from pipeline.orchestrator import generate_outfits

top = Image.open("shirt.jpg")
bottom = Image.open("pants.jpg")

result = generate_outfits(
    top_image=top,
    bottom_image=bottom,
    mannequin_photos={
        "male": Image.open("mannequin_male.jpg"),
        "female": Image.open("mannequin_female.jpg"),
    },
    product_metadata={"gender": "male"},  # dùng field có sẵn trong DB nếu có -> chính xác nhất
    n_variants=3,
)

for i, img in enumerate(result.images):
    img.save(f"outfit_{i+1}.png")
```

Nếu `product_metadata` không có field gender, pipeline fallback sang CLIP zero-shot
(cần cài thêm `pip install git+https://github.com/openai/CLIP.git torch`), rồi cuối
cùng fallback "unisex" nếu vẫn không chắc — đảm bảo luôn có mannequin "unisex" dự phòng.

## Giới hạn của baseline này (quan trọng, đọc trước khi đánh giá chất lượng)
- TPS warp là biến dạng hình học đơn giản, KHÔNG học được nếp gấp vải, bóng đổ,
  hay cách vải thật sự ôm theo form 3D — kết quả sẽ trông "dán ảnh" chứ chưa tự nhiên
- Không xử lý occlusion (ví dụ áo che một phần quần ở thắt lưng) tinh vi — hiện chỉ
  overlay đơn giản (bottom trước, top sau)
- Đây là bước 1 để có pipeline chạy được, đo baseline, và sinh dữ liệu training

## Roadmap nâng cấp chất lượng (Phase 2 — cần GPU, có thể làm trên Colab)
1. **Thu thập data thật**: dùng chính baseline này (hoặc chụp thủ công) để tạo
   cặp (mannequin trống, garment, mannequin đã mặc) làm ground truth
2. **Fine-tune diffusion inpainting model** (OOTDiffusion / IDM-VTON, pretrained, tải
   weight về chạy local) bằng LoRA trên data thu thập được — thay thế toàn bộ
   `warp.py` + `compose.py` bằng 1 lần gọi model inference, giữ nguyên phần
   segmentation/gender/orchestrator
3. **Gender classifier riêng**: khi đã log đủ (ảnh -> gender) từ thực tế, chạy
   `train_gender_classifier.py` để thay CLIP zero-shot bằng model fine-tune,
   chính xác hơn và không cần tải CLIP nặng mỗi lần

## Ghi chú GPU
- Baseline này (segmentation + TPS + blend) chạy tốt kể cả CPU, không cần GPU mạnh
- Phase 2 (fine-tune diffusion) nên chạy trên Colab GPU (T4/A100), dùng LoRA +
  resolution 512x512 nếu GPU giới hạn VRAM

---

# Phase 2 — Fine-tune diffusion generator (chất lượng cao hơn, cần GPU)

Nằm trong thư mục `phase2_diffusion/`. Kiến trúc:
- **Base model**: `runwayml/stable-diffusion-inpainting` (pretrained, tải 1 lần từ
  HuggingFace về máy/Colab, sau đó chạy hoàn toàn local — không gọi API mỗi lần dùng)
- **Garment conditioning**: IP-Adapter (`h94/IP-Adapter`, pretrained) — biến ảnh sản
  phẩm áo/quần thành embedding điều kiện, thay vì chỉ dùng text prompt
- **Fine-tune**: chỉ train LoRA trên UNet attention layers (VAE, text encoder, IP-Adapter
  encoder giữ nguyên/đóng băng) — đây là lý do fine-tune được trên 1 GPU Colab thay vì
  cần train from scratch

**Cách chạy nhanh nhất**: mở `VTON_Colab_Finetune.ipynb` trong Google Colab, chạy lần
lượt từng cell (đã dùng GPU runtime). Notebook làm: cài đặt -> mount Drive -> upload
code -> calibrate mannequin -> chuẩn bị dataset -> train LoRA -> inference thử 3 outfit.

## Chuẩn bị dataset training
```
raw_data/
  sample_0001/
    mannequin_bare.jpg       # mannequin KHÔNG mặc đồ (hoặc mặc đồ khác)
    mannequin_wearing.jpg    # CÙNG mannequin, cùng góc chụp, ĐANG mặc garment này
    garment.jpg               # ảnh sản phẩm garment
  sample_0002/
    ...
```
```bash
python phase2_diffusion/prepare_dataset.py --raw_dir raw_data --out_dir data/train
```
Script tự tách nền garment (dùng lại `pipeline/segmentation.py`), tự tính mask vùng
thay đổi giữa 2 ảnh bare/wearing (hoặc dùng `body_region.json` nếu bạn cung cấp box thủ công).

> **Không có đủ ảnh thật (bare + wearing) cho mọi garment?** Dùng chính pipeline Phase 1
> để tạo `mannequin_wearing.jpg` giả (chạy `demo.py`-style trên garment đó) làm data
> bootstrap ban đầu — không đẹp bằng ảnh thật nhưng dạy model được garment identity.
> Trộn thêm càng nhiều ảnh thật càng tốt để model học được độ chân thực (nếp vải, bóng đổ).

## Train
```bash
accelerate launch phase2_diffusion/train_lora.py \
    --data_dir data/train \
    --output_dir lora_output \
    --resolution 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 30 \
    --learning_rate 1e-4
```
Checkpoint LoRA (nhẹ, vài chục MB) lưu vào `lora_output/checkpoint-epochN/` và `lora_output/final/`.

## Inference (sinh 3 outfit)
```bash
python phase2_diffusion/generate.py \
    --lora_path lora_output/final \
    --mannequin_bare mannequin_female.jpg \
    --mannequin_id female_default --garment_type top \
    --garment shirt.jpg \
    --n_variants 3 \
    --out_dir results/
```
`--mannequin_id` + `--garment_type` tái sử dụng chính keypoints bạn đã calibrate ở
Phase 1 (`calibrate_mannequin.py`) để tự tạo mask vùng cần inpaint — không cần vẽ mask tay.

## Đã test/verify trong môi trường dev (không có GPU + không tải được weight HuggingFace ở đây)
- `prepare_dataset.py`: chạy thành công trên data giả, mask tự sinh đúng vùng thân trên
- `VtonDataset` (trong `train_lora.py`): load đúng shape tensor (target 3x512x512,
  mask 1x512x512, garment 3x224x224 cho CLIP encoder)
- `build_mask_from_keypoints` (trong `generate.py`): sinh mask đúng từ keypoints Phase 1
- Toàn bộ script qua được `py_compile`, API calls (`load_ip_adapter`, `LoraConfig`,
  `unet.add_adapter`) đã đối chiếu đúng chữ ký với `diffusers==0.40`, `peft==0.20`
- **Chưa test được**: phần tải model pretrained thật (`runwayml/stable-diffusion-inpainting`,
  `h94/IP-Adapter`) và 1 vòng train/inference thật, vì môi trường dev này không truy cập
  được huggingface.co. Bạn cần chạy thử trên Colab để xác nhận đoạn tải weight + 1 epoch
  training chạy trơn tru — nếu diffusers có thay đổi API nhỏ ở version mới hơn, báo lại
  lỗi cụ thể để chỉnh.

## Giới hạn / lưu ý
- LoRA rank mặc định 16 — tăng lên (32/64) nếu underfitting (garment identity không rõ),
  giảm nếu overfitting với ít data
- 30 epochs là điểm khởi đầu hợp lý cho vài trăm sample — theo dõi `epoch_loss` giảm dần,
  không giảm nữa thì dừng sớm
- Batch size 1 + gradient_accumulation_steps 4 là an toàn cho GPU 16GB trở xuống; tăng
  batch size nếu Colab cấp GPU VRAM lớn hơn (A100 40GB)
