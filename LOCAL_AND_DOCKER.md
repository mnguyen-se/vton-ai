# Chạy local trên RTX 5070 + đẩy lên GitHub/Docker

## RTX 5070 cần lưu ý gì

RTX 5070 dùng kiến trúc Blackwell (`sm_120`), 12GB VRAM. Đây là dòng GPU rất mới nên:

- **Driver**: cần driver NVIDIA >= 570 (Linux) / Game Ready driver mới nhất (Windows).
- **PyTorch**: bản `sm_120` chính thức chỉ có từ **PyTorch >= 2.7**, dùng wheel `+cu128`
  (CUDA 12.8). Bản PyTorch cũ hơn sẽ chạy được (`torch.cuda.is_available()` trả `True`)
  nhưng báo warning "CUDA capability sm_120 is not compatible" và **âm thầm chạy sai/chậm
  hoặc lỗi kernel** — đây là lỗi hay gặp nhất với card 50-series, không phải lỗi ở code
  pipeline này.
- **xformers**: chưa hỗ trợ `sm_120` (lỗi `capability (12, 0) too new`). Pipeline này
  không dùng xformers nên không bị ảnh hưởng — đừng tự thêm `enable_xformers_memory_efficient_attention()`.
- **VRAM 12GB**: đủ cho train LoRA rank 16 ở resolution 512, batch_size 1 (đúng config
  mặc định trong `train_lora.py`). Nếu vẫn OOM, giảm `--resolution 384` hoặc `--lora_rank 8`.

## Test trước bằng RTX 3050 Ti (4GB VRAM) trong lúc chưa ra net

Tin tốt: RTX 3050 Ti là Ampere (`sm_86`) — GPU đời cũ hơn nhiều so với 5070, nên **không
dính lỗi driver/CUDA/sm_120** như ở trên. Cài PyTorch bản thường (`pip install torch
torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` hoặc cu124 đều
được, không bắt buộc cu128 như 5070) là chạy ngay.

Vấn đề duy nhất là **4GB VRAM rất chật** cho Stable Diffusion inpainting + IP-Adapter.
Mục tiêu lúc này không phải train full mà là **smoke-test**: xác nhận code chạy hết
pipeline không lỗi (đường dẫn, format ảnh, mask, v.v.) trước khi chuyển qua máy 5070 để
train thật.

Inference (đã thêm sẵn cờ `--low_vram` bật attention/VAE slicing + CPU offload):
```bash
python phase2_diffusion/generate.py \
  --lora_path lora_output/final \
  --mannequin_bare raw_data/sample_0001/mannequin_bare.jpg \
  --mannequin_id female_default --garment_type top --sleeve_length short \
  --garment raw_data/sample_0001/garment.png \
  --n_variants 1 --low_vram --out_dir results_smoketest/
```
(`--n_variants 1` để đỡ chờ lâu vì CPU offload chạy chậm hơn nhiều so với để full GPU.)

Train — chỉ chạy thử vài epoch, resolution/rank nhỏ, KHÔNG dùng để lấy checkpoint thật:
```bash
accelerate launch phase2_diffusion/train_lora.py \
  --data_dir data/train --output_dir lora_output_smoketest \
  --resolution 384 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --lora_rank 8 --num_train_epochs 2 --save_every 1
```
Nếu vẫn OOM ở bước train (4GB thực sự sát nút với UNet fp16 + activations), đó là do
phần cứng chứ không phải lỗi code — cứ để việc train thật đợi tới lúc dùng 5070. Mục tiêu
ở bước này chỉ là thấy log chạy qua vài step không crash là đủ tin code ổn.


## Cách 1 — Docker (khuyên dùng, vì dễ mang đi/host ở máy khác)

Cần cài trước trên máy host (Linux hoặc WSL2 trên Windows):
1. Docker + Docker Compose
2. NVIDIA driver >= 570
3. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   để container thấy được GPU

```bash
git clone <repo-của-bạn>
cd vton_pipeline

docker compose build
docker compose run --rm vton bash
```

Trong container, kiểm tra GPU trước:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Nếu in ra `True NVIDIA GeForce RTX 5070` là ổn.

Chạy pipeline (giống hệt notebook Colab, chỉ là chạy trực tiếp bằng CLI):
```bash
# 1. Chuẩn bị dataset (raw_data/ mount từ host, xem docker-compose.yml)
python phase2_diffusion/prepare_dataset.py --raw_dir raw_data --out_dir data/train --image_size 512

# 2. Train LoRA
accelerate launch phase2_diffusion/train_lora.py \
  --data_dir data/train --output_dir lora_output \
  --resolution 512 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 30 --learning_rate 1e-4 --save_every 10

# 3. Inference
python phase2_diffusion/generate.py \
  --lora_path lora_output/final \
  --mannequin_bare raw_data/sample_0001/mannequin_bare.jpg \
  --mannequin_id female_default --garment_type top --sleeve_length short \
  --garment raw_data/sample_0001/garment.png \
  --n_variants 3 --out_dir results/
```

`raw_data/`, `data/`, `lora_output/`, `results/`, `models/` được mount làm volume
(xem `docker-compose.yml`) nên dữ liệu **không mất khi container bị xoá** — đóng vai trò
giống Google Drive trong bản Colab.

## Cách 2 — venv thuần trên máy (không Docker)

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Cài torch riêng với đúng wheel cu128 cho RTX 5070 TRƯỚC - đừng để requirements.txt
# kéo bản torch mặc định (thường quá cũ, không có sm_120).
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install diffusers transformers accelerate peft rembg onnxruntime
```

Kiểm tra GPU giống bước trên (`torch.cuda.get_device_name(0)`).

### Bước hiệu chỉnh mannequin (calibrate) — nên chạy NGOÀI Docker

Bước click 6 điểm mốc trên ảnh mannequin cần cửa sổ hiển thị (GUI), mà container Docker
thường không có màn hình. Vì vậy:

```bash
# chạy ngay trên máy (venv ở Cách 2), KHÔNG cần GPU cho bước này
python calibrate_mannequin.py --image duong_dan_anh_mannequin.jpg --id female_default
```

Kết quả lưu vào `models/mannequin_keypoints/female_default.json` — chỉ cần làm 1 lần,
sau đó file JSON này dùng chung cho cả venv lẫn Docker (mount `models/` như trong
`docker-compose.yml`).

## Đẩy lên GitHub

- `.gitignore` đã loại `raw_data/`, `data/`, `lora_output/`, `results/` và cache HF —
  đây là dữ liệu/checkpoint nặng và có thể là ảnh sản phẩm riêng, không nên public.
- Checkpoint LoRA cuối (`lora_output/final/`) nếu muốn chia sẻ thì nên up lên
  Hugging Face Hub hoặc GitHub Releases (không commit thẳng vào git, dễ vượt giới hạn
  100MB/file của GitHub) — LoRA rank 16 cỡ ~13MB nên vẫn tạm ổn nếu bạn thật sự muốn
  commit, nhưng Releases vẫn sạch hơn về lâu dài.
- Model nền (Stable Diffusion inpainting, IP-Adapter) **không** đóng gói theo repo —
  chúng tự tải từ Hugging Face vào lần chạy đầu tiên (`HF_HOME` cache, xem Dockerfile).
