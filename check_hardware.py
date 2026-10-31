import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import platform
import psutil
import torch
import torch.nn as nn

print("=" * 70)
print("1. ข้อมูลระบบและ CPU (System & CPU Info)")
print("=" * 70)
print(f"OS / Platform     : {platform.system()} {platform.release()} ({platform.version()})")
print(f"Processor         : {platform.processor()}")
print(f"Physical Cores    : {psutil.cpu_count(logical=False)} Cores")
print(f"Logical Threads   : {psutil.cpu_count(logical=True)} Threads")
print(f"Total RAM         : {psutil.virtual_memory().total / (1024**3):.2f} GB")

print("\n" + "=" * 70)
print("2. สถานะ PyTorch และ GPU (CUDA Diagnostic)")
print("=" * 70)
print(f"PyTorch Version   : {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available    : {cuda_avail}")

if cuda_avail:
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Device Name   : {device_name}")
    print(f"GPU VRAM          : {vram_gb:.2f} GB")
    print(f"CUDA Version      : {torch.version.cuda}")
else:
    print("WARNING           : PyTorch กำลังรันในโหมด CPU-Only! (ไม่ได้ใช้การ์ดจอแยก)")

print("\n" + "=" * 70)
print("3. ทดสอบความเร็วจำลอง 2D-CNN GIAB Pileup (Batch=128, Shape=4x50x101)")
print("=" * 70)
x_dummy = torch.randn(128, 4, 50, 101)
model = nn.Sequential(
    nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(6 * 12 * 128, 256), nn.ReLU(), nn.Linear(256, 1)
)

# ทดสอบบน CPU
t0 = time.perf_counter()
for _ in range(10):
    out = model(x_dummy)
    out.sum().backward()
cpu_time = (time.perf_counter() - t0) / 10
print(f"เวลาเฉลี่ยต่อ 1 Step บน CPU  : {cpu_time*1000:.2f} ms")

# ทดสอบบน GPU (ถ้ามี)
if cuda_avail:
    model_gpu = model.cuda()
    x_gpu = x_dummy.cuda()
    # Warm-up
    for _ in range(3):
        model_gpu(x_gpu).sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = model_gpu(x_gpu)
        out.sum().backward()
    torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - t0) / 10
    print(f"เวลาเฉลี่ยต่อ 1 Step บน GPU  : {gpu_time*1000:.2f} ms (เร็วขึ้น {cpu_time/gpu_time:.1f} เท่า)")