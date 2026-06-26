# GPU 配置与云端训练

> 学习阶段用 CPU 跑就行。真要训模型,就得用 GPU。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Learning Objectives

- 用 `nvidia-smi` 和 PyTorch 的 CUDA API 检查本地 GPU 是否可用
- 配置 Google Colab,免费拿到一块 T4 GPU 做云端实验
- 跑一个 CPU vs GPU 的矩阵乘法 benchmark,测出加速比
- 用 fp16 经验法则估算你的 VRAM 最多能塞下多大的模型

## The Problem

Phase 1–3 的大部分章节 CPU 就能跑。但一旦开始训 CNN、Transformer、LLM(Phase 4+),你就要 GPU 加持。CPU 上要跑 8 小时的训练,GPU 上 10 分钟就完事。

你有三个选择:本地 GPU、云端 GPU、Google Colab(免费)。

## The Concept

```
你的选项:

1. 本地 NVIDIA GPU
   成本: $0(你本来就有)
   配置: 装 CUDA + cuDNN
   适用: 日常使用、大数据集

2. Google Colab(免费版)
   成本: $0
   配置: 无
   适用: 快速实验、家里没 GPU

3. 云 GPU(Lambda、RunPod、Vast.ai)
   成本: $0.20-2.00/hr
   配置: SSH + 安装
   适用: 严肃训练、大模型
```

## Build It

### Option 1: Local NVIDIA GPU

先看看你有没有:

```bash
nvidia-smi
```

装带 CUDA 的 PyTorch:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 跑 `!nvidia-smi` 验证一下

直接把本课程的 notebook 上传到 Colab 就能用。

### Option 3: Cloud GPU

以 Lambda Labs、RunPod、Vast.ai 为例:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### No GPU? No problem.

大部分章节 CPU 都能跑。需要 GPU 的章节会标注清楚,并附上 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: GPU vs CPU benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Exercises

1. 跑一遍上面的 benchmark,对比 CPU 和 GPU 的耗时
2. 没 GPU 的话,在 Google Colab 上跑一下并对比
3. 查一下你的 GPU 有多少显存,用经验法则(每个参数 fp16 占 2 字节)估算能塞下多大的模型

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU 编程" | NVIDIA 的并行计算平台,让你的代码可以跑在 GPU 上 |
| VRAM | "GPU 显存" | GPU 上的视频内存,跟系统内存是分开的,直接限制模型规模 |
| fp16 | "半精度" | 16 位浮点,显存占用是 fp32 的一半,精度损失很小 |
| Tensor Core | "专门的矩阵硬件" | GPU 里专门做矩阵乘法的核心,比普通核心快 4–8 倍 |
