FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 把 apt 源换成阿里云,国内构建更快更稳
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list

# Ubuntu 22.04 默认有 python3.10,直接装
# (lesson 写的是 3.12,但 nvidia/cuda 12.4.1-devel 没有 ubuntu24.04 tag,
#  python3.10 对 PyTorch/transformers 全兼容,无影响)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/

# PyTorch 走官方 CUDA 源(国内下载也 OK)
# 注意:torch 2.3.x 在 cu124 索引里没有,cu124 最早是 2.4.0+
# lesson 原文写 2.3.1,这里改成 2.4.1(对 RTX 40 系 / CUDA 12.x 更合适)
RUN python -m pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# AI 库走阿里云 PyPI 镜像
RUN python -m pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    safetensors

WORKDIR /workspace

VOLUME ["/workspace", "/models"]

EXPOSE 8888

CMD ["python"]
