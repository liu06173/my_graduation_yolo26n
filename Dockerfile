# Cloud Studio / Docker GPU 环境
# 构建: docker build -t yolo26-uav .
# 运行: docker run --gpus all -it yolo26-uav

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git wget curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda init bash

# 环境安装
COPY environment.yml /workspace/environment.yml
RUN conda env create -f /workspace/environment.yml && \
    conda clean -a -y

# 激活环境
RUN echo "conda activate yolo26_uav" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /workspace

# 启动时运行setup
COPY . /workspace
CMD ["/bin/bash"]
