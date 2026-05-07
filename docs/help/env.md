# 环境配置问题

---

## Python / PyTorch

### CUDA可用性检测
```bash
python -c "import torch; print(f'PyTorch:{torch.__version__} CUDA:{torch.cuda.is_available()}')"
```

### 安装指定CUDA版本的PyTorch
```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 虚拟环境
```bash
# 创建
python -m venv venv

# 激活 (Windows)
venv\Scripts\activate

# 激活 (Linux/Mac)
source venv/bin/activate

# 退出
deactivate
```

---

## Conda 环境

### 创建环境
```bash
conda create -n yolo26 python=3.10
conda activate yolo26
```

### 导出/导入环境
```bash
conda env export > environment.yml
conda env create -f environment.yml
```

---

## 训练监控

### TensorBoard
```bash
tensorboard --logdir runs/detect/train --port 6006
# 浏览器打开 http://localhost:6006
```
