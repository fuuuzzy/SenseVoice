# SenseVoice CPU/GPU 使用指南

## 设备选择

SenseVoice 支持在 CPU 和 GPU 上运行。根据你的硬件配置选择合适的设备。

### 性能对比

- **GPU**: 速度快，适合实时应用和大批量处理
- **CPU**: 速度较慢，但无需 CUDA 环境，适合开发测试

## 使用方法

### 1. API 服务

通过环境变量 `SENSEVOICE_DEVICE` 指定设备：

```bash
# 使用 CPU
export SENSEVOICE_DEVICE=cpu
python api.py

# 或者使用 uv
export SENSEVOICE_DEVICE=cpu
uv run python api.py
```

```bash
# 使用 GPU (默认)
export SENSEVOICE_DEVICE=cuda:0
python api.py

# 使用第二块 GPU
export SENSEVOICE_DEVICE=cuda:1
python api.py
```

### 2. Web UI

```bash
# CPU 模式
export SENSEVOICE_DEVICE=cpu
python webui.py

# GPU 模式（需要修改 webui.py 中的 device 参数）
python webui.py
```

### 3. 推理脚本

#### 方法 A: 使用提供的 CPU 示例

```bash
python demo_cpu.py
```

#### 方法 B: 在代码中指定设备

```python
from funasr import AutoModel

model_dir = "iic/SenseVoiceSmall"

# CPU 模式
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    device="cpu"
)

# GPU 模式
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    device="cuda:0"
)
```

### 4. 直接推理

```python
from model import SenseVoiceSmall

model_dir = "iic/SenseVoiceSmall"

# CPU 模式
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cpu")

# GPU 模式
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
```

## 安装对应的 PyTorch 版本

### CPU 版本

```bash
# 使用 uv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或使用 pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### GPU 版本 (CUDA 11.8)

```bash
# 使用 uv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或使用 pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### GPU 版本 (CUDA 12.1)

```bash
# 使用 uv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或使用 pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 验证安装

```python
import torch

# 检查 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 如果 CUDA 可用，显示 CUDA 版本
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 常见问题

### Q: 报错 "AssertionError: Torch not compiled with CUDA enabled"

**A**: 这是因为安装的是 CPU 版本的 PyTorch，但代码尝试使用 GPU。解决方法：

1. **方案 1**: 设置使用 CPU
   ```bash
   export SENSEVOICE_DEVICE=cpu
   ```

2. **方案 2**: 安装 GPU 版本的 PyTorch（需要有 NVIDIA GPU 和 CUDA 环境）
   ```bash
   uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Q: CPU 模式下推理速度很慢怎么办？

**A**: CPU 模式速度确实比 GPU 慢很多。建议：

1. 使用较短的音频（< 30 秒）
2. 减小 batch_size
3. 考虑使用 GPU 或云服务（如 Google Colab）

### Q: 如何在 macOS 上使用 MPS 加速？

**A**: macOS 用户可以使用 Apple Silicon 的 MPS 后端：

```python
import torch

# 检查 MPS 是否可用
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = AutoModel(model=model_dir, device=device)
```

## 快速启动命令

```bash
# 1. 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

# 2. 启动 API 服务（CPU 模式）
export SENSEVOICE_DEVICE=cpu
python api.py

# 3. 或启动 Web UI（CPU 模式）
export SENSEVOICE_DEVICE=cpu
python webui.py

# 4. 或运行示例（CPU 模式）
python demo_cpu.py
```

