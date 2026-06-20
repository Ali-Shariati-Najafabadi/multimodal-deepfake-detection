#!/bin/bash
# Installation script for deepfake detection environment
# Execute this in order!

echo "=== Creating new conda environment ==="
conda create -n deepfake python=3.10 -y
conda activate deepfake

echo ""
echo "=== Installing PyTorch (MOST IMPORTANT - do this first!) ==="
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Installing Deep Learning packages ==="
pip install transformers==4.36.0
pip install datasets==4.5.0
pip install pytorchvideo==0.1.5

echo ""
echo "=== Installing Audio Processing ==="
pip install librosa==0.11.0
pip install soundfile==0.13.1
pip install audioread==3.1.0

echo ""
echo "=== Installing Computer Vision ==="
pip install opencv-python==4.13.0.90
pip install pillow==12.0.0
pip install av==16.1.0
pip install kornia==0.8.2

echo ""
echo "=== Installing ML & Data Tools ==="
pip install scikit-learn==1.7.2
pip install scipy==1.15.3
pip install h5py==3.15.1
pip install pandas==2.3.3

echo ""
echo "=== Installing Visualization ==="
pip install matplotlib==3.10.8
pip install seaborn==0.13.2
pip install tensorboard==2.20.0

echo ""
echo "=== Installing Utilities ==="
pip install tqdm==4.67.1
pip install pyyaml==6.0.3
pip install requests==2.32.5
pip install huggingface-hub==0.36.0
pip install safetensors==0.7.0
pip install tokenizers==0.15.2

echo ""
echo "=== Verification ==="
python << 'EOF'
import torch
import torchvision
import torchaudio
import transformers
import pytorchvideo

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ TorchVision: {torchvision.__version__}")
print(f"✓ TorchAudio: {torchaudio.__version__}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")

# Test X3D
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
print(f"✓ X3D-M loaded!")

# Test HuBERT
from transformers import HubertModel
model = HubertModel.from_pretrained('facebook/hubert-base-ls960')
print(f"✓ HuBERT loaded!")

print("\n✓✓✓ INSTALLATION COMPLETE! ✓✓✓")
EOF

echo ""
echo "=== DONE! ==="
echo "To use: conda activate deepfake"
echo "Then set: export HF_HUB_DISABLE_TORCH_LOAD_CHECK=1"
