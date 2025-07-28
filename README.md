# OpenVLA: Vision-Language-Action Model Implementation

This repository contains a complete implementation of OpenVLA (Open Vision-Language-Action), a 7B-parameter vision-language-action model for robotic manipulation. This implementation is based on the research paper: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246).

## 🚀 Features

- **Complete OpenVLA Architecture**: Implements the full model with fused DINOv2 + SigLIP vision encoder and Llama 2 7B language model
- **Training Scripts**: Full training pipeline with distributed training support
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning for consumer GPUs
- **Inference Engine**: Ready-to-use inference with quantization support
- **Robot Integration**: Easy integration with various robot platforms
- **Memory Efficient**: Support for quantization and mixed precision

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Fine-tuning](#fine-tuning)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Performance](#performance)

## 🏗️ Architecture Overview

OpenVLA consists of three main components:

1. **Vision Encoder**: Fused DINOv2 + SigLIP features for robust visual understanding
2. **Projector**: 2-layer MLP that maps vision features to language embedding space
3. **Language Model**: Llama 2 7B parameter model for action generation

The model takes images and text instructions as input and outputs 7-DoF robot actions through a 256-bin discretization scheme.

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 16GB+ GPU memory (for full model training)
- 8GB+ GPU memory (for LoRA fine-tuning)

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd openvla-implementation

# Create conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

# Install Flash Attention (optional, for faster training)
pip install flash-attn==2.5.5 --no-build-isolation
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from openvla_architecture import OpenVLA, OpenVLAConfig
import torch

# Create model
config = OpenVLAConfig()
model = OpenVLA(config)

# Example inference
images = torch.randn(1, 3, 224, 224)
instructions = ["pick up the red cup"]
actions = model.predict_action(images, instructions)
print(f"Predicted actions: {actions}")
```

### 2. Load Pre-trained Model

```python
from inference_openvla import OpenVLAInference

# Initialize inference
inference = OpenVLAInference(
    model_path="checkpoints/openvla_best.pt",
    device="cuda",
    use_quantization=True
)

# Predict action
action = inference.predict_action(image, "pick up the red cup")
```

## 🎯 Training

### Full Training

To train OpenVLA from scratch on the Open X-Embodiment dataset:

```bash
# Single GPU training
python train_openvla.py

# Multi-GPU training
torchrun --standalone --nnodes 1 --nproc-per-node 8 train_openvla.py
```

### Training Configuration

Key training parameters (from the paper):
- **Learning Rate**: 2e-5 (fixed, no warmup)
- **Batch Size**: 2048
- **Epochs**: 27 (until action accuracy > 95%)
- **Image Resolution**: 224×224
- **Vision Encoder**: Fine-tuned (not frozen)

### Training Data

The model is trained on the Open X-Embodiment dataset with 970k robot demonstrations. You'll need to:

1. Download the Open X-Embodiment dataset
2. Preprocess the data to RLDS format
3. Update the `RobotDataset` class to load your specific data

## 🔧 Fine-tuning

### LoRA Fine-tuning

For parameter-efficient fine-tuning on new tasks:

```bash
python finetune_openvla.py \
    --model_path checkpoints/openvla_best.pt \
    --task_name "pick_and_place" \
    --learning_rate 5e-4 \
    --batch_size 16
```

### Fine-tuning Configuration

- **LoRA Rank**: 32 (recommended)
- **Learning Rate**: 5e-4
- **Batch Size**: 16 (for single A100 GPU)
- **Epochs**: 10-15 (typically sufficient)

### Supported Fine-tuning Tasks

- BridgeData V2 tasks
- Custom robot manipulation tasks
- Multi-object manipulation
- Language-conditioned tasks

## 🔮 Inference

### Basic Inference

```bash
python inference_openvla.py \
    --model_path checkpoints/openvla_best.pt \
    --robot_type widowx \
    --precision bfloat16
```

### Interactive Mode

```bash
python inference_openvla.py \
    --model_path checkpoints/openvla_best.pt \
    --interactive
```

### Quantization Support

For memory-efficient deployment:

```python
inference = OpenVLAInference(
    model_path="checkpoints/openvla_best.pt",
    use_quantization=True,
    precision="bfloat16"
)
```

## 🏗️ Model Architecture

### Vision Encoder

```python
class VisionEncoder(nn.Module):
    def __init__(self, config):
        # DINOv2 encoder
        self.dinov2 = timm.create_model('vit_large_patch14_224.dinov2')
        
        # SigLIP encoder
        self.siglip = timm.create_model('vit_large_patch14_224_clip_laion2b')
        
        # Feature fusion
        self.fusion_layer = nn.Linear(dinov2_dim + siglip_dim, hidden_size)
```

### Action Tokenization

```python
class ActionTokenizer:
    def __init__(self, action_bins=256, action_dim=7):
        # 256 bins per action dimension
        # 7-DoF robot actions
        pass
```

### Language Model Integration

The model uses Llama 2 7B as the language model backbone, with the last 256 tokens of the vocabulary replaced with action tokens.

## 📊 Performance

### Model Specifications

- **Parameters**: 7B
- **Vision Encoder**: Fused DINOv2 + SigLIP
- **Language Model**: Llama 2 7B
- **Action Space**: 7-DoF continuous actions
- **Discretization**: 256 bins per dimension

### Training Performance

- **Training Time**: 14 days on 64 A100 GPUs
- **Memory Usage**: 15GB GPU memory (bfloat16)
- **Inference Speed**: ~6Hz on RTX 4090

### Evaluation Results

OpenVLA achieves:
- **BridgeData V2**: 70.6% success rate
- **Google Robot**: 85.0% success rate
- **Outperforms RT-2-X** by 16.5% absolute success rate

## 🛠️ Customization

### Adding New Robot Platforms

1. Update the `RobotController` class
2. Implement robot-specific action execution
3. Add camera integration for observations

### Custom Datasets

1. Implement your dataset class inheriting from `RobotDataset`
2. Format data as (image, instruction, action) tuples
3. Update data loading in training scripts

### Model Modifications

1. Modify `OpenVLAConfig` for different model sizes
2. Update vision encoder for different backbones
3. Change action discretization scheme

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**: Use LoRA fine-tuning or quantization
2. **Slow Training**: Enable Flash Attention
3. **Poor Performance**: Check data preprocessing and action normalization

### Performance Tips

- Use bfloat16 precision for training
- Enable gradient checkpointing for large models
- Use mixed precision training with AMP
- Implement proper data loading with multiple workers

## 📚 API Reference

### OpenVLA Class

```python
class OpenVLA(nn.Module):
    def __init__(self, config: OpenVLAConfig)
    def forward(self, images, text_tokens, action_tokens=None)
    def predict_action(self, images, text_prompts, **kwargs)
```

### Configuration

```python
class OpenVLAConfig:
    hidden_size: int = 4096
    image_size: int = 224
    action_bins: int = 256
    action_dim: int = 7
    learning_rate: float = 2e-5
    batch_size: int = 2048
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This implementation is released under the MIT License.

## 🙏 Acknowledgments

This implementation is based on the original OpenVLA research paper and builds upon the work of the robotics and AI communities. Special thanks to the authors of the original paper and the contributors to the open-source ecosystem.

---
