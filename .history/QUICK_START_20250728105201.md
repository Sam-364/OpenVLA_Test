# 🚀 OpenVLA Training - Quick Start Guide

This guide will help you train the OpenVLA model on high-quality datasets with real-time performance monitoring.

## 📋 Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, etc.)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space for datasets
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Software Requirements
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

## 🛠️ Installation

### 1. Clone and Setup
```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd OpenVLA_Test

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Check system requirements
python train_openvla_complete.py --skip_checks

# Should show:
# ✓ CUDA available: 1 GPU(s)
# ✓ All required packages installed
```

## 📊 Available Datasets

### 1. **BridgeData V2** (Recommended)
- **Size**: 45GB
- **Trajectories**: 60,096
- **Tasks**: 13 skills across 24 environments
- **Quality**: High-quality teleoperated demonstrations

### 2. **RLDS (Robot Learning Data Store)**
- **Size**: 20GB
- **Format**: Standardized robot learning datasets
- **Features**: Multi-robot, multi-task

### 3. **Open X-Embodiment**
- **Size**: 100GB
- **Trajectories**: 1M+
- **Scope**: Multi-robot, large-scale

### 4. **RoboSuite**
- **Size**: 5GB
- **Type**: Simulation environment with demos
- **Use**: Good for testing and development

## 🎯 Quick Start Training

### Option 1: Complete Pipeline (Recommended)
```bash
# Download BridgeData V2 and start training with real-time monitoring
python train_openvla_complete.py --dataset bridgedata_v2

# This will:
# 1. Check system requirements
# 2. Download BridgeData V2 (45GB)
# 3. Setup training environment
# 4. Start training with real-time monitoring
# 5. Save checkpoints and generate reports
```

### Option 2: Download Dataset Only
```bash
# Download dataset first
python train_openvla_complete.py --dataset bridgedata_v2 --download_only

# Then train later
python train_openvla_complete.py --dataset bridgedata_v2
```

### Option 3: Custom Configuration
```bash
# Use custom config
python train_openvla_complete.py \
    --dataset bridgedata_v2 \
    --config configs/my_config.json \
    --epochs 50 \
    --device cuda
```

## 📈 Real-Time Monitoring

### Live Performance Dashboard
The training script includes real-time monitoring with:
- **CPU Usage**: Real-time CPU utilization
- **Memory Usage**: System and GPU memory
- **GPU Utilization**: GPU usage and temperature
- **Training Metrics**: Loss, accuracy, learning rate
- **Live Plots**: 6-panel dashboard updating every second

### Monitor System Resources
```bash
# Start monitoring only (without training)
python realtime_monitor.py --save_metrics

# Generate report from saved metrics
python realtime_monitor.py --report monitoring_logs/metrics_20241201_143022.json
```

## 🔧 Advanced Usage

### Dataset Management
```bash
# List available datasets
python download_datasets.py --list

# Download specific dataset
python download_datasets.py --dataset bridgedata_v2

# Check downloaded datasets
python download_datasets.py --check

# Create training config for dataset
python download_datasets.py --create_config bridgedata_v2
```

### Training with Different Datasets
```bash
# Train on RLDS
python train_openvla_complete.py --dataset rlds

# Train on Open X-Embodiment
python train_openvla_complete.py --dataset open_x_embodiment

# Train on RoboSuite
python train_openvla_complete.py --dataset robosuite
```

### Fine-tuning
```bash
# Fine-tune existing model
python finetune_openvla.py \
    --base_model checkpoints/openvla_best.pt \
    --dataset bridgedata_v2 \
    --epochs 10
```

### Inference Testing
```bash
# Test trained model
python train_openvla_complete.py \
    --test_only \
    --model_path checkpoints/openvla_best.pt
```

## 📊 Expected Performance

### Training Time
- **BridgeData V2**: ~24-48 hours on RTX 3090
- **RLDS**: ~12-24 hours on RTX 3090
- **Open X-Embodiment**: ~72-96 hours on RTX 3090

### Memory Usage
- **GPU Memory**: 12-16GB during training
- **System Memory**: 24-32GB
- **Storage**: 100-200GB for datasets + checkpoints

### Convergence
- **Loss**: Should decrease from ~4.0 to ~1.0
- **Action Accuracy**: Should reach >90% by epoch 20
- **Early Stopping**: Training stops when accuracy >95%

## 🔍 Monitoring and Debugging

### Check Training Progress
```bash
# View real-time logs
tail -f logs/training.log

# Check GPU usage
nvidia-smi

# Monitor system resources
htop
```

### Common Issues and Solutions

#### 1. Out of Memory (OOM)
```bash
# Reduce batch size in config
# Edit configs/training_config.json
{
    "training": {
        "batch_size": 16  # Reduce from 32
    }
}
```

#### 2. Slow Training
```bash
# Enable mixed precision
# Add to training config
{
    "training": {
        "use_amp": true
    }
}
```

#### 3. Dataset Download Issues
```bash
# Manual download
wget https://rail.eecs.berkeley.edu/datasets/bridge_release/data/teleop_data.zip
unzip teleop_data.zip -d data/bridgedata_v2/
```

## 📁 Output Files

After training, you'll find:

```
OpenVLA_Test/
├── checkpoints/
│   ├── openvla_best.pt          # Best model
│   └── openvla_latest.pt        # Latest checkpoint
├── monitoring_logs/
│   └── metrics_*.json           # Performance metrics
├── results/
│   └── training_report.json     # Training summary
├── logs/
│   └── training.log             # Training logs
└── wandb/                       # Weights & Biases logs
```

## 🎯 Performance Tips

### 1. **Optimize for Speed**
```bash
# Use mixed precision training
export CUDA_LAUNCH_BLOCKING=1

# Use gradient checkpointing
# Add to model config
{
    "model": {
        "use_gradient_checkpointing": true
    }
}
```

### 2. **Optimize for Memory**
```bash
# Reduce batch size
# Use gradient accumulation
# Enable CPU offloading
```

### 3. **Monitor Performance**
```bash
# Use real-time monitoring
python realtime_monitor.py --interval 1

# Check for bottlenecks
python -m torch.utils.bottleneck train_openvla_complete.py
```

## 🔬 Advanced Configuration

### Custom Model Configuration
```json
{
    "model": {
        "hidden_size": 4096,
        "vision_hidden_size": 768,
        "action_bins": 256,
        "action_dim": 7,
        "use_gradient_checkpointing": true,
        "use_amp": true
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 27,
        "gradient_clip_val": 1.0,
        "warmup_steps": 1000,
        "scheduler": "cosine"
    }
}
```

### Multi-GPU Training
```bash
# Use DistributedDataParallel
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train_openvla_complete.py \
    --dataset bridgedata_v2
```

## 📞 Support

### Getting Help
1. Check the logs in `logs/training.log`
2. Review monitoring metrics in `monitoring_logs/`
3. Check system requirements with `python train_openvla_complete.py --skip_checks`

### Common Commands Reference
```bash
# Quick start
python train_openvla_complete.py --dataset bridgedata_v2

# Download only
python download_datasets.py --dataset bridgedata_v2

# Monitor only
python realtime_monitor.py --save_metrics

# Test model
python inference_openvla.py --model_path checkpoints/openvla_best.pt

# Evaluate model
python evaluate_openvla.py --model_path checkpoints/openvla_best.pt
```

## 🎉 Success Indicators

Your training is successful when you see:
- ✅ Loss decreasing consistently
- ✅ Action accuracy >90%
- ✅ GPU utilization >80%
- ✅ No OOM errors
- ✅ Checkpoints saved regularly
- ✅ Real-time monitoring working

Happy training! 🚀 