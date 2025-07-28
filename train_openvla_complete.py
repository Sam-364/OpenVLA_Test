#!/usr/bin/env python3
"""
Complete OpenVLA Training Script
Integrates dataset downloading, real-time monitoring, and training
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, Any
import torch
import wandb

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openvla_architecture import OpenVLA, OpenVLAConfig
from data_loader import create_dataloader
from config import get_training_config
from download_datasets import DatasetDownloader
from realtime_monitor import RealTimeMonitor
from train_with_realtime_monitoring import OpenVLARealTimeTrainer

def check_system_requirements():
    """Check if system meets requirements for OpenVLA training"""
    print("Checking system requirements...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA available: {gpu_count} GPU(s)")
        print(f"  GPU: {gpu_name}")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 16:
            print("⚠️  Warning: GPU memory < 16GB. Training may be slow or fail.")
    else:
        print("✗ CUDA not available. Training will be very slow on CPU.")
        return False
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1024**3
    print(f"✓ Free disk space: {free_gb:.1f} GB")
    
    if free_gb < 100:
        print("⚠️  Warning: Less than 100GB free space. Dataset download may fail.")
    
    # Check Python packages
    required_packages = [
        'torch', 'transformers', 'timm', 'wandb', 'tqdm', 
        'numpy', 'Pillow', 'opencv-python', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"✗ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✓ All required packages installed")
    
    return True

def setup_environment():
    """Setup training environment"""
    print("Setting up training environment...")
    
    # Create necessary directories
    directories = [
        "data", "checkpoints", "logs", "monitoring_logs", 
        "configs", "results", "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create default config if it doesn't exist
    config_file = "configs/training_config.json"
    if not os.path.exists(config_file):
        config = get_training_config()
        config.save_config(config_file)
        print(f"✓ Created default config: {config_file}")

def download_dataset(dataset_name: str, data_dir: str = "data"):
    """Download specified dataset"""
    print(f"Downloading dataset: {dataset_name}")
    
    downloader = DatasetDownloader(data_dir)
    
    if dataset_name == "bridgedata_v2":
        success = downloader.download_bridgedata_v2()
    elif dataset_name == "rlds":
        success = downloader.download_rlds()
    elif dataset_name == "open_x_embodiment":
        success = downloader.download_open_x_embodiment()
    elif dataset_name == "robosuite":
        success = downloader.download_robosuite()
    else:
        print(f"Unknown dataset: {dataset_name}")
        return False
    
    if success:
        print(f"✓ Successfully downloaded {dataset_name}")
        # Create training config for this dataset
        config_file = downloader.create_training_config(dataset_name)
        return str(config_file)
    else:
        print(f"✗ Failed to download {dataset_name}")
        return False

def start_training(config_file: str, dataset_path: str, 
                  dataset_type: str, device: str = "cuda"):
    """Start OpenVLA training with real-time monitoring"""
    print("Starting OpenVLA training...")
    
    # Load configuration
    from config import OpenVLAConfig
    config = OpenVLAConfig.load_config(config_file)
    
    # Update configuration
    config.training.data_path = dataset_path
    config.training.dataset_type = dataset_type
    
    # Create trainer
    trainer = OpenVLARealTimeTrainer(
        config=config,
        data_path=dataset_path,
        dataset_type=dataset_type,
        device=device
    )
    
    # Start training
    trainer.train(num_epochs=config.training.num_epochs)
    
    return trainer

def run_inference_test(model_path: str, test_images: list = None):
    """Run inference test on trained model"""
    print("Running inference test...")
    
    # Load model
    config = OpenVLAConfig()
    model = OpenVLA(config)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test images
    if test_images is None:
        # Create dummy test images
        test_images = torch.randn(2, 3, 224, 224)
    
    # Test prompts
    test_prompts = [
        "In: Pick up the red cup from the table.\nOut:",
        "In: Place the object in the bowl.\nOut:"
    ]
    
    # Run inference
    with torch.no_grad():
        actions = model.predict_action(test_images, test_prompts)
    
    print("Inference test results:")
    print(f"Predicted actions shape: {actions.shape}")
    print(f"Actions: {actions}")
    
    return actions

def create_training_report(results_dir: str = "results"):
    """Create comprehensive training report"""
    print("Creating training report...")
    
    report = {
        'training_completed': True,
        'timestamp': str(Path().cwd()),
        'model_performance': {
            'final_loss': None,
            'final_accuracy': None,
            'training_time': None
        },
        'system_performance': {
            'gpu_utilization': None,
            'memory_usage': None,
            'training_efficiency': None
        },
        'recommendations': [
            "Monitor GPU memory usage during training",
            "Use gradient checkpointing for large models",
            "Consider mixed precision training for speed",
            "Regular checkpointing is recommended"
        ]
    }
    
    # Save report
    report_file = os.path.join(results_dir, "training_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Training report saved: {report_file}")
    return report_file

def main():
    parser = argparse.ArgumentParser(description="Complete OpenVLA Training Pipeline")
    parser.add_argument("--dataset", type=str, default="bridgedata_v2",
                       choices=["bridgedata_v2", "rlds", "open_x_embodiment", "robosuite"],
                       help="Dataset to use for training")
    parser.add_argument("--data_path", type=str, default="data",
                       help="Path to dataset directory")
    parser.add_argument("--config", type=str, default="configs/training_config.json",
                       help="Path to training configuration")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on")
    parser.add_argument("--epochs", type=int, default=27,
                       help="Number of training epochs")
    parser.add_argument("--download_only", action="store_true",
                       help="Only download dataset, don't train")
    parser.add_argument("--test_only", action="store_true",
                       help="Only run inference test")
    parser.add_argument("--model_path", type=str,
                       help="Path to trained model for testing")
    parser.add_argument("--skip_checks", action="store_true",
                       help="Skip system requirement checks")
    
    args = parser.parse_args()
    
    print("🚀 OpenVLA Complete Training Pipeline")
    print("=" * 50)
    
    # Check system requirements
    if not args.skip_checks:
        if not check_system_requirements():
            print("❌ System requirements not met. Exiting.")
            return
    
    # Setup environment
    setup_environment()
    
    # Download dataset if needed
    dataset_path = os.path.join(args.data_path, args.dataset)
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        config_file = download_dataset(args.dataset, args.data_path)
        if not config_file:
            print("❌ Failed to download dataset. Exiting.")
            return
    else:
        print(f"✓ Dataset found at {dataset_path}")
        config_file = args.config
    
    if args.download_only:
        print("✅ Dataset download completed. Use --train to start training.")
        return
    
    # Run inference test if requested
    if args.test_only:
        if args.model_path and os.path.exists(args.model_path):
            run_inference_test(args.model_path)
        else:
            print("❌ Model path not provided or model not found.")
        return
    
    # Start training
    try:
        trainer = start_training(
            config_file=config_file,
            dataset_path=dataset_path,
            dataset_type=args.dataset,
            device=args.device
        )
        
        # Create training report
        create_training_report()
        
        print("✅ Training completed successfully!")
        
        # Run inference test on trained model
        model_path = "checkpoints/openvla_best.pt"
        if os.path.exists(model_path):
            print("Running inference test on trained model...")
            run_inference_test(model_path)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main() 