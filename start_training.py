#!/usr/bin/env python3
"""
OpenVLA Training Starter Script
Simple entry point for training OpenVLA with real-time monitoring
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("🚀 OpenVLA Training Pipeline")
    print("=" * 50)
    print("Vision-Language-Action Model for Robot Manipulation")
    print("Based on: https://arxiv.org/abs/2406.09246")
    print("=" * 50)

def check_prerequisites():
    """Check if system is ready for training"""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - training will be slow")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1024**3
    print(f"✓ Free disk space: {free_gb:.1f}GB")
    
    if free_gb < 50:
        print("⚠️  Low disk space - may need more for datasets")
    
    return True

def show_dataset_options():
    """Show available datasets"""
    print("\n📊 Available Datasets:")
    print("-" * 30)
    
    datasets = {
        "bridgedata_v2": {
            "name": "BridgeData V2",
            "size": "45GB",
            "description": "60K trajectories, 13 skills, 24 environments",
            "recommended": True
        },
        "rlds": {
            "name": "RLDS",
            "size": "20GB", 
            "description": "Standardized robot learning datasets",
            "recommended": False
        },
        "open_x_embodiment": {
            "name": "Open X-Embodiment",
            "size": "100GB",
            "description": "1M+ trajectories, multi-robot",
            "recommended": False
        },
        "robosuite": {
            "name": "RoboSuite",
            "size": "5GB",
            "description": "Simulation environment with demos",
            "recommended": False
        }
    }
    
    for key, dataset in datasets.items():
        rec = "⭐" if dataset["recommended"] else "  "
        print(f"{rec} {key}: {dataset['name']} ({dataset['size']})")
        print(f"    {dataset['description']}")

def run_training(dataset, download_only=False, skip_checks=False):
    """Run the training pipeline"""
    print(f"\n🎯 Starting training with {dataset} dataset...")
    
    cmd = [
        "python3", "train_openvla_complete.py",
        "--dataset", dataset
    ]
    
    if download_only:
        cmd.append("--download_only")
    
    if skip_checks:
        cmd.append("--skip_checks")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False

def show_help():
    """Show help information"""
    print("\n📖 Quick Start Guide:")
    print("1. Choose a dataset (recommended: bridgedata_v2)")
    print("2. Download dataset: python3 start_training.py --dataset bridgedata_v2 --download")
    print("3. Start training: python3 start_training.py --dataset bridgedata_v2")
    print("4. Monitor progress: Real-time dashboard will open automatically")
    
    print("\n🔧 Advanced Options:")
    print("--download: Download dataset only")
    print("--skip-checks: Skip system requirement checks")
    print("--help: Show this help message")
    
    print("\n📁 Output Files:")
    print("- checkpoints/: Model checkpoints")
    print("- monitoring_logs/: Performance metrics")
    print("- results/: Training reports")
    print("- logs/: Training logs")

def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA Training Starter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BridgeData V2 dataset
  python3 start_training.py --dataset bridgedata_v2 --download
  
  # Start training with BridgeData V2
  python3 start_training.py --dataset bridgedata_v2
  
  # Skip system checks (if you know your system is ready)
  python3 start_training.py --dataset bridgedata_v2 --skip-checks
        """
    )
    
    parser.add_argument("--dataset", type=str, default="bridgedata_v2",
                       choices=["bridgedata_v2", "rlds", "open_x_embodiment", "robosuite"],
                       help="Dataset to use for training")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset only, don't train")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip system requirement checks")
    parser.add_argument("--help-datasets", action="store_true",
                       help="Show dataset information")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.help_datasets:
        show_dataset_options()
        return
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Please install required packages.")
            print("Run: pip install -r requirements.txt")
            return
    
    # Show dataset info
    show_dataset_options()
    
    # Run training
    success = run_training(
        dataset=args.dataset,
        download_only=args.download,
        skip_checks=args.skip_checks
    )
    
    if success:
        print("\n✅ Training completed successfully!")
        print("Check the following directories for results:")
        print("- checkpoints/: Best model")
        print("- monitoring_logs/: Performance metrics")
        print("- results/: Training report")
    else:
        print("\n❌ Training failed. Check the logs above for errors.")
        show_help()

if __name__ == "__main__":
    main() 