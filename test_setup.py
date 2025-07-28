#!/usr/bin/env python3
"""
Simple test script to verify OpenVLA setup
"""

import os
import sys
import json
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    try:
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
    except ImportError:
        print("✗ Pillow not installed")
        return False
    
    return True

def test_openvla_architecture():
    """Test OpenVLA architecture import"""
    print("\nTesting OpenVLA architecture...")
    
    try:
        from openvla_architecture import OpenVLA, OpenVLAConfig
        print("✓ OpenVLA architecture imported successfully")
        
        # Test config creation
        config = OpenVLAConfig()
        print("✓ OpenVLAConfig created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error importing OpenVLA architecture: {e}")
        return False

def test_config_system():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        from config import get_training_config
        config = get_training_config()
        print("✓ Configuration system working")
        
        # Test config saving/loading
        test_config_file = "test_config.json"
        config.save_config(test_config_file)
        
        from config import OpenVLAConfig
        loaded_config = OpenVLAConfig.load_config(test_config_file)
        print("✓ Config save/load working")
        
        # Clean up
        os.remove(test_config_file)
        
        return True
    except Exception as e:
        print(f"✗ Error in configuration system: {e}")
        return False

def test_file_structure():
    """Test file structure"""
    print("\nTesting file structure...")
    
    required_files = [
        "openvla_architecture.py",
        "train_openvla.py",
        "finetune_openvla.py",
        "inference_openvla.py",
        "data_loader.py",
        "robot_integration.py",
        "evaluate_openvla.py",
        "config.py",
        "requirements.txt",
        "README.md",
        "train_with_realtime_monitoring.py",
        "download_datasets.py",
        "realtime_monitor.py",
        "train_openvla_complete.py",
        "QUICK_START.md"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    return True

def test_directory_creation():
    """Test directory creation"""
    print("\nTesting directory creation...")
    
    directories = [
        "data", "checkpoints", "logs", "monitoring_logs", 
        "configs", "results", "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(directory):
            print(f"✓ Created/verified {directory}/")
        else:
            print(f"✗ Failed to create {directory}/")
            return False
    
    return True

def test_system_info():
    """Test system information"""
    print("\nSystem Information:")
    
    import platform
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1024**3
    print(f"Free disk space: {free_gb:.1f} GB")

def main():
    """Run all tests"""
    print("🧪 OpenVLA Setup Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Structure", test_file_structure),
        ("Directory Creation", test_directory_creation),
        ("OpenVLA Architecture", test_openvla_architecture),
        ("Configuration System", test_config_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Install additional dependencies: pip install -r requirements.txt")
        print("2. Download dataset: python download_datasets.py --dataset bridgedata_v2")
        print("3. Start training: python train_openvla_complete.py --dataset bridgedata_v2")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    # Show system info
    test_system_info()

if __name__ == "__main__":
    main() 