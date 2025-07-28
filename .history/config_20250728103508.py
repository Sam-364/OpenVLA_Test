"""
Configuration module for OpenVLA
Centralizes all settings for training, inference, and evaluation
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Model dimensions
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    
    # Vision encoder settings
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    vision_hidden_size: int = 768
    
    # Action discretization
    action_bins: int = 256
    action_dim: int = 7  # 7-DoF robot actions
    
    # LLM settings
    llm_model_name: str = "meta-llama/Llama-2-7b-hf"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'rms_norm_eps': self.rms_norm_eps,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_channels': self.num_channels,
            'vision_hidden_size': self.vision_hidden_size,
            'action_bins': self.action_bins,
            'action_dim': self.action_dim,
            'llm_model_name': self.llm_model_name
        }

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 2048
    max_length: int = 512
    num_epochs: int = 27
    
    # Data settings
    dataset_type: str = 'bridge'  # 'bridge', 'rlds', 'custom'
    data_path: str = "path/to/your/dataset"
    val_data_path: Optional[str] = None
    
    # Optimization
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_clip_val: float = 1.0
    
    # Logging and checkpointing
    save_every: int = 5
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    
    # Distributed training
    use_distributed: bool = False
    num_workers: int = 4
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'num_epochs': self.num_epochs,
            'dataset_type': self.dataset_type,
            'data_path': self.data_path,
            'val_data_path': self.val_data_path,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'gradient_clip_val': self.gradient_clip_val,
            'save_every': self.save_every,
            'log_every': self.log_every,
            'eval_every': self.eval_every,
            'checkpoint_dir': self.checkpoint_dir,
            'use_distributed': self.use_distributed,
            'num_workers': self.num_workers,
            'patience': self.patience,
            'min_delta': self.min_delta
        }

@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Model settings
    model_path: str = "checkpoints/openvla_best.pt"
    device: str = "cuda"
    
    # Quantization and precision
    use_quantization: bool = False
    precision: str = "bfloat16"  # "bfloat16", "float16", "float32"
    
    # Generation settings
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 7
    
    # Robot settings
    robot_type: str = "pybullet"  # "pybullet", "ros", "real"
    robot_config: Dict[str, Any] = None
    
    # Interactive settings
    interactive: bool = False
    control_frequency: float = 10.0  # Hz
    
    def __post_init__(self):
        if self.robot_config is None:
            self.robot_config = {
                'gui': True,
                'robot_urdf': 'panda.urdf'
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'device': self.device,
            'use_quantization': self.use_quantization,
            'precision': self.precision,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'max_new_tokens': self.max_new_tokens,
            'robot_type': self.robot_type,
            'robot_config': self.robot_config,
            'interactive': self.interactive,
            'control_frequency': self.control_frequency
        }

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Model and data
    model_path: str = "checkpoints/openvla_best.pt"
    data_path: str = "path/to/test/dataset"
    dataset_type: str = "bridge"
    
    # Evaluation settings
    device: str = "cuda"
    num_samples: Optional[int] = None
    batch_size: int = 32
    
    # Metrics
    evaluate_accuracy: bool = True
    evaluate_speed: bool = True
    evaluate_robustness: bool = True
    
    # Visualization
    generate_plots: bool = True
    output_dir: str = "evaluation_results"
    
    # Robustness testing
    noise_levels: List[float] = None
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.1, 0.2, 0.3]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_path': self.model_path,
            'data_path': self.data_path,
            'dataset_type': self.dataset_type,
            'device': self.device,
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'evaluate_accuracy': self.evaluate_accuracy,
            'evaluate_speed': self.evaluate_speed,
            'evaluate_robustness': self.evaluate_robustness,
            'generate_plots': self.generate_plots,
            'output_dir': self.output_dir,
            'noise_levels': self.noise_levels
        }

@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration"""
    # LoRA settings
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    
    # Data settings
    data_path: str = "path/to/finetune/dataset"
    dataset_type: str = "custom"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'data_path': self.data_path,
            'dataset_type': self.dataset_type
        }

class OpenVLAConfig:
    """
    Main configuration class for OpenVLA
    """
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.evaluation = EvaluationConfig()
        self.lora = LoRAConfig()
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'inference': self.inference.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'lora': self.lora.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'OpenVLAConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Update configurations
        for key, value in config_dict.get('model', {}).items():
            setattr(config.model, key, value)
        
        for key, value in config_dict.get('training', {}).items():
            setattr(config.training, key, value)
        
        for key, value in config_dict.get('inference', {}).items():
            setattr(config.inference, key, value)
        
        for key, value in config_dict.get('evaluation', {}).items():
            setattr(config.evaluation, key, value)
        
        for key, value in config_dict.get('lora', {}).items():
            setattr(config.lora, key, value)
        
        return config

# Default configurations
def get_default_config() -> OpenVLAConfig:
    """Get default configuration"""
    return OpenVLAConfig()

def get_training_config() -> OpenVLAConfig:
    """Get configuration optimized for training"""
    config = OpenVLAConfig()
    
    # Optimize for training
    config.training.batch_size = 1024  # Smaller batch size for memory
    config.training.learning_rate = 2e-5
    config.training.num_epochs = 27
    config.training.use_distributed = True
    config.training.num_workers = 8
    
    return config

def get_inference_config() -> OpenVLAConfig:
    """Get configuration optimized for inference"""
    config = OpenVLAConfig()
    
    # Optimize for inference
    config.inference.use_quantization = True
    config.inference.precision = "bfloat16"
    config.inference.do_sample = False
    config.inference.temperature = 1.0
    
    return config

def get_evaluation_config() -> OpenVLAConfig:
    """Get configuration optimized for evaluation"""
    config = OpenVLAConfig()
    
    # Optimize for evaluation
    config.evaluation.batch_size = 64
    config.evaluation.generate_plots = True
    config.evaluation.evaluate_robustness = True
    
    return config

# Example usage
if __name__ == "__main__":
    # Create default configuration
    config = get_default_config()
    
    # Save configuration
    config.save_config("configs/default_config.json")
    
    # Create training configuration
    train_config = get_training_config()
    train_config.save_config("configs/training_config.json")
    
    # Create inference configuration
    inference_config = get_inference_config()
    inference_config.save_config("configs/inference_config.json")
    
    # Create evaluation configuration
    eval_config = get_evaluation_config()
    eval_config.save_config("configs/evaluation_config.json")
    
    print("Configuration files created in configs/ directory") 