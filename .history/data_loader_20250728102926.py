"""
Data loading utilities for OpenVLA training
Supports multiple robot dataset formats including RLDS, BridgeData V2, and custom datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import h5py
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
from pathlib import Path
import random
from transformers import AutoTokenizer
import tensorflow as tf
import tensorflow_datasets as tfds

class RobotDataset(Dataset):
    """
    Base class for robot demonstration datasets
    """
    def __init__(self, data_path: str, tokenizer, action_tokenizer, 
                 max_length: int = 512, image_size: int = 224,
                 augment: bool = True):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.augment = augment
        
        # Image transforms
        self.transform = self._get_transforms()
        
        # Load dataset
        self.samples = self._load_dataset()
        
    def _get_transforms(self):
        """Get image transformation pipeline"""
        import torchvision.transforms as T
        
        transforms = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ]
        
        if self.augment:
            transforms = [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomRotation(degrees=5),
            ] + transforms
            
        return T.Compose(transforms)
    
    def _load_dataset(self) -> List[Dict]:
        """Load robot demonstration data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if isinstance(image_path, str):
            if image_path.endswith('.npy'):
                # Load numpy array
                image = np.load(image_path)
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
            else:
                # Load image file
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
        else:
            # Assume it's already a numpy array
            image = image_path
            
        # Convert to PIL Image for transforms
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Apply transforms
        image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = self._load_image(sample['image'])
        
        # Tokenize instruction
        instruction = sample['instruction']
        text_tokens = self.tokenizer(
            instruction,
            max_length=self.max_length // 2,  # Reserve half for actions
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize actions
        actions = torch.tensor(sample['actions'], dtype=torch.float32)
        action_tokens = self.action_tokenizer.encode(actions.unsqueeze(0)).squeeze(0)
        
        return {
            'image': image,
            'text_tokens': text_tokens['input_ids'].squeeze(0),
            'attention_mask': text_tokens['attention_mask'].squeeze(0),
            'action_tokens': action_tokens,
            'actions': actions
        }

class BridgeDataV2Dataset(RobotDataset):
    """
    BridgeData V2 dataset loader
    """
    def _load_dataset(self) -> List[Dict]:
        """Load BridgeData V2 format data"""
        samples = []
        
        # BridgeData V2 structure
        # data_path should point to a directory with episode subdirectories
        data_dir = Path(self.data_path)
        
        for episode_dir in data_dir.glob("episode_*"):
            if not episode_dir.is_dir():
                continue
                
            # Load episode metadata
            metadata_file = episode_dir / "metadata.json"
            if not metadata_file.exists():
                continue
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load images and actions
            images_dir = episode_dir / "images"
            actions_file = episode_dir / "actions.npy"
            
            if not images_dir.exists() or not actions_file.exists():
                continue
                
            # Load actions
            actions = np.load(actions_file)  # Shape: (num_steps, action_dim)
            
            # Load images
            image_files = sorted(images_dir.glob("*.jpg"))
            
            # Create samples for each timestep
            for i, (image_file, action) in enumerate(zip(image_files, actions)):
                # Create instruction based on metadata
                instruction = self._create_instruction(metadata, i)
                
                sample = {
                    'image': str(image_file),
                    'instruction': instruction,
                    'actions': action
                }
                samples.append(sample)
                
        return samples
    
    def _create_instruction(self, metadata: Dict, timestep: int) -> str:
        """Create instruction text from metadata"""
        task_name = metadata.get('task_name', 'robot task')
        return f"In: What action should the robot take to {task_name}?\nOut:"

class RLDSDataset(RobotDataset):
    """
    RLDS (Robot Learning Data Store) dataset loader
    """
    def _load_dataset(self) -> List[Dict]:
        """Load RLDS format data"""
        samples = []
        
        # Load RLDS dataset
        dataset = tfds.load(
            self.data_path,
            split='train',
            as_supervised=False
        )
        
        for episode in dataset:
            # Extract episode data
            images = episode['observation']['image']
            actions = episode['action']
            language_instruction = episode.get('language_instruction', '')
            
            # Create samples for each timestep
            for i, (image, action) in enumerate(zip(images, actions)):
                # Convert image to numpy
                image_np = image.numpy()
                
                # Create instruction
                if language_instruction:
                    instruction = f"In: {language_instruction}\nOut:"
                else:
                    instruction = f"In: What action should the robot take at step {i}?\nOut:"
                
                sample = {
                    'image': image_np,
                    'instruction': instruction,
                    'actions': action.numpy()
                }
                samples.append(sample)
                
        return samples

class CustomRobotDataset(RobotDataset):
    """
    Custom robot dataset loader
    """
    def _load_dataset(self) -> List[Dict]:
        """Load custom format data"""
        samples = []
        
        # Load from JSON file
        data_file = Path(self.data_path)
        if data_file.suffix == '.json':
            with open(data_file, 'r') as f:
                data = json.load(f)
                
            for item in data:
                sample = {
                    'image': item['image_path'],
                    'instruction': item['instruction'],
                    'actions': np.array(item['actions'])
                }
                samples.append(sample)
        
        # Load from HDF5 file
        elif data_file.suffix == '.h5':
            with h5py.File(data_file, 'r') as f:
                num_episodes = len(f['episodes'])
                
                for episode_idx in range(num_episodes):
                    episode = f['episodes'][episode_idx]
                    
                    images = episode['images'][:]  # Shape: (num_steps, H, W, C)
                    actions = episode['actions'][:]  # Shape: (num_steps, action_dim)
                    instructions = episode['instructions'][:]  # Shape: (num_steps,)
                    
                    for i in range(len(images)):
                        sample = {
                            'image': images[i],
                            'instruction': instructions[i].decode() if isinstance(instructions[i], bytes) else instructions[i],
                            'actions': actions[i]
                        }
                        samples.append(sample)
        
        return samples

def create_dataloader(
    data_path: str,
    tokenizer,
    action_tokenizer,
    dataset_type: str = 'bridge',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for robot demonstration data
    
    Args:
        data_path: Path to dataset
        tokenizer: Text tokenizer
        action_tokenizer: Action tokenizer
        dataset_type: Type of dataset ('bridge', 'rlds', 'custom')
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader instance
    """
    if dataset_type == 'bridge':
        dataset = BridgeDataV2Dataset(data_path, tokenizer, action_tokenizer, **kwargs)
    elif dataset_type == 'rlds':
        dataset = RLDSDataset(data_path, tokenizer, action_tokenizer, **kwargs)
    elif dataset_type == 'custom':
        dataset = CustomRobotDataset(data_path, tokenizer, action_tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

# Example usage
if __name__ == "__main__":
    from openvla_architecture import ActionTokenizer
    from transformers import LlamaTokenizer
    
    # Initialize tokenizers
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    action_tokenizer = ActionTokenizer()
    
    # Create dataloader
    dataloader = create_dataloader(
        data_path="/path/to/your/dataset",
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        dataset_type='bridge',
        batch_size=4
    )
    
    # Test dataloader
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Text tokens shape: {batch['text_tokens'].shape}")
        print(f"Action tokens shape: {batch['action_tokens'].shape}")
        break 