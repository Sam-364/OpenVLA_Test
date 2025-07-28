"""
Fine-tuning script for OpenVLA model with LoRA support
Based on the research paper: https://arxiv.org/abs/2406.09246

This script implements parameter-efficient fine-tuning using LoRA,
which allows fine-tuning on consumer GPUs with minimal memory requirements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
from peft import LoraConfig, get_peft_model, TaskType

from openvla_architecture import OpenVLA, OpenVLAConfig, ActionTokenizer

class LoRAOpenVLA(nn.Module):
    """
    OpenVLA model with LoRA fine-tuning support
    """
    def __init__(self, base_model: OpenVLA, lora_config: LoraConfig):
        super().__init__()
        self.base_model = base_model
        
        # Apply LoRA to the model
        self.model = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def predict_action(self, *args, **kwargs):
        return self.model.predict_action(*args, **kwargs)

class RobotFineTuneDataset(Dataset):
    """
    Dataset for fine-tuning on specific robot tasks
    """
    def __init__(self, data_path: str, tokenizer, action_tokenizer: ActionTokenizer, 
                 max_length: int = 512, image_size: int = 224, task_name: str = ""):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.task_name = task_name
        
        # Load dataset
        self.samples = self._load_dataset()
        
    def _load_dataset(self) -> List[Dict]:
        """Load fine-tuning data"""
        # This is a placeholder - you would implement actual data loading
        # based on your specific fine-tuning dataset
        samples = []
        
        # Example structure for fine-tuning data
        # You would load actual data from your dataset
        for i in range(100):  # Placeholder - typically 10-150 demonstrations
            sample = {
                'image': f"finetune_image_{i}.jpg",
                'instruction': f"Task: {self.task_name} - Pick up object {i}",
                'actions': np.random.randn(7)  # 7-DoF actions
            }
            samples.append(sample)
            
        return samples
    
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
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        # Placeholder - implement actual image loading
        # In practice, you would load from file and apply transforms
        image = torch.randn(3, self.image_size, self.image_size)
        return image

class OpenVLAFineTuner:
    """
    Fine-tuner for OpenVLA model with LoRA support
    """
    def __init__(self, model: LoRAOpenVLA, train_dataset: Dataset, 
                 val_dataset: Optional[Dataset] = None,
                 device: str = 'cuda', use_wandb: bool = True,
                 learning_rate: float = 5e-4, batch_size: int = 16):
        self.model = model
        self.device = device
        self.use_wandb = use_wandb
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # Setup optimizer (only for LoRA parameters)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Setup wandb
        if use_wandb:
            wandb.init(
                project="openvla-finetune",
                config={
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'task': train_dataset.task_name if hasattr(train_dataset, 'task_name') else 'unknown'
                },
                name=f"openvla-lora-{learning_rate}-{batch_size}"
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_action_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(self.device)
            text_tokens = batch['text_tokens'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            action_tokens = batch['action_tokens'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                images=images,
                text_tokens=text_tokens,
                action_tokens=action_tokens,
                attention_mask=attention_mask
            )
            
            # Compute loss
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute action accuracy
            logits = outputs.logits
            action_logits = logits[:, -action_tokens.shape[1]:, :]
            action_preds = torch.argmax(action_logits, dim=-1)
            action_accuracy = (action_preds == action_tokens).float().mean()
            
            # Update metrics
            total_loss += loss.item()
            total_action_accuracy += action_accuracy.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'action_acc': f"{action_accuracy.item():.4f}"
            })
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/action_accuracy': action_accuracy.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.current_step
                })
            
            self.current_step += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_action_accuracy = total_action_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'action_accuracy': avg_action_accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        total_action_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                images = batch['image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                action_tokens = batch['action_tokens'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    images=images,
                    text_tokens=text_tokens,
                    action_tokens=action_tokens,
                    attention_mask=attention_mask
                )
                
                # Compute loss
                loss = outputs.loss
                
                # Compute action accuracy
                logits = outputs.logits
                action_logits = logits[:, -action_tokens.shape[1]:, :]
                action_preds = torch.argmax(action_logits, dim=-1)
                action_accuracy = (action_preds == action_tokens).float().mean()
                
                # Update metrics
                total_loss += loss.item()
                total_action_accuracy += action_accuracy.item()
                num_batches += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_action_accuracy = total_action_accuracy / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_action_accuracy': avg_action_accuracy
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save LoRA adapter checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'lora_config': self.model.model.peft_config
        }
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/openvla_lora_epoch_{epoch}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics.get('val_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = "checkpoints/openvla_lora_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs: int = 10):
        """Main fine-tuning loop"""
        print(f"Starting fine-tuning for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Print metrics
            print(f"Epoch {epoch}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **metrics
                })

def create_lora_config(rank: int = 32, alpha: int = 64, dropout: float = 0.1) -> LoraConfig:
    """Create LoRA configuration"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

def main():
    """Main fine-tuning function"""
    # Create configuration
    config = OpenVLAConfig()
    
    # Load pre-trained model
    print("Loading pre-trained OpenVLA model...")
    base_model = OpenVLA(config)
    
    # Load pre-trained weights (if available)
    checkpoint_path = "checkpoints/openvla_best.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained weights from {checkpoint_path}")
    
    # Fit action tokenizer on training data
    # In practice, you would load actual training actions
    dummy_actions = torch.randn(1000, 7)  # 1k samples, 7-DoF
    base_model.action_tokenizer.fit(dummy_actions)
    
    # Create LoRA configuration
    lora_config = create_lora_config(rank=32, alpha=64, dropout=0.1)
    
    # Create LoRA model
    model = LoRAOpenVLA(base_model, lora_config)
    
    # Create fine-tuning datasets
    train_dataset = RobotFineTuneDataset(
        data_path="path/to/finetune/train/data",
        tokenizer=base_model.tokenizer,
        action_tokenizer=base_model.action_tokenizer,
        max_length=config.max_length,
        image_size=config.image_size,
        task_name="pick_and_place"
    )
    
    val_dataset = RobotFineTuneDataset(
        data_path="path/to/finetune/val/data",
        tokenizer=base_model.tokenizer,
        action_tokenizer=base_model.action_tokenizer,
        max_length=config.max_length,
        image_size=config.image_size,
        task_name="pick_and_place"
    )
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create fine-tuner
    fine_tuner = OpenVLAFineTuner(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        use_wandb=True,
        learning_rate=5e-4,
        batch_size=16
    )
    
    # Start fine-tuning
    fine_tuner.train(num_epochs=10)

if __name__ == "__main__":
    main() 