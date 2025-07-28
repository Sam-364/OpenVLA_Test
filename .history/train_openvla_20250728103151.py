"""
Training script for OpenVLA model
Based on the research paper: https://arxiv.org/abs/2406.09246

This script implements the training procedure described in the paper:
- 27 epochs through training dataset
- Learning rate: 2e-5 (fixed, no warmup)
- Batch size: 2048
- Action token accuracy target: >95%
- Vision encoder fine-tuning enabled
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image
import random

from openvla_architecture import OpenVLA, OpenVLAConfig, ActionTokenizer
from data_loader import create_dataloader

class OpenVLATrainer:
    """
    Trainer for OpenVLA model
    """
    def __init__(self, config: OpenVLAConfig, model: OpenVLA, 
                 data_path: str, dataset_type: str = 'bridge',
                 device: str = 'cuda', use_wandb: bool = True):
        self.config = config
        self.model = model
        self.device = device
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup data loaders using the new data loader module
        self.train_loader = create_dataloader(
            data_path=data_path,
            tokenizer=self.model.tokenizer,
            action_tokenizer=self.model.action_tokenizer,
            dataset_type=dataset_type,
            batch_size=config.batch_size,
            num_workers=4,
            shuffle=True
        )
        
        # For validation, use a subset of the same data
        # In practice, you'd have a separate validation dataset
        self.val_loader = None
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Setup wandb
        if use_wandb:
            wandb.init(
                project="openvla-training",
                config=vars(config),
                name=f"openvla-{config.batch_size}-{config.learning_rate}"
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
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/openvla_epoch_{epoch}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics.get('val_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = "checkpoints/openvla_best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs: int = 27):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
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
            
            # Check if we should stop training (action accuracy > 95%)
            if train_metrics['action_accuracy'] > 0.95:
                print(f"Action accuracy > 95% reached! Stopping training.")
                break

def setup_distributed_training():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        return True
    return False

def main():
    """Main training function"""
    # Setup distributed training if needed
    is_distributed = setup_distributed_training()
    
    # Create configuration
    config = OpenVLAConfig()
    
    # Create model
    model = OpenVLA(config)
    
    # Fit action tokenizer on training data
    # In practice, you would load actual training actions
    dummy_actions = torch.randn(10000, 7)  # 10k samples, 7-DoF
    model.action_tokenizer.fit(dummy_actions)
    
    # Create datasets
    train_dataset = RobotDataset(
        data_path="path/to/train/data",
        tokenizer=model.tokenizer,
        action_tokenizer=model.action_tokenizer,
        max_length=config.max_length,
        image_size=config.image_size
    )
    
    val_dataset = RobotDataset(
        data_path="path/to/val/data",
        tokenizer=model.tokenizer,
        action_tokenizer=model.action_tokenizer,
        max_length=config.max_length,
        image_size=config.image_size
    )
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create trainer
    trainer = OpenVLATrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        use_wandb=True
    )
    
    # Start training
    trainer.train(num_epochs=27)

if __name__ == "__main__":
    main() 