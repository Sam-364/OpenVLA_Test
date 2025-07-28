#!/usr/bin/env python3
"""
Simplified OpenVLA Training Script
Works with basic dependencies and provides a working training example
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openvla_architecture import OpenVLA, OpenVLAConfig

class SimpleRobotDataset(Dataset):
    """Simple dataset for testing OpenVLA training"""
    
    def __init__(self, num_samples=1000, image_size=224, action_dim=7):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        
        # Generate synthetic data
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.actions = torch.randn(num_samples, action_dim)
        self.texts = [
            f"Pick up object {i} from the table" for i in range(num_samples)
        ]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'action': self.actions[idx],
            'text': self.texts[idx]
        }

class SimpleOpenVLATrainer:
    """Simplified OpenVLA trainer"""
    
    def __init__(self, config: OpenVLAConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Create model
        self.model = OpenVLA(config)
        self.model = self.model.to(device)
        
        # Create dataset
        self.dataset = SimpleRobotDataset()
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Create output directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        print(f"Training epoch {self.current_epoch + 1}...")
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move data to device
            images = batch['image'].to(self.device)
            actions = batch['action'].to(self.device)
            texts = batch['text']
            
            # Tokenize text (simplified)
            text_tokens = torch.randint(0, 1000, (images.shape[0], 50)).to(self.device)
            attention_mask = torch.ones_like(text_tokens)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # For simplicity, we'll use a basic forward pass
            # In real implementation, this would use the full OpenVLA forward
            outputs = self.model.llm(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                labels=text_tokens
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, 'checkpoints/openvla_latest.pt')
        
        # Save best
        if metrics['loss'] < self.best_loss:
            self.best_loss = metrics['loss']
            torch.save(checkpoint, 'checkpoints/openvla_best.pt')
            print(f"New best model saved with loss: {self.best_loss:.4f}")
    
    def train(self, num_epochs: int = 5):
        """Main training loop"""
        print(f"Starting OpenVLA training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Dataset size: {len(self.dataset)} samples")
        print(f"Batch size: {self.config.batch_size}")
        
        training_log = []
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                metrics = self.train_epoch()
                
                # Print metrics
                print(f"Epoch {epoch + 1}: Loss = {metrics['loss']:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, metrics)
                
                # Log metrics
                training_log.append({
                    'epoch': epoch + 1,
                    'loss': metrics['loss'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Early stopping (simplified)
                if metrics['loss'] < 0.1:
                    print(f"Loss < 0.1 reached! Stopping training.")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            # Save training log
            with open('results/training_log.json', 'w') as f:
                json.dump(training_log, f, indent=2)
            
            print("Training completed!")
            print("Checkpoints saved in: checkpoints/")
            print("Training log saved in: results/training_log.json")

def main():
    parser = argparse.ArgumentParser(description="Simplified OpenVLA Training")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on")
    
    args = parser.parse_args()
    
    print("🚀 Simplified OpenVLA Training")
    print("=" * 40)
    
    # Create configuration
    config = OpenVLAConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # Create trainer
    trainer = SimpleOpenVLATrainer(config, device=args.device)
    
    # Start training
    trainer.train(num_epochs=args.epochs)

if __name__ == "__main__":
    main() 