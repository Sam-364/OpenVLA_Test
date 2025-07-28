#!/usr/bin/env python3
"""
Basic OpenVLA Training Script
Demonstrates the training process with minimal dependencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from datetime import datetime

class BasicRobotDataset(Dataset):
    """Basic synthetic dataset for OpenVLA training"""
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        
        # Generate synthetic data
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.actions = torch.randn(num_samples, 7)  # 7-DoF actions
        self.texts = [f"Pick up object {i}" for i in range(num_samples)]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'action': self.actions[idx],
            'text': self.texts[idx]
        }

class BasicOpenVLATrainer:
    """Basic OpenVLA trainer with synthetic data"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Create a simple model for demonstration
        self.model = nn.Sequential(
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # 7-DoF actions
        ).to(device)
        
        # Create dataset
        self.dataset = BasicRobotDataset()
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=4,
            shuffle=True
        )
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Create output directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        print("Training epoch...")
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Get data
            images = batch['image'].to(self.device)
            actions = batch['action'].to(self.device)
            
            # Flatten images
            images_flat = images.view(images.shape[0], -1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_actions = self.model(images_flat)
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        torch.save(checkpoint, f'checkpoints/openvla_epoch_{epoch}.pt')
        print(f"Checkpoint saved: checkpoints/openvla_epoch_{epoch}.pt")
    
    def train(self, num_epochs=3):
        """Main training loop"""
        print("🚀 Starting Basic OpenVLA Training")
        print("=" * 40)
        print(f"Device: {self.device}")
        print(f"Dataset size: {len(self.dataset)} samples")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 40)
        
        training_log = []
        
        try:
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                
                # Train one epoch
                loss = self.train_epoch()
                
                print(f"Epoch {epoch + 1} completed. Average Loss: {loss:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, loss)
                
                # Log metrics
                training_log.append({
                    'epoch': epoch + 1,
                    'loss': loss,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Early stopping
                if loss < 0.01:
                    print(f"Loss < 0.01 reached! Stopping training.")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            # Save training log
            with open('results/training_log.json', 'w') as f:
                json.dump(training_log, f, indent=2)
            
            print("\nTraining completed!")
            print("Checkpoints saved in: checkpoints/")
            print("Training log saved in: results/training_log.json")
            
            # Show final results
            if training_log:
                final_loss = training_log[-1]['loss']
                print(f"Final loss: {final_loss:.4f}")

def main():
    print("Starting OpenVLA training process...")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = BasicOpenVLATrainer(device=device)
    
    # Start training
    trainer.train(num_epochs=3)

if __name__ == "__main__":
    main() 