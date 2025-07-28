#!/usr/bin/env python3
"""
Advanced OpenVLA Training Script
Includes real-time monitoring and better training process
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import time
import threading
import queue
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List

class AdvancedRobotDataset(Dataset):
    """Advanced synthetic dataset for OpenVLA training"""
    
    def __init__(self, num_samples=500):
        self.num_samples = num_samples
        
        # Generate more realistic synthetic data
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.actions = torch.randn(num_samples, 7)  # 7-DoF actions
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

class RealTimeMonitor:
    """Real-time training monitor"""
    
    def __init__(self):
        self.metrics_queue = queue.Queue()
        self.is_running = False
        self.monitor_thread = None
        
        # Metrics storage
        self.loss_history = []
        self.epoch_history = []
        self.time_history = []
        
        # Setup plotting
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        self.fig.suptitle('OpenVLA Training - Real-Time Monitoring', fontsize=14)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Real-time monitoring started...")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        plt.ioff()
        print("Real-time monitoring stopped...")
    
    def update_metrics(self, epoch, loss, timestamp):
        """Update training metrics"""
        self.metrics_queue.put({
            'epoch': epoch,
            'loss': loss,
            'timestamp': timestamp
        })
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Get latest metrics
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    self.loss_history.append(metrics['loss'])
                    self.epoch_history.append(metrics['epoch'])
                    self.time_history.append(metrics['timestamp'])
                
                # Update plot
                if self.loss_history:
                    self._update_plot()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def _update_plot(self):
        """Update the real-time plot"""
        self.ax.clear()
        self.ax.plot(self.epoch_history, self.loss_history, 'b-', linewidth=2)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss Over Time')
        self.ax.grid(True, alpha=0.3)
        
        # Add current loss annotation
        if self.loss_history:
            current_loss = self.loss_history[-1]
            self.ax.annotate(f'Loss: {current_loss:.4f}', 
                           xy=(self.epoch_history[-1], current_loss),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.pause(0.01)

class AdvancedOpenVLATrainer:
    """Advanced OpenVLA trainer with real-time monitoring"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Create a more sophisticated model
        self.model = nn.Sequential(
            nn.Linear(3 * 224 * 224, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # 7-DoF actions
        ).to(device)
        
        # Create dataset
        self.dataset = AdvancedRobotDataset()
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        # Setup optimizer and loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        self.criterion = nn.MSELoss()
        
        # Create output directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('monitoring_logs', exist_ok=True)
        
        # Initialize monitor
        self.monitor = RealTimeMonitor()
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.dataloader)}, Loss: {loss.item():.4f}")
        
        # Update learning rate
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, loss):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save latest
        torch.save(checkpoint, 'checkpoints/openvla_latest.pt')
        
        # Save epoch-specific
        torch.save(checkpoint, f'checkpoints/openvla_epoch_{epoch}.pt')
        
        print(f"Checkpoint saved: checkpoints/openvla_epoch_{epoch}.pt")
    
    def train(self, num_epochs=10):
        """Main training loop with real-time monitoring"""
        print("🚀 Starting Advanced OpenVLA Training")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Dataset size: {len(self.dataset)} samples")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.dataloader.batch_size}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 50)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        training_log = []
        best_loss = float('inf')
        
        try:
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                epoch_start_time = time.time()
                
                # Train one epoch
                loss = self.train_epoch()
                
                epoch_time = time.time() - epoch_start_time
                
                print(f"Epoch {epoch + 1} completed.")
                print(f"  Average Loss: {loss:.4f}")
                print(f"  Time: {epoch_time:.2f}s")
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Update monitor
                self.monitor.update_metrics(epoch + 1, loss, datetime.now().isoformat())
                
                # Save checkpoint
                self.save_checkpoint(epoch, loss)
                
                # Log metrics
                training_log.append({
                    'epoch': epoch + 1,
                    'loss': loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'time': epoch_time,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Track best loss
                if loss < best_loss:
                    best_loss = loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'loss': loss
                    }, 'checkpoints/openvla_best.pt')
                    print(f"  New best model saved! Loss: {best_loss:.4f}")
                
                # Early stopping
                if loss < 0.01:
                    print(f"Loss < 0.01 reached! Stopping training.")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Save training log
            with open('results/training_log.json', 'w') as f:
                json.dump(training_log, f, indent=2)
            
            # Save monitoring data
            monitoring_data = {
                'loss_history': self.monitor.loss_history,
                'epoch_history': self.monitor.epoch_history,
                'time_history': self.monitor.time_history
            }
            with open('monitoring_logs/monitoring_data.json', 'w') as f:
                json.dump(monitoring_data, f, indent=2)
            
            print("\nTraining completed!")
            print("Checkpoints saved in: checkpoints/")
            print("Training log saved in: results/training_log.json")
            print("Monitoring data saved in: monitoring_logs/monitoring_data.json")
            
            # Show final results
            if training_log:
                final_loss = training_log[-1]['loss']
                print(f"Final loss: {final_loss:.4f}")
                print(f"Best loss: {best_loss:.4f}")
                print(f"Total training time: {sum(log['time'] for log in training_log):.2f}s")

def main():
    print("Starting Advanced OpenVLA training process...")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = AdvancedOpenVLATrainer(device=device)
    
    # Start training
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main() 