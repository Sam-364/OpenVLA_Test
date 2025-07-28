#!/usr/bin/env python3
"""
OpenVLA Training with Real-Time Performance Monitoring
Includes dataset integration and live metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import wandb
from tqdm import tqdm
import numpy as np
import time
import threading
import queue
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import subprocess
import psutil
import GPUtil
from datetime import datetime
import cv2
from PIL import Image
import io
import base64

from openvla_architecture import OpenVLA, OpenVLAConfig, ActionTokenizer
from data_loader import create_dataloader
from config import get_training_config

class RealTimeMonitor:
    """
    Real-time performance monitoring for OpenVLA training
    """
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.metrics_queue = queue.Queue()
        self.is_running = False
        self.monitor_thread = None
        
        # Performance metrics
        self.gpu_metrics = {}
        self.cpu_metrics = {}
        self.memory_metrics = {}
        self.training_metrics = {}
        
        # Visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('OpenVLA Training - Real-Time Monitoring', fontsize=16)
        
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
        print("Real-time monitoring stopped...")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Update visualization
                self._update_visualization()
                
                # Log to wandb
                self._log_to_wandb()
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Monitor first GPU
                gpu_util = gpu.load * 100
                gpu_memory_percent = gpu.memoryUtil * 100
                gpu_memory_used_gb = gpu.memoryUsed / 1024
                gpu_memory_total_gb = gpu.memoryTotal / 1024
                gpu_temperature = gpu.temperature
            else:
                gpu_util = 0
                gpu_memory_percent = 0
                gpu_memory_used_gb = 0
                gpu_memory_total_gb = 0
                gpu_temperature = 0
        except:
            gpu_util = 0
            gpu_memory_percent = 0
            gpu_memory_used_gb = 0
            gpu_memory_total_gb = 0
            gpu_temperature = 0
        
        # Store metrics
        self.cpu_metrics = {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count
        }
        
        self.memory_metrics = {
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb
        }
        
        self.gpu_metrics = {
            'gpu_util': gpu_util,
            'gpu_memory_percent': gpu_memory_percent,
            'gpu_memory_used_gb': gpu_memory_used_gb,
            'gpu_memory_total_gb': gpu_memory_total_gb,
            'gpu_temperature': gpu_temperature
        }
    
    def _update_visualization(self):
        """Update real-time visualization"""
        try:
            # Clear previous plots
            for ax in self.axes.flat:
                ax.clear()
            
            # CPU Usage
            self.axes[0, 0].bar(['CPU'], [self.cpu_metrics.get('cpu_percent', 0)])
            self.axes[0, 0].set_title('CPU Usage (%)')
            self.axes[0, 0].set_ylim(0, 100)
            
            # Memory Usage
            memory_percent = self.memory_metrics.get('memory_percent', 0)
            self.axes[0, 1].pie([memory_percent, 100-memory_percent], 
                               labels=['Used', 'Free'], autopct='%1.1f%%')
            self.axes[0, 1].set_title('Memory Usage')
            
            # GPU Usage
            gpu_util = self.gpu_metrics.get('gpu_util', 0)
            gpu_memory = self.gpu_metrics.get('gpu_memory_percent', 0)
            self.axes[1, 0].bar(['GPU Util', 'GPU Memory'], [gpu_util, gpu_memory])
            self.axes[1, 0].set_title('GPU Usage (%)')
            self.axes[1, 0].set_ylim(0, 100)
            
            # Training Metrics (if available)
            if self.training_metrics:
                loss = self.training_metrics.get('loss', 0)
                accuracy = self.training_metrics.get('action_accuracy', 0)
                self.axes[1, 1].plot([loss], [accuracy], 'ro', markersize=10)
                self.axes[1, 1].set_xlabel('Loss')
                self.axes[1, 1].set_ylabel('Action Accuracy')
                self.axes[1, 1].set_title('Training Progress')
                self.axes[1, 1].set_xlim(0, max(loss * 1.2, 1))
                self.axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
    
    def _log_to_wandb(self):
        """Log metrics to wandb"""
        try:
            if wandb.run:
                wandb.log({
                    'system/cpu_percent': self.cpu_metrics.get('cpu_percent', 0),
                    'system/memory_percent': self.memory_metrics.get('memory_percent', 0),
                    'system/memory_used_gb': self.memory_metrics.get('memory_used_gb', 0),
                    'system/gpu_util': self.gpu_metrics.get('gpu_util', 0),
                    'system/gpu_memory_percent': self.gpu_metrics.get('gpu_memory_percent', 0),
                    'system/gpu_memory_used_gb': self.gpu_metrics.get('gpu_memory_used_gb', 0),
                    'system/gpu_temperature': self.gpu_metrics.get('gpu_temperature', 0)
                })
        except Exception as e:
            print(f"Error logging to wandb: {e}")
    
    def update_training_metrics(self, metrics: Dict[str, float]):
        """Update training metrics"""
        self.training_metrics.update(metrics)

class OpenVLARealTimeTrainer:
    """
    OpenVLA trainer with real-time monitoring
    """
    
    def __init__(self, config: OpenVLAConfig, data_path: str, 
                 dataset_type: str = 'bridge', device: str = 'cuda'):
        self.config = config
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.device = device
        
        # Initialize model
        self.model = OpenVLA(config.model)
        self.model = self.model.to(device)
        
        # Initialize data loader
        self.train_loader = create_dataloader(
            data_path=data_path,
            tokenizer=self.model.tokenizer,
            action_tokenizer=self.model.action_tokenizer,
            dataset_type=dataset_type,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            shuffle=True
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize monitor
        self.monitor = RealTimeMonitor()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize wandb
        wandb.init(
            project="openvla-realtime",
            config={
                'model': config.model.to_dict(),
                'training': config.training.to_dict()
            },
            name=f"openvla-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with real-time monitoring"""
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
            
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
            
            # Update real-time monitor
            self.monitor.update_training_metrics({
                'loss': loss.item(),
                'action_accuracy': action_accuracy.item()
            })
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'action_acc': f"{action_accuracy.item():.4f}"
            })
            
            # Log to wandb
            wandb.log({
                'train/loss': loss.item(),
                'train/action_accuracy': action_accuracy.item(),
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/step': self.current_step,
                'train/epoch': self.current_epoch
            })
            
            self.current_step += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_action_accuracy = total_action_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'action_accuracy': avg_action_accuracy
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.training.checkpoint_dir, "openvla_latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if metrics['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['loss']
            best_path = os.path.join(self.config.training.checkpoint_dir, "openvla_best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs: int = 27):
        """Main training loop with real-time monitoring"""
        print(f"Starting OpenVLA training with real-time monitoring...")
        print(f"Dataset: {self.data_path}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        
        # Start real-time monitoring
        self.monitor.start_monitoring()
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                train_metrics = self.train_epoch()
                
                # Print metrics
                print(f"Epoch {epoch}:")
                for key, value in train_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, train_metrics)
                
                # Check if we should stop training (action accuracy > 95%)
                if train_metrics['action_accuracy'] > 0.95:
                    print(f"Action accuracy > 95% reached! Stopping training.")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
            wandb.finish()
            print("Training completed!")

def download_bridgedata_v2(data_dir: str = "data/bridgedata_v2"):
    """
    Download BridgeData V2 dataset
    """
    print("Downloading BridgeData V2...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download URLs (you'll need to update these with actual download links)
    download_urls = {
        'teleop': "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/teleop_data.zip",
        'scripted': "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/scripted_data.zip"
    }
    
    for data_type, url in download_urls.items():
        print(f"Downloading {data_type} data...")
        output_path = os.path.join(data_dir, f"{data_type}_data.zip")
        
        try:
            # Use wget or curl to download
            subprocess.run([
                "wget", "-O", output_path, url
            ], check=True)
            
            # Extract
            subprocess.run([
                "unzip", "-o", output_path, "-d", data_dir
            ], check=True)
            
            print(f"Downloaded and extracted {data_type} data")
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {data_type} data: {e}")
            print("Please download manually from: https://rail.eecs.berkeley.edu/datasets/bridge_release/data/")
    
    return data_dir

def setup_dataset(data_path: str, dataset_type: str = 'bridge'):
    """
    Set up dataset for training
    """
    if dataset_type == 'bridge':
        if not os.path.exists(data_path):
            print(f"Dataset not found at {data_path}")
            print("Downloading BridgeData V2...")
            data_path = download_bridgedata_v2()
    
    return data_path

def main():
    parser = argparse.ArgumentParser(description="OpenVLA Training with Real-Time Monitoring")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, default="bridge",
                       choices=["bridge", "rlds", "custom"],
                       help="Type of dataset")
    parser.add_argument("--config", type=str, default="configs/training_config.json",
                       help="Path to training configuration")
    parser.add_argument("--epochs", type=int, default=27,
                       help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to train on")
    parser.add_argument("--download_dataset", action="store_true",
                       help="Download BridgeData V2 if not available")
    
    args = parser.parse_args()
    
    # Load configuration
    from config import OpenVLAConfig
    config = OpenVLAConfig.load_config(args.config)
    
    # Update configuration
    config.training.data_path = args.data_path
    config.training.dataset_type = args.dataset_type
    config.training.num_epochs = args.epochs
    
    # Set up dataset
    if args.download_dataset:
        data_path = setup_dataset(args.data_path, args.dataset_type)
    else:
        data_path = args.data_path
    
    # Create trainer
    trainer = OpenVLARealTimeTrainer(
        config=config,
        data_path=data_path,
        dataset_type=args.dataset_type,
        device=args.device
    )
    
    # Start training
    trainer.train(num_epochs=args.epochs)

if __name__ == "__main__":
    main() 