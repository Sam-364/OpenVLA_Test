#!/usr/bin/env python3
"""
Real-Time Performance Monitor for OpenVLA Training
Provides live monitoring of system resources, training metrics, and model performance
"""

import time
import threading
import queue
import psutil
import GPUtil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse
import subprocess
import cv2
from PIL import Image
import io
import base64

class RealTimeMonitor:
    """
    Real-time performance monitoring for OpenVLA training
    """
    
    def __init__(self, log_interval: int = 1, save_metrics: bool = True):
        self.log_interval = log_interval
        self.save_metrics = save_metrics
        self.is_running = False
        self.monitor_thread = None
        
        # Performance metrics storage
        self.metrics_history = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': [],
            'gpu_util': [],
            'gpu_memory_percent': [],
            'gpu_memory_used_gb': [],
            'gpu_temperature': [],
            'training_loss': [],
            'training_accuracy': [],
            'learning_rate': []
        }
        
        # Current metrics
        self.current_metrics = {}
        
        # Visualization
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('OpenVLA Training - Real-Time Monitoring', fontsize=16)
        
        # Setup plots
        self._setup_plots()
        
        # Metrics queue for thread communication
        self.metrics_queue = queue.Queue()
        
        # Save directory
        self.save_dir = "monitoring_logs"
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _setup_plots(self):
        """Setup matplotlib plots"""
        # CPU Usage
        self.axes[0, 0].set_title('CPU Usage (%)')
        self.axes[0, 0].set_ylim(0, 100)
        self.cpu_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2)
        
        # Memory Usage
        self.axes[0, 1].set_title('Memory Usage (%)')
        self.axes[0, 1].set_ylim(0, 100)
        self.memory_line, = self.axes[0, 1].plot([], [], 'r-', linewidth=2)
        
        # GPU Usage
        self.axes[0, 2].set_title('GPU Usage (%)')
        self.axes[0, 2].set_ylim(0, 100)
        self.gpu_line, = self.axes[0, 2].plot([], [], 'g-', linewidth=2)
        
        # Training Loss
        self.axes[1, 0].set_title('Training Loss')
        self.loss_line, = self.axes[1, 0].plot([], [], 'orange', linewidth=2)
        
        # Training Accuracy
        self.axes[1, 1].set_title('Action Accuracy')
        self.accuracy_line, = self.axes[1, 1].plot([], [], 'purple', linewidth=2)
        self.axes[1, 1].set_ylim(0, 1)
        
        # Learning Rate
        self.axes[1, 2].set_title('Learning Rate')
        self.lr_line, = self.axes[1, 2].plot([], [], 'brown', linewidth=2)
        
        plt.tight_layout()
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Real-time monitoring started...")
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, interval=self.log_interval * 1000,
            blit=False
        )
        plt.show()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if self.save_metrics:
            self._save_metrics()
        
        print("Real-time monitoring stopped...")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Update metrics history
                self._update_metrics_history()
                
                # Put metrics in queue for plotting
                self.metrics_queue.put(self.current_metrics.copy())
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
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
        
        # Store current metrics
        self.current_metrics = {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'gpu_util': gpu_util,
            'gpu_memory_percent': gpu_memory_percent,
            'gpu_memory_used_gb': gpu_memory_used_gb,
            'gpu_memory_total_gb': gpu_memory_total_gb,
            'gpu_temperature': gpu_temperature
        }
    
    def _update_metrics_history(self):
        """Update metrics history"""
        if not self.current_metrics:
            return
        
        # Add timestamp
        self.metrics_history['timestamps'].append(self.current_metrics['timestamp'])
        
        # Add system metrics
        self.metrics_history['cpu_percent'].append(self.current_metrics['cpu_percent'])
        self.metrics_history['memory_percent'].append(self.current_metrics['memory_percent'])
        self.metrics_history['memory_used_gb'].append(self.current_metrics['memory_used_gb'])
        self.metrics_history['gpu_util'].append(self.current_metrics['gpu_util'])
        self.metrics_history['gpu_memory_percent'].append(self.current_metrics['gpu_memory_percent'])
        self.metrics_history['gpu_memory_used_gb'].append(self.current_metrics['gpu_memory_used_gb'])
        self.metrics_history['gpu_temperature'].append(self.current_metrics['gpu_temperature'])
        
        # Keep only last 1000 points to prevent memory issues
        max_history = 1000
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_history:
                self.metrics_history[key] = self.metrics_history[key][-max_history:]
    
    def update_training_metrics(self, metrics: Dict[str, float]):
        """Update training metrics from training process"""
        self.current_metrics.update(metrics)
        
        # Add to history
        if 'loss' in metrics:
            self.metrics_history['training_loss'].append(metrics['loss'])
        if 'action_accuracy' in metrics:
            self.metrics_history['training_accuracy'].append(metrics['action_accuracy'])
        if 'learning_rate' in metrics:
            self.metrics_history['learning_rate'].append(metrics['learning_rate'])
    
    def _update_plots(self, frame):
        """Update all plots"""
        try:
            # Get latest metrics from queue
            while not self.metrics_queue.empty():
                latest_metrics = self.metrics_queue.get_nowait()
            
            # Update CPU plot
            if self.metrics_history['timestamps']:
                timestamps = [t.timestamp() for t in self.metrics_history['timestamps']]
                self.cpu_line.set_data(timestamps, self.metrics_history['cpu_percent'])
                self.axes[0, 0].relim()
                self.axes[0, 0].autoscale_view()
            
            # Update Memory plot
            if self.metrics_history['timestamps']:
                self.memory_line.set_data(timestamps, self.metrics_history['memory_percent'])
                self.axes[0, 1].relim()
                self.axes[0, 1].autoscale_view()
            
            # Update GPU plot
            if self.metrics_history['timestamps']:
                self.gpu_line.set_data(timestamps, self.metrics_history['gpu_util'])
                self.axes[0, 2].relim()
                self.axes[0, 2].autoscale_view()
            
            # Update Training Loss plot
            if self.metrics_history['training_loss']:
                loss_timestamps = timestamps[-len(self.metrics_history['training_loss']):]
                self.loss_line.set_data(loss_timestamps, self.metrics_history['training_loss'])
                self.axes[1, 0].relim()
                self.axes[1, 0].autoscale_view()
            
            # Update Training Accuracy plot
            if self.metrics_history['training_accuracy']:
                acc_timestamps = timestamps[-len(self.metrics_history['training_accuracy']):]
                self.accuracy_line.set_data(acc_timestamps, self.metrics_history['training_accuracy'])
                self.axes[1, 1].relim()
                self.axes[1, 1].autoscale_view()
            
            # Update Learning Rate plot
            if self.metrics_history['learning_rate']:
                lr_timestamps = timestamps[-len(self.metrics_history['learning_rate']):]
                self.lr_line.set_data(lr_timestamps, self.metrics_history['learning_rate'])
                self.axes[1, 2].relim()
                self.axes[1, 2].autoscale_view()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def _save_metrics(self):
        """Save metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"metrics_{timestamp}.json")
        
        # Convert timestamps to strings for JSON serialization
        metrics_to_save = {}
        for key, values in self.metrics_history.items():
            if key == 'timestamps':
                metrics_to_save[key] = [t.isoformat() for t in values]
            else:
                metrics_to_save[key] = values
        
        with open(filename, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"Metrics saved to: {filename}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get current system summary"""
        if not self.current_metrics:
            return {}
        
        return {
            'cpu_usage': f"{self.current_metrics['cpu_percent']:.1f}%",
            'memory_usage': f"{self.current_metrics['memory_percent']:.1f}%",
            'memory_used': f"{self.current_metrics['memory_used_gb']:.1f}GB",
            'gpu_usage': f"{self.current_metrics['gpu_util']:.1f}%",
            'gpu_memory': f"{self.current_metrics['gpu_memory_percent']:.1f}%",
            'gpu_memory_used': f"{self.current_metrics['gpu_memory_used_gb']:.1f}GB",
            'gpu_temperature': f"{self.current_metrics['gpu_temperature']:.1f}°C"
        }

def create_performance_report(metrics_file: str) -> str:
    """Create a performance report from saved metrics"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    report = []
    report.append("OpenVLA Training Performance Report")
    report.append("=" * 50)
    
    # System performance
    if 'cpu_percent' in metrics and metrics['cpu_percent']:
        avg_cpu = np.mean(metrics['cpu_percent'])
        max_cpu = np.max(metrics['cpu_percent'])
        report.append(f"CPU Usage: Avg {avg_cpu:.1f}%, Max {max_cpu:.1f}%")
    
    if 'memory_percent' in metrics and metrics['memory_percent']:
        avg_memory = np.mean(metrics['memory_percent'])
        max_memory = np.max(metrics['memory_percent'])
        report.append(f"Memory Usage: Avg {avg_memory:.1f}%, Max {max_memory:.1f}%")
    
    if 'gpu_util' in metrics and metrics['gpu_util']:
        avg_gpu = np.mean(metrics['gpu_util'])
        max_gpu = np.max(metrics['gpu_util'])
        report.append(f"GPU Usage: Avg {avg_gpu:.1f}%, Max {max_gpu:.1f}%")
    
    # Training performance
    if 'training_loss' in metrics and metrics['training_loss']:
        final_loss = metrics['training_loss'][-1]
        min_loss = np.min(metrics['training_loss'])
        report.append(f"Training Loss: Final {final_loss:.4f}, Min {min_loss:.4f}")
    
    if 'training_accuracy' in metrics and metrics['training_accuracy']:
        final_acc = metrics['training_accuracy'][-1]
        max_acc = np.max(metrics['training_accuracy'])
        report.append(f"Action Accuracy: Final {final_acc:.4f}, Max {max_acc:.4f}")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Real-time performance monitor for OpenVLA")
    parser.add_argument("--interval", type=int, default=1,
                       help="Monitoring interval in seconds")
    parser.add_argument("--save_metrics", action="store_true",
                       help="Save metrics to file")
    parser.add_argument("--report", type=str,
                       help="Generate performance report from metrics file")
    
    args = parser.parse_args()
    
    if args.report:
        report = create_performance_report(args.report)
        print(report)
    else:
        monitor = RealTimeMonitor(
            log_interval=args.interval,
            save_metrics=args.save_metrics
        )
        
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            monitor.stop_monitoring()

if __name__ == "__main__":
    main() 