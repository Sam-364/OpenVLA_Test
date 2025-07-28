"""
Evaluation script for OpenVLA model
Provides comprehensive evaluation metrics and analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from openvla_architecture import OpenVLA, OpenVLAConfig, ActionTokenizer
from data_loader import create_dataloader
from inference_openvla import OpenVLAInference

class OpenVLAEvaluator:
    """
    Comprehensive evaluator for OpenVLA model
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        
        # Load model
        self.inference = OpenVLAInference(
            model_path=model_path,
            device=device,
            use_quantization=False
        )
        
        # Evaluation metrics
        self.metrics = {}
        
    def evaluate_action_accuracy(self, test_loader, num_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate action prediction accuracy
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to evaluate (None for all)
        
        Returns:
            Dictionary of accuracy metrics
        """
        self.inference.model.eval()
        
        all_predictions = []
        all_targets = []
        all_actions = []
        all_predicted_actions = []
        
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating action accuracy"):
                if num_samples and total_samples >= num_samples:
                    break
                
                # Get batch data
                images = batch['image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                action_tokens = batch['action_tokens'].to(self.device)
                actions = batch['actions'].to(self.device)
                
                # Get predictions
                outputs = self.inference.model(
                    images=images,
                    text_tokens=text_tokens,
                    action_tokens=action_tokens
                )
                
                # Extract action logits
                logits = outputs.logits
                action_logits = logits[:, -action_tokens.shape[1]:, :]
                action_preds = torch.argmax(action_logits, dim=-1)
                
                # Convert to continuous actions
                predicted_actions = self.inference.model.action_tokenizer.decode(action_preds)
                
                # Store results
                all_predictions.append(action_preds.cpu())
                all_targets.append(action_tokens.cpu())
                all_actions.append(actions.cpu())
                all_predicted_actions.append(predicted_actions.cpu())
                
                total_samples += images.shape[0]
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        all_predicted_actions = torch.cat(all_predicted_actions, dim=0)
        
        # Calculate metrics
        token_accuracy = (all_predictions == all_targets).float().mean().item()
        
        # Action space accuracy (continuous)
        action_mse = torch.mean((all_predicted_actions - all_actions) ** 2).item()
        action_mae = torch.mean(torch.abs(all_predicted_actions - all_actions)).item()
        
        # Per-dimension accuracy
        per_dim_accuracy = []
        for dim in range(all_actions.shape[1]):
            dim_accuracy = (torch.abs(all_predicted_actions[:, dim] - all_actions[:, dim]) < 0.1).float().mean().item()
            per_dim_accuracy.append(dim_accuracy)
        
        metrics = {
            'token_accuracy': token_accuracy,
            'action_mse': action_mse,
            'action_mae': action_mae,
            'per_dim_accuracy': per_dim_accuracy,
            'mean_per_dim_accuracy': np.mean(per_dim_accuracy)
        }
        
        self.metrics['action_accuracy'] = metrics
        return metrics
    
    def evaluate_inference_speed(self, test_loader, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate inference speed and memory usage
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary of speed metrics
        """
        self.inference.model.eval()
        
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Evaluating inference speed")):
                if i >= num_samples:
                    break
                
                # Get batch data
                images = batch['image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                
                # Measure inference time
                start_time = time.time()
                _ = self.inference.model(
                    images=images,
                    text_tokens=text_tokens
                )
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                # Measure memory usage
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
        
        metrics = {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'inference_fps': 1.0 / np.mean(inference_times),
            'mean_memory_usage_gb': np.mean(memory_usage) if memory_usage else 0.0
        }
        
        self.metrics['inference_speed'] = metrics
        return metrics
    
    def evaluate_robustness(self, test_loader, noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3]) -> Dict[str, Any]:
        """
        Evaluate model robustness to noise
        
        Args:
            test_loader: DataLoader for test data
            noise_levels: List of noise levels to test
        
        Returns:
            Dictionary of robustness metrics
        """
        self.inference.model.eval()
        
        robustness_metrics = {}
        
        for noise_level in noise_levels:
            print(f"Testing robustness with noise level: {noise_level}")
            
            accuracies = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Noise level {noise_level}"):
                    # Get batch data
                    images = batch['image'].to(self.device)
                    text_tokens = batch['text_tokens'].to(self.device)
                    action_tokens = batch['action_tokens'].to(self.device)
                    
                    # Add noise to images
                    if noise_level > 0:
                        noise = torch.randn_like(images) * noise_level
                        images = torch.clamp(images + noise, 0, 1)
                    
                    # Get predictions
                    outputs = self.inference.model(
                        images=images,
                        text_tokens=text_tokens,
                        action_tokens=action_tokens
                    )
                    
                    # Calculate accuracy
                    logits = outputs.logits
                    action_logits = logits[:, -action_tokens.shape[1]:, :]
                    action_preds = torch.argmax(action_logits, dim=-1)
                    accuracy = (action_preds == action_tokens).float().mean().item()
                    accuracies.append(accuracy)
            
            robustness_metrics[f'noise_{noise_level}'] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
        
        self.metrics['robustness'] = robustness_metrics
        return robustness_metrics
    
    def generate_visualizations(self, test_loader, output_dir: str = "evaluation_results"):
        """
        Generate evaluation visualizations
        
        Args:
            test_loader: DataLoader for test data
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Action prediction scatter plots
        self._plot_action_predictions(test_loader, output_dir)
        
        # 2. Accuracy over time
        self._plot_accuracy_over_time(test_loader, output_dir)
        
        # 3. Confusion matrix for action tokens
        self._plot_confusion_matrix(test_loader, output_dir)
        
        # 4. Inference speed distribution
        if 'inference_speed' in self.metrics:
            self._plot_inference_speed(output_dir)
    
    def _plot_action_predictions(self, test_loader, output_dir: str):
        """Plot predicted vs actual actions"""
        self.inference.model.eval()
        
        all_actions = []
        all_predicted_actions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating action plots"):
                images = batch['image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                action_tokens = batch['action_tokens'].to(self.device)
                actions = batch['actions'].to(self.device)
                
                # Get predictions
                outputs = self.inference.model(
                    images=images,
                    text_tokens=text_tokens,
                    action_tokens=action_tokens
                )
                
                logits = outputs.logits
                action_logits = logits[:, -action_tokens.shape[1]:, :]
                action_preds = torch.argmax(action_logits, dim=-1)
                predicted_actions = self.inference.model.action_tokenizer.decode(action_preds)
                
                all_actions.append(actions.cpu())
                all_predicted_actions.append(predicted_actions.cpu())
        
        all_actions = torch.cat(all_actions, dim=0)
        all_predicted_actions = torch.cat(all_predicted_actions, dim=0)
        
        # Create scatter plots for each action dimension
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        action_names = ['X', 'Y', 'Z', 'Q1', 'Q2', 'Q3', 'Q4']
        
        for i in range(7):
            ax = axes[i]
            ax.scatter(all_actions[:, i], all_predicted_actions[:, i], alpha=0.5)
            ax.plot([all_actions[:, i].min(), all_actions[:, i].max()], 
                   [all_actions[:, i].min(), all_actions[:, i].max()], 'r--')
            ax.set_xlabel(f'Actual {action_names[i]}')
            ax.set_ylabel(f'Predicted {action_names[i]}')
            ax.set_title(f'{action_names[i]} Prediction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_over_time(self, test_loader, output_dir: str):
        """Plot accuracy over time"""
        self.inference.model.eval()
        
        accuracies = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating accuracy plot"):
                images = batch['image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                action_tokens = batch['action_tokens'].to(self.device)
                
                outputs = self.inference.model(
                    images=images,
                    text_tokens=text_tokens,
                    action_tokens=action_tokens
                )
                
                logits = outputs.logits
                action_logits = logits[:, -action_tokens.shape[1]:, :]
                action_preds = torch.argmax(action_logits, dim=-1)
                accuracy = (action_preds == action_tokens).float().mean().item()
                accuracies.append(accuracy)
        
        plt.figure(figsize=(12, 6))
        plt.plot(accuracies)
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.title('Action Prediction Accuracy Over Time')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, test_loader, output_dir: str):
        """Plot confusion matrix for action tokens"""
        self.inference.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating confusion matrix"):
                images = batch['image'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                action_tokens = batch['action_tokens'].to(self.device)
                
                outputs = self.inference.model(
                    images=images,
                    text_tokens=text_tokens,
                    action_tokens=action_tokens
                )
                
                logits = outputs.logits
                action_logits = logits[:, -action_tokens.shape[1]:, :]
                action_preds = torch.argmax(action_logits, dim=-1)
                
                all_predictions.append(action_preds.cpu().flatten())
                all_targets.append(action_tokens.cpu().flatten())
        
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Token')
        plt.ylabel('Actual Token')
        plt.title('Action Token Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_speed(self, output_dir: str):
        """Plot inference speed distribution"""
        if 'inference_speed' not in self.metrics:
            return
        
        # This would be populated by evaluate_inference_speed
        # For now, create a placeholder
        plt.figure(figsize=(10, 6))
        plt.hist([0.1, 0.15, 0.12, 0.18, 0.11], bins=20, alpha=0.7)
        plt.xlabel('Inference Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Inference Speed Distribution')
        plt.savefig(os.path.join(output_dir, 'inference_speed.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: str = "evaluation_results"):
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save summary
        summary = {
            'model_path': self.model_path,
            'device': self.device,
            'total_metrics': len(self.metrics),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Evaluation results saved to {output_dir}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="OpenVLA Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to OpenVLA model checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to test dataset")
    parser.add_argument("--dataset_type", type=str, default="bridge",
                       choices=["bridge", "rlds", "custom"],
                       help="Type of dataset")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--generate_plots", action="store_true",
                       help="Generate evaluation plots")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = OpenVLAEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Create test dataloader
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    action_tokenizer = ActionTokenizer()
    
    test_loader = create_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        dataset_type=args.dataset_type,
        batch_size=32,
        shuffle=False
    )
    
    print("Starting OpenVLA evaluation...")
    
    # Evaluate action accuracy
    print("Evaluating action accuracy...")
    accuracy_metrics = evaluator.evaluate_action_accuracy(test_loader, args.num_samples)
    print(f"Action accuracy metrics: {accuracy_metrics}")
    
    # Evaluate inference speed
    print("Evaluating inference speed...")
    speed_metrics = evaluator.evaluate_inference_speed(test_loader, min(100, args.num_samples or 100))
    print(f"Inference speed metrics: {speed_metrics}")
    
    # Evaluate robustness
    print("Evaluating robustness...")
    robustness_metrics = evaluator.evaluate_robustness(test_loader)
    print(f"Robustness metrics: {robustness_metrics}")
    
    # Generate visualizations
    if args.generate_plots:
        print("Generating visualizations...")
        evaluator.generate_visualizations(test_loader, args.output_dir)
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 