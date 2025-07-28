"""
Inference script for OpenVLA model
Based on the research paper: https://arxiv.org/abs/2406.09246

This script provides inference capabilities for the trained OpenVLA model,
including quantization support for memory-efficient deployment.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import time
from typing import List, Optional, Tuple, Dict, Any
import argparse
import json
import os
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch.nn.functional as F

from openvla_architecture import OpenVLA, OpenVLAConfig, ActionTokenizer

class OpenVLAInference:
    """
    Inference class for OpenVLA model
    """
    def __init__(self, model_path: str, device: str = 'cuda', 
                 use_quantization: bool = False, precision: str = 'bfloat16'):
        self.device = device
        self.use_quantization = use_quantization
        self.precision = precision
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup for inference
        self.model.eval()
        
        # Performance tracking
        self.inference_times = []
        
    def _load_model(self, model_path: str) -> OpenVLA:
        """Load OpenVLA model from checkpoint"""
        print(f"Loading OpenVLA model from {model_path}...")
        
        # Create model
        config = OpenVLAConfig()
        model = OpenVLA(config)
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {model_path}")
        else:
            print(f"Warning: Checkpoint {model_path} not found. Using random weights.")
        
        # Move to device
        model = model.to(self.device)
        
        # Apply quantization if requested
        if self.use_quantization:
            model = self._apply_quantization(model)
        
        # Apply precision
        if self.precision == 'bfloat16':
            model = model.to(torch.bfloat16)
        elif self.precision == 'float16':
            model = model.to(torch.float16)
        
        return model
    
    def _apply_quantization(self, model: OpenVLA) -> OpenVLA:
        """Apply quantization to model for memory efficiency"""
        print("Applying quantization...")
        
        # Dynamic quantization for linear layers
        model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        Args:
            image: (H, W, C) numpy array
        Returns:
            processed_image: (1, C, H, W) torch tensor
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
        
        return image_tensor
    
    def predict_action(self, image: np.ndarray, instruction: str,
                      do_sample: bool = False, temperature: float = 1.0) -> np.ndarray:
        """
        Predict robot action given image and instruction
        Args:
            image: (H, W, C) numpy array
            instruction: Text instruction for the robot
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
        Returns:
            action: (7,) numpy array - 7-DoF robot action
        """
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        processed_image = processed_image.to(self.device)
        
        # Format instruction
        formatted_instruction = f"In: What action should the robot take to {instruction}?\nOut:"
        
        # Predict action
        with torch.no_grad():
            actions = self.model.predict_action(
                images=processed_image,
                text_prompts=[formatted_instruction],
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=7
            )
        
        # Convert to numpy
        action = actions.cpu().numpy()[0]  # (7,)
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return action
    
    def batch_predict(self, images: List[np.ndarray], instructions: List[str],
                     do_sample: bool = False, temperature: float = 1.0) -> np.ndarray:
        """
        Predict actions for a batch of images and instructions
        Args:
            images: List of (H, W, C) numpy arrays
            instructions: List of text instructions
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
        Returns:
            actions: (batch_size, 7) numpy array
        """
        # Preprocess images
        processed_images = []
        for image in images:
            processed_image = self.preprocess_image(image)
            processed_images.append(processed_image)
        
        processed_images = torch.cat(processed_images, dim=0)
        processed_images = processed_images.to(self.device)
        
        # Format instructions
        formatted_instructions = [
            f"In: What action should the robot take to {instruction}?\nOut:"
            for instruction in instructions
        ]
        
        # Predict actions
        with torch.no_grad():
            actions = self.model.predict_action(
                images=processed_images,
                text_prompts=formatted_instructions,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=7
            )
        
        return actions.cpu().numpy()
    
    def get_inference_stats(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'total_inferences': len(times),
            'inference_fps': 1.0 / float(np.mean(times))
        }

class RobotController:
    """
    Robot controller interface for OpenVLA
    """
    def __init__(self, robot_type: str = "widowx"):
        self.robot_type = robot_type
        self.is_connected = False
        
        # Robot-specific parameters
        if robot_type == "widowx":
            self.action_dim = 7
            self.control_frequency = 5  # Hz
        elif robot_type == "franka":
            self.action_dim = 7
            self.control_frequency = 10  # Hz
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
    
    def connect(self):
        """Connect to robot"""
        # Connect to robot using the robot integration module
        from robot_integration import create_robot_interface
        
        try:
            self.robot = create_robot_interface(
                robot_type=self.robot_type,
                **self.robot_config
            )
            self.is_connected = self.robot.connect()
            if self.is_connected:
                print(f"Successfully connected to {self.robot_type} robot")
            else:
                print(f"Failed to connect to {self.robot_type} robot")
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            self.is_connected = False
    
    def disconnect(self):
        """Disconnect from robot"""
        if self.is_connected:
            print("Disconnecting from robot...")
            self.is_connected = False
            print("Robot disconnected!")
    
    def execute_action(self, action: np.ndarray):
        """
        Execute robot action
        Args:
            action: (7,) numpy array - 7-DoF robot action
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected!")
        
        # Execute action using robot interface
        success = self.robot.execute_action(action)
        if not success:
            print(f"Failed to execute action: {action}")
        
        # Simulate execution time
        time.sleep(1.0 / self.control_frequency)
    
    def get_observation(self) -> np.ndarray:
        """
        Get current robot observation (image)
        Returns:
            image: (H, W, C) numpy array
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected!")
        
        # Placeholder - implement actual camera capture
        # In practice, you would capture from robot's camera
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="OpenVLA Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to OpenVLA model checkpoint")
    parser.add_argument("--robot_type", type=str, default="widowx",
                       choices=["widowx", "franka"],
                       help="Type of robot to control")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    parser.add_argument("--quantization", action="store_true",
                       help="Use quantization for memory efficiency")
    parser.add_argument("--precision", type=str, default="bfloat16",
                       choices=["bfloat16", "float16", "float32"],
                       help="Model precision")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = OpenVLAInference(
        model_path=args.model_path,
        device=args.device,
        use_quantization=args.quantization,
        precision=args.precision
    )
    
    # Initialize robot controller
    robot = RobotController(robot_type=args.robot_type)
    
    if args.interactive:
        # Interactive mode
        print("Starting interactive robot control...")
        print("Type 'quit' to exit")
        
        try:
            robot.connect()
            
            while True:
                # Get user instruction
                instruction = input("Enter robot instruction: ")
                if instruction.lower() == 'quit':
                    break
                
                # Get robot observation
                image = robot.get_observation()
                
                # Predict action
                action = inference.predict_action(image, instruction)
                
                # Execute action
                robot.execute_action(action)
                
                # Print stats
                stats = inference.get_inference_stats()
                print(f"Inference time: {stats.get('mean_inference_time', 0):.3f}s")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            robot.disconnect()
    else:
        # Demo mode
        print("Running demo inference...")
        
        # Example images and instructions
        demo_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        demo_instructions = [
            "pick up the red cup",
            "place the object in the bowl",
            "move the block to the left"
        ]
        
        # Batch prediction
        actions = inference.batch_predict(demo_images, demo_instructions)
        
        print("Demo predictions:")
        for i, (instruction, action) in enumerate(zip(demo_instructions, actions)):
            print(f"  {i+1}. {instruction}: {action}")
        
        # Print stats
        stats = inference.get_inference_stats()
        print(f"\nInference statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 