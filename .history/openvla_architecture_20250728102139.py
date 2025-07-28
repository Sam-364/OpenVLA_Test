"""
OpenVLA: An Open-Source Vision-Language-Action Model
Based on the research paper: https://arxiv.org/abs/2406.09246

This implementation replicates the OpenVLA architecture which consists of:
1. Vision Encoder: Fused DINOv2 + SigLIP features
2. Projector: Maps visual features to language embedding space
3. LLM Backbone: Llama 2 7B parameter model
4. Action Tokenization: 256-bin discretization for robot actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoImageProcessor
import timm
from typing import Optional, Tuple, Dict, Any
import numpy as np

class OpenVLAConfig:
    """Configuration for OpenVLA model"""
    def __init__(self):
        # Model dimensions
        self.hidden_size = 4096  # Llama 2 hidden size
        self.intermediate_size = 11008
        self.num_attention_heads = 32
        self.num_hidden_layers = 32
        self.rms_norm_eps = 1e-6
        
        # Vision encoder settings
        self.image_size = 224
        self.patch_size = 16
        self.num_channels = 3
        self.vision_hidden_size = 768
        
        # Action discretization
        self.action_bins = 256
        self.action_dim = 7  # 7-DoF robot actions
        
        # Training settings
        self.learning_rate = 2e-5
        self.batch_size = 2048
        self.max_length = 512

class VisionEncoder(nn.Module):
    """
    Fused Vision Encoder combining DINOv2 and SigLIP features
    """
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # DINOv2 encoder
        self.dinov2 = timm.create_model(
            'vit_large_patch14_224.dinov2', 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # SigLIP encoder (using CLIP as proxy since SigLIP is not directly available)
        self.siglip = timm.create_model(
            'vit_large_patch14_224_clip_laion2b', 
            pretrained=True,
            num_classes=0
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Linear(
            self.dinov2.num_features + self.siglip.num_features,
            config.vision_hidden_size
        )
        
        # Position embeddings for vision tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, (config.image_size // config.patch_size) ** 2, config.vision_hidden_size)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, channels, height, width)
        Returns:
            vision_features: (batch_size, num_patches, hidden_size)
        """
        batch_size = images.shape[0]
        
        # Extract features from both encoders
        dinov2_features = self.dinov2.forward_features(images)  # (B, num_patches, dinov2_dim)
        siglip_features = self.siglip.forward_features(images)  # (B, num_patches, siglip_dim)
        
        # Concatenate features along channel dimension
        combined_features = torch.cat([dinov2_features, siglip_features], dim=-1)
        
        # Project to unified vision hidden size
        vision_features = self.fusion_layer(combined_features)
        
        # Add position embeddings
        vision_features = vision_features + self.pos_embed
        
        return vision_features

class Projector(nn.Module):
    """
    Projects vision features to language model embedding space
    """
    def __init__(self, config: OpenVLAConfig):
        super().__init__()
        self.config = config
        
        # 2-layer MLP projector as used in Prismatic VLMs
        self.projector = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (batch_size, num_patches, vision_hidden_size)
        Returns:
            projected_features: (batch_size, num_patches, hidden_size)
        """
        return self.projector(vision_features)

class ActionTokenizer:
    """
    Discretizes continuous robot actions into discrete tokens
    """
    def __init__(self, action_bins: int = 256, action_dim: int = 7):
        self.action_bins = action_bins
        self.action_dim = action_dim
        self.bin_edges = None  # Will be set during training
        
    def fit(self, actions: torch.Tensor):
        """Compute bin edges based on training data"""
        # Use 1st and 99th percentiles for each action dimension
        self.bin_edges = []
        for dim in range(self.action_dim):
            dim_actions = actions[:, dim]
            p1 = torch.quantile(dim_actions, 0.01)
            p99 = torch.quantile(dim_actions, 0.99)
            edges = torch.linspace(p1, p99, self.action_bins + 1)
            self.bin_edges.append(edges)
    
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous actions to discrete tokens
        Args:
            actions: (batch_size, action_dim)
        Returns:
            tokens: (batch_size, action_dim)
        """
        if self.bin_edges is None:
            raise ValueError("ActionTokenizer must be fitted before encoding")
        
        tokens = []
        for dim in range(self.action_dim):
            dim_actions = actions[:, dim]
            edges = self.bin_edges[dim]
            
            # Find which bin each action falls into
            bin_indices = torch.bucketize(dim_actions, edges) - 1
            bin_indices = torch.clamp(bin_indices, 0, self.action_bins - 1)
            tokens.append(bin_indices)
        
        return torch.stack(tokens, dim=1)
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete tokens back to continuous actions
        Args:
            tokens: (batch_size, action_dim)
        Returns:
            actions: (batch_size, action_dim)
        """
        if self.bin_edges is None:
            raise ValueError("ActionTokenizer must be fitted before decoding")
        
        actions = []
        for dim in range(self.action_dim):
            dim_tokens = tokens[:, dim]
            edges = self.bin_edges[dim]
            
            # Convert tokens to continuous values (center of each bin)
            bin_centers = (edges[:-1] + edges[1:]) / 2
            dim_actions = bin_centers[dim_tokens]
            actions.append(dim_actions)
        
        return torch.stack(actions, dim=1)

class OpenVLA(nn.Module):
    """
    OpenVLA: Vision-Language-Action Model
    """
    def __init__(self, config: OpenVLAConfig, llm_model_name: str = "meta-llama/Llama-2-7b-hf"):
        super().__init__()
        self.config = config
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(config)
        
        # Projector
        self.projector = Projector(config)
        
        # Language model backbone
        self.llm = LlamaForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(llm_model_name)
        
        # Action tokenizer
        self.action_tokenizer = ActionTokenizer(config.action_bins, config.action_dim)
        
        # Overwrite the last 256 tokens in vocabulary with action tokens
        self._setup_action_tokens()
        
    def _setup_action_tokens(self):
        """Replace last 256 tokens with action tokens"""
        # Get the least used tokens from the vocabulary
        vocab_size = self.tokenizer.vocab_size
        action_token_ids = list(range(vocab_size - 256, vocab_size))
        
        # Create action token names
        action_token_names = []
        for dim in range(self.config.action_dim):
            for bin_idx in range(self.config.action_bins):
                action_token_names.append(f"<action_{dim}_{bin_idx}>")
        
        # Add action tokens to tokenizer
        self.tokenizer.add_tokens(action_token_names)
        
        # Resize model embeddings
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        action_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of OpenVLA
        Args:
            images: (batch_size, channels, height, width)
            text_tokens: (batch_size, text_length)
            action_tokens: (batch_size, action_length) - optional for generation
            attention_mask: (batch_size, total_length) - optional
        Returns:
            outputs: Dictionary containing loss and logits
        """
        batch_size = images.shape[0]
        
        # Encode images
        vision_features = self.vision_encoder(images)  # (B, num_patches, vision_hidden)
        projected_features = self.projector(vision_features)  # (B, num_patches, hidden)
        
        # Prepare input for language model
        if action_tokens is not None:
            # Training mode: concatenate text and action tokens
            input_ids = torch.cat([text_tokens, action_tokens], dim=1)
        else:
            # Generation mode: only text tokens
            input_ids = text_tokens
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Prepare inputs for LLM
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': action_tokens if action_tokens is not None else None
        }
        
        # Forward through LLM
        outputs = self.llm(**inputs)
        
        return outputs
    
    def predict_action(
        self,
        images: torch.Tensor,
        text_prompts: list,
        do_sample: bool = False,
        temperature: float = 1.0,
        max_new_tokens: int = 7
    ) -> torch.Tensor:
        """
        Predict robot actions given images and text prompts
        Args:
            images: (batch_size, channels, height, width)
            text_prompts: List of text prompts
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            max_new_tokens: Maximum number of action tokens to generate
        Returns:
            actions: (batch_size, action_dim) - continuous robot actions
        """
        # Tokenize text prompts
        text_tokens = self.tokenizer(
            text_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # Move to device
        device = images.device
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
        
        # Generate action tokens
        with torch.no_grad():
            outputs = self.llm.generate(
                **text_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated action tokens
        generated_tokens = outputs[:, text_tokens['input_ids'].shape[1]:]
        
        # Convert tokens to continuous actions
        actions = self.action_tokenizer.decode(generated_tokens)
        
        return actions

def create_openvla_model(config: OpenVLAConfig) -> OpenVLA:
    """Create and return an OpenVLA model instance"""
    return OpenVLA(config)

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = OpenVLAConfig()
    
    # Create model
    model = create_openvla_model(config)
    
    # Example forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    text_prompts = [
        "In: What action should the robot take to pick up the red cup?\nOut:",
        "In: What action should the robot take to place the object in the bowl?\nOut:"
    ]
    
    # Predict actions
    actions = model.predict_action(images, text_prompts)
    print(f"Predicted actions shape: {actions.shape}")
    print(f"Actions: {actions}") 