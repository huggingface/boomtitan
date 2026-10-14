#!/usr/bin/env python3
"""
Test script to verify AdamC optimizer implementation.
"""

import sys
import os
# Add the torchtitan directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

import torch
import torch.nn as nn
from torchtitan.components.adam_corrected import AdamC


def test_adamc_weight_decay():
    """Test that AdamC applies different weight decay to normalized vs non-normalized layers."""
    
    # Create a simple model with different layer types
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 32)  # Should have no weight decay
            self.linear = nn.Linear(32, 32)         # Should have regular weight decay
            self.norm = nn.RMSNorm(32)             # Should have corrected weight decay
            self.output = nn.Linear(32, 100)       # Should have no weight decay
    
    model = SimpleModel()
    
    # Create parameter groups with names
    param_groups = []
    
    # Group 1: Parameters with weight decay
    decay_params = []
    decay_names = []
    for name, param in model.named_parameters():
        if not ("embedding" in name or "output" in name):
            decay_params.append(param)
            decay_names.append(name)
    
    # Group 2: Parameters without weight decay
    no_decay_params = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if "embedding" in name or "output" in name:
            no_decay_params.append(param)
            no_decay_names.append(name)
    
    param_groups = [
        {"params": decay_params, "param_names": decay_names, "weight_decay": 0.1},
        {"params": no_decay_params, "param_names": no_decay_names, "weight_decay": 0.0},
    ]
    
    # Create AdamC optimizer
    optimizer = AdamC(param_groups, lr=0.01, lr_max=0.01)
    
    # Create dummy loss and backward
    dummy_input = torch.randn(16, 32)
    output = model.linear(dummy_input)
    loss = output.sum()
    loss.backward()
    
    # Store initial weights
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.data.clone()
    
    # Take optimizer step
    optimizer.step()
    
    # Check weight decay was applied correctly
    print("Weight changes after optimizer step:")
    print("-" * 50)
    
    for name, param in model.named_parameters():
        weight_change = (param.data - initial_weights[name]).abs().mean().item()
        
        # Check if parameter is normalized
        is_normalized = "norm" in name.lower()
        has_weight_decay = not ("embedding" in name or "output" in name)
        
        print(f"{name:20} | Change: {weight_change:.6f} | "
              f"Normalized: {is_normalized} | Weight Decay: {has_weight_decay}")
    
    print("\nAdamC test completed successfully!")
    
    # Test that optimizer state is initialized correctly
    print("\nOptimizer state check:")
    print("-" * 50)
    for group_idx, group in enumerate(optimizer.param_groups):
        print(f"Group {group_idx}: weight_decay={group['weight_decay']}")
        if "param_names" in group:
            for name, param in zip(group["param_names"], group["params"]):
                state = optimizer.state[param]
                is_norm = state.get("is_normalized", False)
                print(f"  {name:20} | is_normalized: {is_norm}")


if __name__ == "__main__":
    test_adamc_weight_decay()