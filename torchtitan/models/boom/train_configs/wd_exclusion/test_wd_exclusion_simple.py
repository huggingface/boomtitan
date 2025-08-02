#!/usr/bin/env python3
"""Simple test script for weight decay exclusion feature"""

import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

from torchtitan.config import ConfigManager
from torchtitan.models.boom import boom_configs, Transformer
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import init_logger
import torch

def test_weight_decay_exclusion():
    # Initialize logger (will use LOGLEVEL env var)
    init_logger()
    
    # Parse config
    config_manager = ConfigManager()
    config = config_manager.parse_args([
        "--job.config_file", "torchtitan/models/boom/train_configs/wd_exclusion/poc/wd_exclusion_embeddings_smollm135M.toml"
    ])
    
    # Build model
    model_args = boom_configs["smollm135M"]
    model_args.update_from_config(config)
    
    with torch.device("meta"):
        model = Transformer(model_args)
    
    # Move to CPU for testing
    model.to_empty(device="cpu")
    model.init_weights()
    
    # Create dummy parallel dims
    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1, 
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1
    )
    
    # Build optimizer - this should trigger debug logs
    print("\nBuilding optimizer with weight decay exclusion...")
    optimizer = build_optimizers([model], config.optimizer, parallel_dims)
    
    # Check parameter groups
    print(f"\nOptimizer has {len(optimizer.optimizers)} optimizer(s)")
    for i, opt in enumerate(optimizer.optimizers):
        print(f"\nOptimizer {i} has {len(opt.param_groups)} parameter group(s)")
        for j, group in enumerate(opt.param_groups):
            print(f"  Group {j}: {len(group['params'])} parameters, weight_decay={group.get('weight_decay', 'not set')}")
    
    # Check if output layer exists and which group it's in
    print("\nChecking for output layer...")
    output_found = False
    for name, param in model.named_parameters():
        if 'output' in name:
            output_found = True
            print(f"Found: {name}")
            # Find which group it's in
            for i, opt in enumerate(optimizer.optimizers):
                for j, group in enumerate(opt.param_groups):
                    if any(p is param for p in group['params']):
                        print(f"  -> In group {j} with weight_decay={group.get('weight_decay', 'not set')}")
    
    if not output_found:
        print("No output layer found (model might use tied embeddings)")

if __name__ == "__main__":
    test_weight_decay_exclusion()