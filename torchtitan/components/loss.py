# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(
    pred: torch.Tensor, 
    labels: torch.Tensor,
    z_loss_weight: float = 0.0,
    return_z_loss: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Cross-entropy loss function with optional z-loss regularization.
    
    Z-loss encourages the log normalizer (log Z) of the softmax to stay close to 0,
    which helps stabilize training of large language models.
    
    Args:
        pred: Model predictions of shape [batch_size, seq_len, vocab_size]
        labels: Target labels of shape [batch_size, seq_len]
        z_loss_weight: Weight for the z-loss term (0 to disable, typically 1e-4)
        return_z_loss: If True and z_loss_weight > 0, returns (total_loss, z_loss_unweighted)
    
    Returns:
        If return_z_loss is False or z_loss_weight is 0: returns total loss
        If return_z_loss is True and z_loss_weight > 0: returns (total_loss, z_loss_unweighted)
    """
    # Flatten predictions and labels for cross-entropy computation
    pred_flat = pred.flatten(0, 1).float()
    labels_flat = labels.flatten(0, 1)
    
    # Compute standard cross-entropy loss
    ce_loss = torch.nn.functional.cross_entropy(pred_flat, labels_flat)
    
    if z_loss_weight > 0:
        # Compute z-loss: z_loss_weight * log(Z)^2 where Z is the partition function
        log_z = torch.logsumexp(pred_flat, dim=-1)
        z_loss_unweighted = torch.square(log_z).mean()
        total_loss = ce_loss + z_loss_weight * z_loss_unweighted
        
        if return_z_loss:
            return total_loss, z_loss_unweighted
        return total_loss
    
    if return_z_loss:
        return ce_loss, torch.tensor(0.0, device=ce_loss.device)
    return ce_loss


def build_cross_entropy_loss(job_config: JobConfig):
    # Check for incompatible configurations
    z_loss_weight = getattr(job_config.training, 'z_loss_weight', 0.0)
    
    if z_loss_weight > 0:
        if hasattr(job_config.parallelism, 'disable_loss_parallel'):
            if not job_config.parallelism.disable_loss_parallel:
                raise NotImplementedError(
                    "Z-loss is not yet implemented for loss_parallel=True (TP with sharded loss). "
                    "Please set --parallelism.disable_loss_parallel=true to use z-loss."
                )
        logger.info(f"Using cross-entropy loss with z-loss (weight={z_loss_weight})")
        # Return z-loss for logging when enabled
        loss_fn = functools.partial(cross_entropy_loss, z_loss_weight=z_loss_weight, return_z_loss=True)
    else:
        loss_fn = cross_entropy_loss
    
    if job_config.training.compile:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """

    @functools.wraps(unwrapped_loss_fn)
    def accumulated_loss_fn(*args, **kwargs):
        result = unwrapped_loss_fn(*args, **kwargs)
        
        # Handle tuple return for z-loss
        if isinstance(result, tuple):
            loss, z_loss = result
            return loss / accumulation_steps, z_loss
        else:
            return result / accumulation_steps

    return accumulated_loss_fn
