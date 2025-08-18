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
    z_loss_weight: float = 0.0
) -> torch.Tensor:
    """
    Cross-entropy loss function with optional z-loss regularization.
    
    Z-loss encourages the log normalizer (log Z) of the softmax to stay close to 0,
    which helps stabilize training of large language models.
    
    Args:
        pred: Model predictions of shape [batch_size, seq_len, vocab_size]
        labels: Target labels of shape [batch_size, seq_len]
        z_loss_weight: Weight for the z-loss term (0 to disable, typically 1e-4)
    
    Returns:
        Cross-entropy loss, optionally with z-loss added
    """
    # Flatten predictions and labels for cross-entropy computation
    pred_flat = pred.flatten(0, 1).float()
    labels_flat = labels.flatten(0, 1)
    
    # Compute standard cross-entropy loss
    ce_loss = torch.nn.functional.cross_entropy(pred_flat, labels_flat)
    
    if z_loss_weight > 0:
        # Compute z-loss: z_loss_weight * log(Z)^2 where Z is the partition function
        # log(Z) = log(sum(exp(logits))) = logsumexp(logits)
        log_z = torch.logsumexp(pred_flat, dim=-1)  # Shape: [batch_size * seq_len]
        z_loss = z_loss_weight * torch.square(log_z).mean()
        
        # Store z-loss value for logging (without weight for clearer monitoring)
        total_loss = ce_loss + z_loss
        # Attach z-loss components as attributes for logging
        total_loss.ce_loss = ce_loss.detach()
        total_loss.z_loss = z_loss.detach()
        total_loss.z_loss_unweighted = torch.square(log_z).mean().detach()
        
        return total_loss
    
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
        loss_fn = functools.partial(cross_entropy_loss, z_loss_weight=z_loss_weight)
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
        loss = unwrapped_loss_fn(*args, **kwargs)
        scaled_loss = loss / accumulation_steps
        
        # Preserve z-loss attributes if they exist
        if hasattr(loss, 'z_loss'):
            scaled_loss.z_loss = loss.z_loss / accumulation_steps
            scaled_loss.z_loss_unweighted = loss.z_loss_unweighted
            scaled_loss.ce_loss = loss.ce_loss / accumulation_steps
        
        return scaled_loss

    return accumulated_loss_fn
