# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AdamC: Adam with Corrected Weight Decay

Based on PyTorch's AdamW implementation with modified weight decay:
- For normalized layers: weight decay is scaled by (lr²/lr_max)
- For non-normalized layers: same as AdamW (weight decay scaled by lr)

Reference: https://arxiv.org/pdf/2506.02285
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["AdamC", "adamc"]


class AdamC(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        lr_max: Optional[float] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # For AdamC, lr_max is the peak learning rate (used for normalized layer scaling)
        if lr_max is None:
            lr_max = lr
        elif lr_max < lr:
            raise ValueError(f"lr_max must be >= lr, got lr_max={lr_max}, lr={lr}")

        defaults = dict(
            lr=lr,
            lr_max=lr_max,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

        # We don't support fused for AdamC yet
        if fused:
            raise ValueError("AdamC does not support fused=True. Use foreach=True instead.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("lr_max", group["lr"])
            state_values = list(self.state.values())
            step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
                state_values[0]["step"]
            )
            if not step_is_tensor:
                for s in state_values:
                    s["step"] = torch.tensor(float(s["step"]))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamC does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State Initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            state_steps.append(state["step"])

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adamc(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                lr_max=group["lr_max"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=False,
                param_names=group.get("param_names", None),
            )

        return loss


def _is_normalized_param(param_name: Optional[str]) -> bool:
    """Check if a parameter belongs to a normalized layer based on its name."""
    if param_name is None:
        return False
    
    # Common normalization layer patterns
    norm_patterns = [
        "attention_norm",  # RMSNorm before attention
        "ffn_norm",        # RMSNorm before FFN
        ".norm",           # Final RMSNorm layer
        "qk_norm",         # Query-Key normalization
        "query_norm",      # QKNorm components
        "key_norm",        # QKNorm components
        "layernorm",       # LayerNorm
        "layer_norm",
        "rmsnorm",
        "rms_norm",
    ]
    
    param_name_lower = param_name.lower()
    return any(pattern in param_name_lower for pattern in norm_patterns)


def adamc(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    lr_max: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    param_names: Optional[List[str]] = None,
):
    r"""Functional API that performs AdamC algorithm computation."""

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        # Default to foreach for better performance
        foreach = True

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamc
    else:
        func = _single_tensor_adamc

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        lr_max=lr_max,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
        param_names=param_names,
    )


def _single_tensor_adamc(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    lr_max: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
    param_names: Optional[List[str]] = None,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Check if this is a normalized parameter
        is_normalized = False
        if param_names and i < len(param_names):
            is_normalized = _is_normalized_param(param_names[i])

        # Update step
        step_t += 1

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if amsgrad:
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sqs[i].sqrt().add_(eps)
        else:
            denom = exp_avg_sq.sqrt().add_(eps)

        bias_correction1 = 1 - beta1 ** step_t.item()
        bias_correction2 = 1 - beta2 ** step_t.item()
        step_size = lr / bias_correction1

        bias_correction2_sqrt = bias_correction2**0.5

        # Update parameters (non-in-place for FSDP compatibility)
        param.data.addcdiv_(exp_avg, denom, value=-step_size / bias_correction2_sqrt)

        # AdamC modification: different weight decay for normalized layers
        if weight_decay != 0:
            if is_normalized:
                # For normalized layers: decay = weight_decay * (lr²/lr_max)
                decay_scale = (lr * lr) / lr_max
                param.data.mul_(1 - decay_scale * weight_decay)
            else:
                # For non-normalized layers: standard AdamW decay
                param.data.mul_(1 - lr * weight_decay)


def _multi_tensor_adamc(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    lr_max: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
    param_names: Optional[List[str]] = None,
):
    if len(params) == 0:
        return

    # Determine which parameters are normalized
    is_normalized_list = []
    if param_names:
        is_normalized_list = [_is_normalized_param(name) for name in param_names]
    else:
        is_normalized_list = [False] * len(params)

    # Handle maximize
    if maximize:
        grads = torch._foreach_neg(grads)

    # Update steps
    torch._foreach_add_(state_steps, 1)

    # Decay the first and second moment running average coefficient
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    # Compute denominator
    if amsgrad:
        torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
        sqrt_inputs = max_exp_avg_sqs
    else:
        sqrt_inputs = exp_avg_sqs

    # Calculate sqrt and add eps
    sqrt_result = torch._foreach_sqrt(sqrt_inputs)
    torch._foreach_add_(sqrt_result, eps)

    # Bias correction
    # Note: we can't use foreach for bias correction with different steps
    # So we fall back to a loop for the actual parameter updates
    for i, (param, exp_avg, denom, step) in enumerate(
        zip(params, exp_avgs, sqrt_result, state_steps)
    ):
        bias_correction1 = 1 - beta1 ** step.item()
        bias_correction2 = 1 - beta2 ** step.item()
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5

        # Update parameters (non-in-place for FSDP compatibility)
        param.data.addcdiv_(exp_avg, denom, value=-step_size / bias_correction2_sqrt)

    # Apply weight decay (AdamC modification)
    if weight_decay != 0:
        # Separate params by type
        normalized_params = []
        non_normalized_params = []
        
        for i, (param, is_norm) in enumerate(zip(params, is_normalized_list)):
            if is_norm:
                normalized_params.append(param)
            else:
                non_normalized_params.append(param)
        
        # Apply corrected decay to normalized params
        if normalized_params:
            decay_scale = (lr * lr) / lr_max
            # Use list comprehension for FSDP compatibility
            for param in normalized_params:
                param.data.mul_(1 - decay_scale * weight_decay)
        
        # Apply standard decay to non-normalized params
        if non_normalized_params:
            # Use list comprehension for FSDP compatibility
            for param in non_normalized_params:
                param.data.mul_(1 - lr * weight_decay)


