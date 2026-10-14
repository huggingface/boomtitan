# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from torchtitan.components.ft import FTManager, has_torchft
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger
from torchtitan.components.adam_corrected import AdamC

__all__ = [
    "OptimizersContainer",
    "build_optimizers",
]


if has_torchft:
    import torchft as ft


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        param_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        
        if param_groups is not None:
            # Use provided parameter groups (for weight decay exclusion)
            self.optimizers.append(optimizer_cls(param_groups, **optimizer_kwargs))
            for group in param_groups:
                all_params.extend(group["params"])
        else:
            # Default behavior: one optimizer per model part
            for model in self.model_parts:
                params = [p for p in model.parameters() if p.requires_grad]
                self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
                all_params.extend(params)
                
        self._validate_length(len(self.model_parts) if param_groups is None else 1)
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer, or one optimizer for all param groups."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class FTOptimizersContainer(OptimizersContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager: "ft.Manager",
        use_ft_optimizer: bool = True,
        param_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(model_parts, optimizer_cls, optimizer_kwargs, param_groups)

        # Force to initialize the optimizer state so that `optim.step()`
        # won't be called by state_dict() and load_state_dict().
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = ft.Optimizer(ft_manager, self)
        # Whether to determine quorum using FT.optimizer,
        # in semi-sync training we use the synchronization step to start quorum
        self._use_ft_optimizer: bool = use_ft_optimizer

    def init_cache_state_dict(self) -> None:
        self.cache_state_dict = super().state_dict()

    def state_dict(self) -> dict[str, Any]:
        return self.cache_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # We have to invalidate the `cache_state_dict` because optimizer uses
        # assign instead of copy when doing `load_state_dict()`. Without
        # invalidating the `cache_state_dict`, there will be memory leakage.
        self.cache_state_dict = {}
        super().load_state_dict(state_dict)
        self.init_cache_state_dict()

    def step(self, *args, **kwargs) -> None:
        """Calling the correct step() depending on the caller.

        TorchFT's OptimizerWrapper.step() is designed to be called only once
        per train step per ft.Manager regardless how many optimizers are used.
        Hence we will need to appropriately dispatch the call.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.step(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Calling the correct zero_grad() depending on the caller.

        Check the comment in ``step()``.
        """
        if self._use_ft_optimizer:
            self._use_ft_optimizer = False
            self._ft_optimizer.zero_grad(*args, **kwargs)
            self._use_ft_optimizer = True
        else:
            super().zero_grad(*args, **kwargs)


def get_param_groups(optimizer_config: OptimizerConfig, model: nn.Module, for_adamc: bool = False) -> list[dict[str, Any]]:
    """
    Create parameter groups with different weight decay settings.
    
    Args:
        optimizer_config: Optimizer configuration with weight decay settings
        model: The model to create parameter groups for
        for_adamc: Whether to create groups for AdamC (includes parameter names)
        
    Returns:
        List of parameter groups for the optimizer
    """
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    decay_names = []
    no_decay_names = []
    
    # Track parameter counts for logging
    decay_count = 0
    no_decay_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        should_decay = True
        
        # Check for embedding parameters
        # This includes both input embeddings (tok_embeddings) and output projection layer (output)
        # Following the practice from models like OLMo where tied embeddings share weight decay behavior
        if ("tok_embeddings" in name or "output" in name) and not optimizer_config.wd_embeddings:
            should_decay = False
            
        # Check for RMSNorm parameters (attention_norm, ffn_norm, norm)
        elif any(norm_name in name for norm_name in ["attention_norm", "ffn_norm", ".norm"]) and not optimizer_config.wd_norm:
            should_decay = False
            
        # Check for QKNorm parameters
        elif "qk_norm" in name and not optimizer_config.wd_qknorm:
            should_decay = False
            
        if should_decay:
            decay_params.append(param)
            decay_names.append(name)
            decay_count += 1
        else:
            no_decay_params.append(param)
            no_decay_names.append(name)
            no_decay_count += 1
    
    # Log parameter distribution only on rank 0 (debug level)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.debug(f"Weight decay parameter groups:")
        logger.debug(f"  - With weight decay: {decay_count} parameters")
        logger.debug(f"  - Without weight decay: {no_decay_count} parameters")
        logger.debug(f"  - Settings: wd_embeddings={optimizer_config.wd_embeddings}, wd_norm={optimizer_config.wd_norm}, wd_qknorm={optimizer_config.wd_qknorm}")
        if no_decay_names:
            logger.debug(f"  - No decay parameters: {no_decay_names[:10]}{'...' if len(no_decay_names) > 10 else ''}")
        
        # Validate that exclusions are working as expected
        embeddings_found = any("tok_embeddings" in name or "output" in name for name in no_decay_names)
        norms_found = any(any(norm_name in name for norm_name in ["attention_norm", "ffn_norm", ".norm"]) for name in no_decay_names)
        qknorm_found = any("qk_norm" in name for name in no_decay_names)
        
        if not optimizer_config.wd_embeddings and not embeddings_found:
            logger.warning("wd_embeddings=False but no embedding/output parameters found to exclude from weight decay")
        if not optimizer_config.wd_norm and not norms_found:
            logger.warning("wd_norm=False but no normalization parameters found to exclude from weight decay")
        if not optimizer_config.wd_qknorm and qknorm_found:
            # Only warn if QKNorm is actually present in the model
            logger.warning("wd_qknorm=False but no QKNorm parameters found to exclude from weight decay")
    
    # Create parameter groups
    param_groups = []
    if decay_params:
        group = {
            "params": decay_params,
            "weight_decay": optimizer_config.weight_decay,
        }
        if for_adamc:
            group["param_names"] = decay_names
        param_groups.append(group)
    if no_decay_params:
        group = {
            "params": no_decay_params,
            "weight_decay": 0.0,
        }
        if for_adamc:
            group["param_names"] = no_decay_names
        param_groups.append(group)
    
    return param_groups


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``optimizer_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Optimizer config containing the optimizer name and parameters.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
    """
    optim_in_bwd = optimizer_config.early_step_in_backward
    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )

    name = optimizer_config.name
    lr = optimizer_config.lr
    beta1 = optimizer_config.beta1
    beta2 = optimizer_config.beta2
    eps = optimizer_config.eps
    weight_decay = optimizer_config.weight_decay

    optim_implementation = optimizer_config.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    # Check if we need to use parameter groups for weight decay exclusion
    use_param_groups = (
        not optimizer_config.wd_embeddings or 
        not optimizer_config.wd_norm or 
        not optimizer_config.wd_qknorm
    )
    
    # Check for incompatible parallelism configurations
    if use_param_groups:
        if parallel_dims.pp_enabled:
            raise ValueError(
                "Weight decay exclusion (wd_embeddings=False, wd_norm=False, or wd_qknorm=False) "
                "is not compatible with Pipeline Parallelism. Please either disable weight decay "
                "exclusions or use a different parallelism strategy."
            )
        if parallel_dims.tp_enabled:
            logger.warning(
                "Weight decay exclusion with Tensor Parallelism is experimental and may not work "
                "correctly. Proceed with caution and verify parameter norms using log_param_norms=True."
            )
    
    # Build base optimizer kwargs
    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "fused": fused,
        "foreach": foreach,
    }
    
    # Parameter groups are only supported for single model (non-PP) and not with optim_in_bwd
    param_groups = None
    is_adamc = name == "AdamC"
    
    if use_param_groups and len(model_parts) == 1 and not optim_in_bwd:
        param_groups = get_param_groups(optimizer_config, model_parts[0], for_adamc=is_adamc)
        # Don't add weight_decay to optimizer_kwargs when using param groups
    else:
        # Add weight_decay when not using parameter groups
        optimizer_kwargs["weight_decay"] = weight_decay

    # Add lr_max for AdamC
    if is_adamc:
        # In torchtitan, lr is the peak learning rate (reached after warmup)
        # so we use it as lr_max for AdamC
        optimizer_kwargs["lr_max"] = lr

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "AdamC": AdamC,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd:
        if use_param_groups:
            logger.warning(
                "Weight decay exclusion is not supported with early_step_in_backward. "
                "All parameters will use the same weight decay value."
            )
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    if ft_manager and ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=ft_manager.use_async_quorum,
            param_groups=param_groups,
        )

    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs, param_groups)
