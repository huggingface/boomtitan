# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import time
from datetime import timedelta
from typing import Any, Generator, Iterable, Optional

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import DataloaderStopIteration
from torchtitan.components.ft import FTManager, maybe_semi_sync_training
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config import ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)

import torch.distributed as dist
from pathlib import Path
from pprint import pformat

import dataclasses
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union
from collections import defaultdict

@dataclasses.dataclass
class DataStageMetadata:
    """
    consumed_train_samples: The number of samples consumed by the model in the this stage (resets at each stage).
    consumed_tokens_per_dataset_folder: The number of tokens consumed by the model in the this stage for each dataset folder. (resets at each stage)
    """

    name: str
    start_training_step: int
    consumed_train_samples: int # We use this for sampler, and it's reset at each stage
    sequence_length: Optional[int] = None # TODO: put back as non-optional
    consumed_tokens_per_dataset_folder: Dict[str, int] = dataclasses.field(default_factory=dict) # this gets reset at each stage

    def __post_init__(self):
        if self.sequence_length is None:
            self.sequence_length = 4096 # TODO: temp

    def sanity_consumed_train_samples(self):
        assert self.consumed_train_samples*self.sequence_length == sum(self.consumed_tokens_per_dataset_folder.values()), \
            f"Mismatch between the total consumed samples and the sum of consumed samples across dataset folders! consumed_train_samples={self.consumed_train_samples}, sequence_length={self.sequence_length}, consumed_tokens_per_dataset_folder={self.consumed_tokens_per_dataset_folder}"

    @property
    def consumed_tokens_all_datasets(self):
        return sum(self.consumed_tokens_per_dataset_folder.values())


@dataclasses.dataclass
class TrainingMetadata:
    """
    consumed_train_samples: The number of samples consumed globally, across all stages.
    last_train_step: The last training step across all stages.
    last_stage_idx: The index of the last stage that was trained.
    data_stages: The metadata for each stage.
    """

    consumed_train_samples: int # TODO: Legacy. This assumed same sequence length across all stages. Not used anymore
    last_train_step: int
    consumed_tokens_total: Optional[int] = None # TODO: put back as non-optional

    # TODO(xrsrke): make this not optional, once we entirely remove
    # the old checkpoint version
    last_stage_idx: Optional[int] = None
    data_stages: Optional[List[DataStageMetadata]] = None

    def __post_init__(self):
        # NOTE: this is a sanity check after loading a trained checkpoint
        assert (
            self.consumed_train_samples == sum(stage.consumed_train_samples for stage in self.data_stages)
        ), "Mismatch between the total consumed samples and the sum of consumed samples across stages! Something went wrong in the training."

        if self.consumed_tokens_total is not None:
            assert self.consumed_tokens_total == sum(stage.consumed_tokens_all_datasets for stage in self.data_stages), "Mismatch between the total consumed tokens and the sum of consumed tokens across stages! Something went wrong in the training."
        else:
            self.consumed_tokens_total = sum(stage.consumed_tokens_all_datasets for stage in self.data_stages)

        # TODO(xrsrke): remove this once we entirely remove non-data-stage training
        if self.last_stage_idx is not None:
            assert self.data_stages is not None, "data_stages should not be None if last_stage_idx is not None"

    @property
    def consumed_tokens_per_dataset_folder_total(self):
        consumed = defaultdict(int)
        for stage in self.data_stages:
            for dataset_folder, tokens in stage.consumed_tokens_per_dataset_folder.items():
                consumed[dataset_folder] += tokens
        return consumed

    @property
    def current_stage(self) -> DataStageMetadata:
        return self.data_stages[self.last_stage_idx]


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    # core configs
    job_config: JobConfig
    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec

    # swappable training components in TrainSpec
    tokenizer: train_spec_module.BaseTokenizer | None
    dataloader: train_spec_module.BaseDataLoader
    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer
    validator: train_spec_module.BaseValidator
    metrics_processor: train_spec_module.MetricsProcessor

    # non-swappable training components
    checkpointer: CheckpointManager
    ft_manager: FTManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: Generator[None, None, None]
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int
    metadata: TrainingMetadata

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.train")

        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            world_size=world_size,
        )

        world_mesh = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.ft_manager = FTManager(job_config.fault_tolerance)
        dp_degree, dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        self.loss_fn = self.train_spec.build_loss_fn(job_config)

        # verify batch sizes
        self.global_batch_size = job_config.training.global_batch_size
        if self.global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation step.
            self.global_batch_size = job_config.training.local_batch_size * dp_degree
        assert self.global_batch_size > 0
        assert (
            self.global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({self.global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {dp_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = self.global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
        assert self.gradient_accumulation_steps > 0
        self.loss_fn = rescale_accumulated_loss(
            self.loss_fn, self.gradient_accumulation_steps
        )

        # build tokenizer and dataloader
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        if not job_config.data_stages:
            logger.info("Creating HF stateful dataloader...")
            # Use HuggingFace standard dataloader with load_dataset
            self.dataloader = self.train_spec.build_dataloader_fn(
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                job_config=job_config,
            )
        else:
            self.dataloader = None

        # build model (using meta init)
        model_args = self.train_spec.model_args[job_config.model.flavor]
        # set the model args from training job configs
        model_args.update_from_config(job_config)

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        with torch.device("meta"):
            model = self.train_spec.model_cls(model_args)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(
            job_config, parallel_dims, model_args
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {self.train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(model, parallel_dims, job_config)

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config.optimizer, parallel_dims, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        # This is only used with datatrove dataloader (multi-stage multi-stage )
        self.metadata = TrainingMetadata(
            consumed_train_samples=0,
            consumed_tokens_total=0,
            last_train_step=0,
            last_stage_idx=0,
            data_stages=[DataStageMetadata(
                name=stage.name,
                start_training_step=stage.start_training_step,
                consumed_train_samples=0,
                sequence_length=stage.sequence_length if stage.sequence_length is not None else job_config.training.seq_len
            ) for stage in self.job_config.data_stages]
        )

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                self.train_spec.state_dict_adapter(model_args)
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=job_config.job.dump_folder,
            ft_manager=self.ft_manager,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

        # Build validator if validation is configured
        if job_config.validation.enabled:
            assert self.train_spec.build_validator_fn is not None

            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = self.train_spec.build_validator_fn(
                job_config=job_config,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.train_spec.build_loss_fn(job_config),
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {self.global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})"
        )

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderStopIteration() from ex

            # Handle different dataloader formats
            if isinstance(batch, dict):
                # Single dict format from datadrove: {input_ids, position_ids, label_ids, label_mask}
                input_dict = {k: v for k, v in batch.items() if k != 'label_ids'}
                labels = batch['label_ids']
            else:
                # Tuple format: (input_dict, labels)
                input_dict, labels = batch

            ntokens_batch = labels.size

            ntokens_batch = labels.numel() if isinstance(labels, torch.Tensor) else labels.size
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            if self.job_config.data_stages:
                # updating token consumption per dataset (basically the dataloder state)
                self._update_datatrove_consumption_metrics(data_iterable)

            # Move tensors to the appropriate device
            for k, v in input_dict.items():
                # DataCollatorForCLMWithPositionIds returns numpy arrays
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
                else:
                    input_dict[k] = torch.from_numpy(v).to(device_type)

            if isinstance(labels, torch.Tensor):
                labels = labels.to(device_type)
            else:
                # Convert NumPy array to PyTorch tensor
                labels = torch.from_numpy(labels).to(device_type)

            yield input_dict, labels

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs = input_dict.get("input", None)
        if inputs is None:
            inputs = input_dict.get("input_ids")

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=parallel_dims.world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, eos_id=self.tokenizer.eos_id)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    def _create_datatrove_dataloader(self) -> DataLoader:
        # NOTE (MC): This is a porting from Nanotron and it assumes that the trainer state has been reloaded.
        # This is why this method needs to be called right after reloading the trainer state in self.train().
        # https://github.com/huggingface/nanotron/blob/bc1f5a612e9546ba19512130b13ec94b5f8609d2/run_train.py#L312-L391

        current_stage = None
        # WARNING: we assume we train on last stage
        stage_idx = len(self.job_config.data_stages) - 1
        stage_args = self.job_config.data_stages[stage_idx]

        if self.step+1 == stage_args.start_training_step:
            print(f"Starting new stage {stage_args.name}")
            # we start a new stage
            if stage_idx >= len(self.metadata.data_stages):
                self.metadata.data_stages.append(DataStageMetadata(
                    name=stage_args.name,
                    start_training_step=stage_args.start_training_step,
                    consumed_train_samples=0,
                    consumed_tokens_per_dataset_folder={},
                    sequence_length=self.job_config.training.seq_len,
                ))
        elif len(self.metadata.data_stages) < len(self.job_config.data_stages):
            raise ValueError(f"If you're trying to start a new stage, you need to set `start_training_step` to the step after the last stage's: {trainer.iteration_step+1}")

        current_stage = self.metadata.data_stages[stage_idx]
        cur_stage_consumed_train_samples = current_stage.consumed_train_samples
        consumed_tokens_per_dataset_folder = current_stage.consumed_tokens_per_dataset_folder
        stage_args_data = self.job_config.data_stages[stage_idx].data

        # warn that if seqlen of stage - 1 has changed, consumed_train_samples=0 so we'll assume we're reading from new folder (so that we can resume training)
        if current_stage.sequence_length != self.metadata.data_stages[-1].sequence_length:
            raise NotImplementedError("We don't support changing sequence length between stages yet")
            # if current_stage.consumed_train_samples == 0:
            #     log_rank(
            #         f"Warning: The sequence length of the last stage has changed from {trainer.metadata.data_stages[-1].sequence_length} to {current_stage.sequence_length}. We'll assume we're reading from the beginning of the dataset folders.",
            #         logger=logger,
            #         level=logging.WARNING,
            #         rank=0,
            #     )
            # else:
            #     # we're resuming training, so that's fine
            #     pass
            # cur_stage_consumed_train_samples = current_stage.consumed_train_samples

        else:
            # Prepare last_stages_consumed_tokens_per_dataset_folder which will
            # be used to offset BlendableDataset to avoid reseeing consumed
            # tokens even when sampler has restarted for this stage
            last_stages_consumed_tokens_per_dataset_folder = {}
            for stage in self.metadata.data_stages[:-1]:
                for folder_path, consumed_tokens in stage.consumed_tokens_per_dataset_folder.items():
                    last_stages_consumed_tokens_per_dataset_folder[folder_path] = last_stages_consumed_tokens_per_dataset_folder.get(folder_path, 0) + consumed_tokens

        # Use datatrove dataloader with TokenizedBytesFolderDataset
        return self.train_spec.build_dataloader_fn(
            global_batch_size=self.global_batch_size,
            micro_batch_size=self.job_config.training.local_batch_size,
            job_config=self.job_config,
            stage_args_data=stage_args_data,
            parallel_dims=self.parallel_dims, # TODO: meaybe this one should be self.ft_manager (Fault Tolerance)
            input_pp_rank=0, # TODO
            output_pp_rank=0,  # TODO
            consumed_train_samples_stage=cur_stage_consumed_train_samples,
            consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
            last_stages_consumed_tokens_per_dataset_folder=last_stages_consumed_tokens_per_dataset_folder,
            tokenizer=self.tokenizer,
            current_iteration=self.step # this is only used for estimating how many needed samples are still available in the dataset
        )

    def _update_datatrove_consumption_metrics(self, data_iterable):
        if hasattr(data_iterable, "dataset") and hasattr(data_iterable.dataset, "update_consumption_metrics"):
            # TODO: only works for BlendableDataset (MC: not a problem for us)
            data_iterable.dataset.update_consumption_metrics(
                start_idx=(self.step - 1) * self.global_batch_size,  # assumes we start from iteration_step=1
                end_idx=self.step * self.global_batch_size,
                sequence_length=self.job_config.training.seq_len,
            )

        # Training Logs
        # Track consumed tokens for all dataset folders in current stage
        if hasattr(data_iterable, "dataset"):
            consumption_stats = data_iterable.dataset.get_consumption_stats()
            current_stage = self.metadata.data_stages[self.metadata.last_stage_idx]

            # Update consumed tokens for all folders in the consumption stats
            for folder_path, stats in consumption_stats.items():
                current_stage.consumed_tokens_per_dataset_folder[folder_path] = stats["tokens"]

        # Original consumption tracking
        self.metadata.consumed_train_samples += self.global_batch_size # TODO: Legacy: idc abt this
        self.metadata.consumed_tokens_total += self.global_batch_size * self.job_config.training.seq_len
        self.metadata.last_train_step = self.step
        self.metadata.current_stage.consumed_train_samples += self.global_batch_size
        assert self.metadata.current_stage.sequence_length == self.job_config.training.seq_len, \
            f"Sequence length mismatch between the current stage {self.metadata.current_stage.sequence_length} and the global sequence length {self.job_config.training.seq_len}"

    def _compute_param_norms(self) -> dict[str, float]:
        """Compute L1 and L2 norms for each layer's parameters."""
        param_norms = {}

        for model in self.model_parts:
            for name, param in model.named_parameters():
                if param.requires_grad and param.data is not None:
                    # Compute L1 and L2 norms
                    l1_norm = param.data.abs().sum().item()
                    l2_norm = param.data.norm(2).item()

                    # Parse the parameter name to create a structured metric name
                    if "tok_embeddings" in name:
                        # Embedding layer
                        metric_base = "param_norms/layer_0/embeddings"
                    elif "output" in name:
                        # Output layer (lm_head)
                        metric_base = "param_norms/output/weight"
                    elif "layers." in name:
                        # Extract layer number and component
                        parts = name.split(".")
                        layer_idx = None
                        component = None

                        # Find layer index
                        for i, part in enumerate(parts):
                            if part == "layers" and i + 1 < len(parts):
                                layer_idx = int(parts[i + 1]) + 1  # Add 1 so layer 0 becomes layer 1
                                remaining = ".".join(parts[i + 2:])

                                # Identify component type
                                if "attention.wq" in remaining:
                                    component = "attn_q"
                                elif "attention.wk" in remaining:
                                    component = "attn_k"
                                elif "attention.wv" in remaining:
                                    component = "attn_v"
                                elif "attention.wo" in remaining:
                                    component = "attn_o"
                                elif "attention_norm" in remaining:
                                    component = "attn_norm"
                                elif "feed_forward.w1" in remaining:
                                    component = "ffn_gate"
                                elif "feed_forward.w2" in remaining:
                                    component = "ffn_down"
                                elif "feed_forward.w3" in remaining:
                                    component = "ffn_up"
                                elif "ffn_norm" in remaining:
                                    component = "ffn_norm"
                                elif "qk_norm.query_norm" in remaining:
                                    component = "qk_norm_q"
                                elif "qk_norm.key_norm" in remaining:
                                    component = "qk_norm_k"
                                break

                        if layer_idx is not None and component is not None:
                            metric_base = f"param_norms/layer_{layer_idx}/{component}"
                        else:
                            # Fallback for unrecognized patterns
                            metric_base = f"param_norms/other/{name.replace('.', '_')}"
                    else:
                        # Handle other parameters (e.g., final norm)
                        if "norm" in name:
                            metric_base = "param_norms/final_norm/weight"
                        else:
                            metric_base = f"param_norms/other/{name.replace('.', '_')}"

                    # Add L1 and L2 norms
                    param_norms[f"{metric_base}/l1"] = l1_norm
                    param_norms[f"{metric_base}/l2"] = l2_norm

        return param_norms

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        accumulated_losses = []
        # If data runs out during gradient accumulation, that
        # entire step will not be executed.
        for microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            loss = self.forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=(
                parallel_dims.world_mesh["pp"] if parallel_dims.pp_enabled else None
            ),
            ep_dense_params_mesh_ndim=(
                parallel_dims.dense_params_mesh_ndim
                if parallel_dims.ep_enabled
                else None
            ),
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            ft_pg = self.ft_manager.loss_sync_pg
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, parallel_dims.world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss, parallel_dims.world_mesh["dp_cp"], ft_pg),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()

        extra_metrics = {
            "consumed_tokens_total": self.metadata.consumed_tokens_total,
            "n_tokens_seen": self.ntokens_seen,
            "lr": lr,
        }
        # consumed tokens per dataset across all stages
        extra_metrics.update(self.metadata.consumed_tokens_per_dataset_folder_total)

        # Compute parameter norms for weight decay monitoring if enabled
        if (self.job_config.metrics.log_param_norms and
            self.step % self.job_config.metrics.log_param_norms_freq == 0):
            param_norms = self._compute_param_norms()
            extra_metrics.update(param_norms)

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}")

        # Create datatrove multi-stage dataloader after reloading trainer state
        # to get token consumption statistics and restart from the correct step/tokens
        if self.job_config.data_stages:
            self.dataloader = self._create_datatrove_dataloader()

        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )
        with (
            maybe_enable_profiling(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
                leaf_folder=leaf_folder,
            ) as memory_profiler,
            maybe_semi_sync_training(
                job_config.fault_tolerance,
                ft_manager=self.ft_manager,
                model_parts=self.model_parts,
                optimizer=self.optimizers,
            ),
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderStopIteration:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                # Run validation if validator is available
                if (
                    self.job_config.validation.enabled
                    and self.validator.should_validate(self.step)
                ):
                    self.validator.validate(self.model_parts, self.step)

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
                )

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.parallel_dims.world_mesh,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "ntokens_seen": self.ntokens_seen,
            "metadata": self.metadata
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]
        self.metadata = state_dict["metadata"]

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
        if self.metrics_processor:
            self.metrics_processor.close()


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    os.makedirs(config.job.dump_folder, exist_ok=True)
    init_logger(Path(config.job.dump_folder) / "train.log")
    trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")
