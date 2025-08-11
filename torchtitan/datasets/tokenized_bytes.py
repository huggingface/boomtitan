import os
import abc
import sys
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import random
import time

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from fsspec.core import url_to_fs
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader, Dataset

from torchtitan.tools.logging import logger
from torchtitan.distributed import ParallelDims
from torchtitan.datasets.clm_collator import DataCollatorForCLM, DataCollatorForCLMWithPositionIds
from torchtitan.datasets.datatrove_dataset import DatatroveFolderDataset
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchdata.stateful_dataloader import StatefulDataLoader
from contextlib import contextmanager


def human_format(num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    if abs(num) < 1:
        return "{:.3g}".format(num)
    SIZES = ["", "K", "M", "B", "T", "P", "E"]
    num = float("{:.3g}".format(num))
    magnitude = 0
    i = 0
    while abs(num) >= 1000 and i < len(SIZES) - 1:
        magnitude += 1
        num /= 1000.0 if not divide_by_1024 else 1024.0
        i += 1
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[magnitude])

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_dataloader_worker_init(dp_rank: int):
    def dataloader_worker_init(worker_id):
        seed = 2 ** (1 + worker_id) * 3 ** (1 + dp_rank) % (2**32)
        set_random_seed(seed)

    return dataloader_worker_init

def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""

    path = os.path.abspath(os.path.dirname(__file__))
    print("make", "-C", path)
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        sys.exit(1)

@contextmanager
def main_rank_first(group: Optional[dist.ProcessGroup] = None):
    """Context manager that executes the code in the context with the rank zero of the group going first."""
    is_main = dist.get_rank(group) == 0
    if is_main:
        yield

    dist.barrier(group)

    if not is_main:
        yield


class BlendableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets,
        weights: List[float],
        size: int,
        parallel_dims: ParallelDims,
        seed: int,
        consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
        offsets_in_samples: Optional[Dict[str, int]] = None,
    ):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indices.
        start_time = time.time()
        # from https://github.com/NVIDIA/Megatron-LM/commit/c6e65b2e96e8376ccc84225dd1a9b60dd242fc48
        assert num_datasets < 32767
        self.dataset_index = np.zeros(self.size, dtype=np.int16)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)
        self.dataset_num_samples = np.zeros(num_datasets, dtype=np.int64)
        self.random_seed = seed

        # TODO: not sure about this...
        with main_rank_first():
            try:
                from . import helpers
            except ImportError:
            # This block will now be executed by rank 0 first.
            # When other ranks execute it, the 'helpers' module will
            # already be compiled and available.
                try:
                    compile_helper()
                    from . import helpers
                except ImportError:
                    raise ImportError(
                        "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                    )

        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,  # sequential for each dataset_source
            self.dataset_num_samples,
            weights,
            num_datasets,
            self.size,
            torch.distributed.get_rank() == 0,
        )

        if torch.distributed.get_rank() == 0:
            logger.info("> elapsed time for building blendable dataset indices: " "{:.2f} (sec)".format(time.time() - start_time))

        # Initialize consumption tracking
        self.consumed_tokens = {idx: 0 for idx in range(len(datasets))}
        if consumed_tokens_per_dataset_folder is not None:
            # find idx of dataset that matches the folder path
            for idx, dataset in enumerate(datasets):
                for folder_path, consumed_tokens in consumed_tokens_per_dataset_folder.items():
                    if dataset.folder_path == folder_path:
                        self.consumed_tokens[idx] = consumed_tokens
                        if torch.distributed.get_rank() == 0:
                            logger.info(f"[BlendableDataset] Setting consumed_tokens for dataset {idx} ({dataset.folder_path}) to {consumed_tokens}")

        self.sequence_length = None  # Will be set when first batch is processed

        # Setup offsets for already consumed tokens from previous stages
        self.offsets_in_samples = {idx: 0 for idx in range(len(datasets))} # last stage's consumed_tokens_per_dataset_folder
        if offsets_in_samples is not None:
            for idx, dataset in enumerate(datasets):
                for folder_path, offset in offsets_in_samples.items():
                    if dataset.folder_path == folder_path:
                        self.offsets_in_samples[idx] = offset
                        if torch.distributed.get_rank() == 0:
                            logger.info(f"[BlendableDataset] Applying offset {offset} samples to dataset {idx} ({dataset.folder_path})")

        self.sequence_length = None  # Will be set when first batch is processed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]

        return self.datasets[dataset_idx][sample_idx]

    def update_consumption_metrics(self, start_idx: int, end_idx: int, sequence_length: int):
        """Update consumed samples/tokens for the current batch.

        Args:
            start_idx: Starting index of current batch for all dp ranks
            end_idx: Ending index of current batch for all dp ranks
            sequence_length: Sequence length for token calculation
        """
        if self.sequence_length is None:
            self.sequence_length = sequence_length

        # Get dataset indices for current batch
        batch_indices = self.dataset_index[start_idx:end_idx]
        unique_indices, counts = np.unique(batch_indices, return_counts=True)

        # Update consumption dictionaries
        for dataset_idx, count in zip(unique_indices, counts):
            self.consumed_tokens[dataset_idx] += int(count * sequence_length)

    def get_consumption_stats(self):
        """Get current consumption statistics for all datasets.

        Returns:
            dict: Dictionary containing samples and tokens consumed per dataset
        """
        stats = {}
        for dataset_idx, dataset in enumerate(self.datasets):
            stats[dataset.folder_path] = {"tokens": self.consumed_tokens[dataset_idx]}
        return stats


class EmptyInfiniteDataset:
    """
    Hack as removing all columns from a datasets.Dataset makes the number of rows 0.
    This provides a dataset that returns empty dicts but maintains the original length.

    Args:
        length: Number of items in the dataset
    """

    def __init__(self, length: int):
        self._length = length

    def __getitem__(self, item) -> dict:
        if isinstance(item, int):
            return {}
        raise NotImplementedError(f"{item} of type {type(item)} is not supported yet")

    def __len__(self) -> int:
        return self._length


@dataclass
class BaseMegatronSampler:
    total_samples: int
    consumed_samples: int
    micro_batch_size: int
    data_parallel_rank: int
    data_parallel_size: int
    global_batch_size: int
    drop_last: bool = True
    pad_samples_to_global_batch_size: Optional[bool] = False

    def __post_init__(self):
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * self.data_parallel_size

        # Sanity checks.
        if self.total_samples <= 0:
            raise RuntimeError("no sample to consume: {}".format(self.total_samples))
        if self.consumed_samples >= self.total_samples:
            raise RuntimeError("no samples left to consume: {}, {}".format(self.consumed_samples, self.total_samples))
        if self.micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {self.micro_batch_size}")
        if self.data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {self.data_parallel_size}")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise RuntimeError(
                "data_parallel_rank should be smaller than data size, but {} >= {}".format(
                    self.data_parallel_rank, self.data_parallel_size
                )
            )
        if self.global_batch_size % (self.micro_batch_size * self.data_parallel_size) != 0:
            raise RuntimeError(
                f"`global_batch_size` ({self.global_batch_size}) is not divisible by "
                f"`micro_batch_size ({self.micro_batch_size}) x data_parallel_size "
                f"({self.data_parallel_size})`"
            )
        if self.pad_samples_to_global_batch_size and self.global_batch_size is None:
            raise RuntimeError(
                "`pad_samples_to_global_batch_size` can be `True` only when "
                "`global_batch_size` is set to an integer value"
            )

    @abc.abstractmethod
    def __iter__(self):
        ...


@dataclass
class MegatronPretrainingSampler(BaseMegatronSampler):
    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __len__(self):
        num_available_samples: int = self.total_samples - self.consumed_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                return num_available_samples // self.global_batch_size
            else:
                return (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    def __iter__(self):
        batch = []
        batch_idx = 0
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            # print("batch:",batch)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch[start_idx:end_idx]
                batch = []
                batch_idx += 1

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            if self.pad_samples_to_global_batch_size:
                for i in range(
                    self.data_parallel_rank, self.global_batch_size, self.micro_batch_times_data_parallel_size
                ):
                    indices = [batch[j] for j in range(i, max(len(batch), i + self.micro_batch_size))]
                    num_pad = self.micro_batch_size - len(indices)
                    indices = indices + [-1] * num_pad
                    yield indices
            else:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]



class TokenizedBytesFolderDataset(DatatroveFolderDataset):
    def __init__(
        self,
        folder_path: str,
        seq_len: int,
        filename_pattern: str = None,
        recursive: bool = True,
        token_size: int = 2,
        max_tokens: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
        return_positions: bool = False,
        eos_token_id: int | None = None,
        skip_in_stream: bool = True,
        num_samples: Optional[int] = None,
        folder_read_path: Optional[str] = None,
        force_update_cache: bool = os.environ.get("FORCE_UPDATE_CACHE_S3", 0) == "1",
    ):
        super().__init__(
            folder_path=folder_path,
            seq_len=seq_len,
            filename_pattern=filename_pattern,
            recursive=recursive,
            token_size=token_size,
            max_tokens=max_tokens,
            shuffle=shuffle,
            seed=seed,
            return_positions=return_positions,
            eos_token_id=eos_token_id,
            read_path=folder_read_path,
            matched_files=None,
            file_sizes=None,
        )


def get_tb_datasets(
    data_config,
    sequence_length: int,
    global_batch_size: int,
    train_steps: int,
    current_iteration: int,
    parallel_dims: ParallelDims ,
    return_positions: bool = True,
    eos_token_id: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 6,
    consumed_samples: int = 0,
    consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
    last_stages_consumed_tokens_per_dataset_folder: Optional[Dict[str, int]] = None,
):
    """Build TokenizedBytes datasets

    Args:
        sequence_length (int): sequence length
        global_batch_size (int): global batch size
        train_steps (int): number of training steps
    """

    if dist.get_rank() == 0:
        logger.info("Building Streamable datasets.")
    train_num_samples = train_steps * global_batch_size

    dataset_max_tokens = data_config.dataset_max_tokens
    if dataset_max_tokens is None:
        dataset_max_tokens = [None] * len(data_config.dataset_folder)

    # last_stages_consumed_samples_per_dataset_folder = {k: v // sequence_length for k, v in last_stages_consumed_tokens_per_dataset_folder.items()}

    # MC: use_old_brrr_dataloader is not supported anymore
    datasets = [
        TokenizedBytesFolderDataset(
            folder_path=folder_path,
            filename_pattern="*.ds",
            seq_len=sequence_length,
            recursive=False,
            token_size=data_config.token_size_in_bytes,
            max_tokens=max_tokens, # TODO: remove
            shuffle=shuffle,
            return_positions=return_positions,  # if set to True, the position ids are directly read from datatrove
            eos_token_id=eos_token_id,
            seed=seed,
            skip_in_stream=data_config.skip_in_stream,
            num_samples=train_num_samples,
            folder_read_path=data_config.dataset_read_path[i] if data_config.dataset_read_path else None,
        )

        for i, (folder_path, max_tokens) in enumerate(zip(data_config.dataset_folder, dataset_max_tokens))
    ]

    # TODO: not sure this is really useful (MC)
    # # in case of dataset_read_path check we have enough files locally for the training
    # if data_config.dataset_read_path:
    #
    #     weights = data_config.dataset_weights
    #     if not weights:
    #         weights = [1] * len(datasets)
    #
    #     # Normalize weights
    #     weights = np.array(weights, dtype=np.float64)
    #     sum_weights = np.sum(weights)
    #     assert sum_weights > 0.0
    #     weights /= sum_weights
    #
    #     # check we have enough files locally for the training
    #     for i, dataset in enumerate(datasets):
    #         # warmup datasets
    #         estimate_current_sample = int(consumed_samples * weights[i]) + last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)
    #         _ = dataset[estimate_current_sample]
    #         # print which file we're currently reading from
    #         logger.info(f"Dataset {i} ({dataset.folder_path}) is reading from file {dataset.current_file_path}")
    #         # estimate number of tokens needed for this dataset
    #         needed_num_samples_dataset = int((train_steps - current_iteration) * global_batch_size * weights[i])
    #         needed_num_tokens_dataset = needed_num_samples_dataset * sequence_length
    #         needed_size_tokens_dataset = human_format(needed_num_tokens_dataset * data_config.token_size_in_bytes)
    #         logger.info(f"Dataset {i} ({dataset.folder_path}) needs {needed_num_tokens_dataset} tokens (size: {needed_size_tokens_dataset}) for current stage")
    #
    #         # NOTE: let's assume that s3 folder keep the same old files when resuming
    #         # check that sum of lens of files in dataset is greater than needed_num_samples_dataset (use dataset.lens)
    #         total_num_samples_dataset = int(train_steps * global_batch_size * weights[i])
    #         logger.info(f"Dataset {i} ({dataset.folder_path}) on s3 has {len(dataset) * sequence_length} tokens (size: {human_format(len(dataset) * sequence_length * data_config.token_size_in_bytes)}) and needs {total_num_samples_dataset * sequence_length} tokens (size: {human_format(total_num_samples_dataset * sequence_length * data_config.token_size_in_bytes)}) for all stages")
    #         assert total_num_samples_dataset <= len(dataset), f"Not enough files on s3 for dataset {i} ({dataset.folder_path})"
    #         # check that local files exist for the needed_num_samples_dataset
    #         estimate_end_sample = estimate_current_sample + needed_num_samples_dataset
    #         for file_idx, file in enumerate(dataset.files):
    #             # intersection [start_sample, end_sample] with [dataset.lens[file_idx], dataset.lens[file_idx+1]]
    #             a, b, c, d = estimate_current_sample, estimate_end_sample, dataset.lens[file_idx], dataset.lens[file_idx+1]
    #             if max(a, c) < min(b, d): # ranges overlap
    #                 assert os.path.exists(file.file_path), f"Dataset {i} ({dataset.folder_path}) will need file {file.file_path} but it does not exist"
    #                 logger.info(f"Dataset {i} ({dataset.folder_path}) will need file {file.file_path} from sample {max(a, c)} to {min(b, d)} (offset: {last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)})")
    #             else:
    #                 logger.info(f"Dataset {i} ({dataset.folder_path}) will not need file {file.file_path} to train from sample {estimate_current_sample} to {estimate_end_sample} (offset: {last_stages_consumed_samples_per_dataset_folder.get(dataset.folder_path, 0)})")

    if len(datasets) == 1 and False:
    # if len(datasets) == 1:
        outputs_dataset = datasets[0]
    else:
        if dist.get_rank() == 0:
            try:
                compile_helper()
            except ImportError:
                raise ImportError(
                    "Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file."
                )
        dist.barrier()

        weights = data_config.dataset_weights
        if not weights:
            weights = [1] * len(datasets)

        outputs_dataset = BlendableDataset(
            datasets,
            weights,
            size=train_num_samples,
            parallel_dims=parallel_dims,
            seed=seed,
            # TODO: (MC) not sure how to handle this
            # consumed_tokens_per_dataset_folder=consumed_tokens_per_dataset_folder,
            # offsets_in_samples=last_stages_consumed_samples_per_dataset_folder
        )
    return outputs_dataset


def get_tb_dataloader(
    dataset: Union[DatatroveFolderDataset, BlendableDataset, Dataset],
    sequence_length: int,
    micro_batch_size: int,
    global_batch_size: int,
    cfg,
    num_workers: int,
    consumed_samples: int,
    num_samples: int,
    parallel_dims: ParallelDims,
    input_pp_rank: int,
    output_pp_rank: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_position_ids: bool = False,
    use_doc_masking: bool = False,
) -> DataLoader:

    # TODO (MC): is get_rank correct? should be get_local_rank?
    if parallel_dims.pp_enabled:
        pp_rank = parallel_dims.world_mesh["pp"].get_rank()
        pp_size = parallel_dims.world_mesh["pp"].size()
    else:
        pp_rank, pp_size = 0, 1

    if parallel_dims.dp_enabled:
        dp_rank = parallel_dims.world_mesh["dp"].get_rank()
        dp_size = parallel_dims.world_mesh["dp"].size()
    else:
        dp_rank, dp_size = 0, 1

    # Only some rank require to run the dataloader.
    if pp_rank not in [input_pp_rank, output_pp_rank]:
        dataset = EmptyInfiniteDataset(length=len(dataset))

    logger.info(
        f"Building dataloader with consumed samples for current datastage: {consumed_samples}"
    )

    # Megatron sampler
    # batch_sampler = MegatronPretrainingRandomSampler(
    batch_sampler = MegatronPretrainingSampler(
        total_samples=num_samples,
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        drop_last=dataloader_drop_last,
        global_batch_size=global_batch_size,
        pad_samples_to_global_batch_size=cfg.pad_samples_to_global_batch_size,
    )

    # We use the data collator to put the tensors on the right pipeline parallelism rank
    if use_position_ids:
        data_collator = DataCollatorForCLMWithPositionIds(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_dims=parallel_dims,
            use_doc_masking=use_doc_masking,
        )
    else:
        data_collator = DataCollatorForCLM(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_dims=parallel_dims,
        )

    # TODO (MC): should we use ParallelAwareDataloader?
    return StatefulDataLoader(
        dataset,
        # dp_rank=dp_rank,
        # dp_world_size=dp_size,
        # batch_size=micro_batch_size,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
    )
