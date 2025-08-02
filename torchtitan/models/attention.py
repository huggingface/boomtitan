# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)

from torchtitan.tools.utils import has_cuda_capability
from torchtitan.tools.logging import logger

# FlexAttention mask type. For each mask type, we initialize it at most once per
# batch. To record what it is initialized, FLEX_ATTN_MASK_T is used as the key to
# track the initialized mask.
FLEX_ATTN_MASK_T = tuple[str, int | None]


class FlexAttention(torch.nn.Module):
    """FlexAttention module that uses torch.nn.attention.flex_attention.

    This module is a wrapper around torch.nn.attention.flex_attention. This module
    implements certain common attention types, such as causal and block_causal.

    Args:
        attn_mask_type (str): The type of attention mask. Currently, we support
            "causal" and "block_causal". "causal" means the lower triangle of the
            attention matrix is masked. "block_causal" means the attention matrix
            is divided into blocks, where block boundary is defined by EOS token,
            and the lower triangle of each block is masked.
        fixed_block_size (int | None): The block size to be used to perform attention.
            If specified, each sequence will be further divided to blocks, where each
            block has the maximum size of ``fixed_block_size``. A query will only attend
            to the keys within the same block.
    """

    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation.
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, mode="max-autotune-no-cudagraphs"
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    used_attn_mask_types: ClassVar[set[FLEX_ATTN_MASK_T]] = set()
    # Attention mask type to the created BlockMask.
    # This allows us to keep track the created block masks for each
    # new batch. We will use this to update the block mask when a
    # new batch is created. This also allows user to create different
    # block masks for different layers.
    block_masks: ClassVar[dict[FLEX_ATTN_MASK_T, BlockMask]] = {}

    # Instance variables.
    attn_mask_type: str

    def __init__(
        self, attn_mask_type: str, fixed_block_size: int | None = None
    ) -> None:
        super().__init__()
        if attn_mask_type not in ["causal", "block_causal", "document_causal"]:
            raise ValueError(f"Unrecognized attn_mask_type {attn_mask_type}.")
        self.attn_mask_type = attn_mask_type
        self.fixed_block_size = fixed_block_size

        FlexAttention.used_attn_mask_types.add(self.mask_key)

    @property
    def mask_key(self) -> FLEX_ATTN_MASK_T:
        return (self.attn_mask_type, self.fixed_block_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        block_mask = FlexAttention.block_masks[self.mask_key]
        return FlexAttention.flex_attn(q, k, v, block_mask=block_mask, scale=scale)

    @staticmethod
    def _get_causal_mask_mod() -> _mask_mod_signature:
        def causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return q_idx >= kv_idx

        return causal_mask

    @staticmethod
    def _get_block_causal_mask_mod(
        batch: torch.Tensor, eos_id: int
    ) -> _mask_mod_signature:
        # batch is [b, s, h, d] shape
        mask = batch == eos_id
        mask[:, -1] = True
        acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
        seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
        seq_idx[:, 1:] = acc_mask[:, :-1]

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

        return block_causal_mask

    @staticmethod
    def _get_document_causal_mask_mod(
        batch: torch.Tensor, eos_id: int
    ) -> _mask_mod_signature:
        # batch is [b, s, h, d] shape
        # Document boundaries are marked by EOS tokens
        eos_mask = batch == eos_id
        
        # Mark document boundaries (1 where new doc starts)
        doc_boundaries = torch.zeros_like(batch, dtype=torch.int32)
        doc_boundaries[:, 0] = 1  # First position is always a new doc
        doc_boundaries[:, 1:] = eos_mask[:, :-1].int()
        
        # Cumulative sum gives document IDs
        document_ids = doc_boundaries.cumsum(dim=1)

        def document_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # Only allow attention within the same document AND causal
            return (document_ids[b, q_idx] == document_ids[b, kv_idx]) & (q_idx >= kv_idx)

        return document_causal_mask

    @staticmethod
    def _fixed_block_mask_mod(
        mask_mod: _mask_mod_signature, fixed_block_size: int
    ) -> _mask_mod_signature:
        """
        Given an arbirary mask_mod, divide the input sequence to blocks
        and only allow attention within the same block.

        Args:
            mask_mod: The mask mod to apply to the documents
            fixed_block_size: The number of tokens in each block.
        """

        # Credit to @drisspg.
        def blocked_mask_mod(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # Get the block index of the query and key
            q_block = q_idx // fixed_block_size
            kv_block = kv_idx // fixed_block_size
            # Only allow attention within the same block
            same_block = q_block == kv_block
            # Apply the original mask mod
            inner_mask = mask_mod(
                b, h, q_idx % fixed_block_size, kv_idx % fixed_block_size
            )

            return same_block & inner_mask

        blocked_mask_mod.__name__ = (
            f"blocked_mask_mod_{mask_mod.__name__}_fixed_block_size_{fixed_block_size}"
        )

        return blocked_mask_mod

    
    @staticmethod
    @torch.no_grad()
    def init_attention_mask(batch: torch.Tensor, eos_id: int | None) -> None:
        # batch is [b, s, h, d] shape
        for mask_key in FlexAttention.used_attn_mask_types:
            attn_mask_type, fixed_block_size = mask_key
            match attn_mask_type:
                case "causal":
                    if FlexAttention.block_masks.get(mask_key, None) is not None:
                        continue
                    # We don't care about batch dimension --
                    # all samples have the same lower triangle mask.
                    batch_dimension = 1
                    mask_mod = FlexAttention._get_causal_mask_mod()
                case "block_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for block_causal mask."
                        )
                    batch_dimension = batch.shape[0]
                    mask_mod = FlexAttention._get_block_causal_mask_mod(batch, eos_id)
                case "document_causal":
                    if eos_id is None:
                        raise RuntimeError(
                            "eos_id must be provided for document_causal mask."
                        )
                    batch_dimension = batch.shape[0]
                    mask_mod = FlexAttention._get_document_causal_mask_mod(batch, eos_id)
                case _:
                    raise RuntimeError(f"Shouldn't reach here. {attn_mask_type}")

            if fixed_block_size is not None and fixed_block_size > 0:
                mask_mod = FlexAttention._fixed_block_mask_mod(
                    mask_mod, fixed_block_size
                )

            seq_len = batch.shape[1]
            block_mask = FlexAttention.compiled_create_block_mask(
                mask_mod, batch_dimension, None, seq_len, seq_len
            )
            FlexAttention.block_masks[mask_key] = block_mask


class ScaledDotProductAttention(torch.nn.Module):
    backends: ClassVar[list[SDPBackend]] = []

    def __init__(self, attn_mask_type: str) -> None:
        super().__init__()
        if attn_mask_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )

        ScaledDotProductAttention._init_backend()

    @classmethod
    def _init_backend(cls) -> None:
        if cls.backends:
            return

        # Add CuDNN on B200 w/ highest priority
        cls.backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        if has_cuda_capability(10, 0):
            cls.backends.insert(0, SDPBackend.CUDNN_ATTENTION)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        assert self.backends, "SDPA Backends should not be empty."
        with sdpa_kernel(self.backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def build_attention(
    use_flex_attn: bool, attn_mask_type: str, fixed_block_size: int | None = None
):
    if use_flex_attn:
        return FlexAttention(attn_mask_type, fixed_block_size)
    else:
        if fixed_block_size is not None:
            raise ValueError(
                "TorchTitan with SDPA currently does not support fixed_block_size."
            )
        if attn_mask_type != "causal":
            raise ValueError(
                "TorchTitan with SDPA currently only supports causal mask."
            )
        return ScaledDotProductAttention(attn_mask_type)


def visualize_attention_mask(
    batch: torch.Tensor, 
    eos_id: int, 
    attn_mask_type: str,
    max_tokens: int = 50,
    batch_idx: int = 0
) -> str:
    """
    Visualize attention mask for debugging purposes.
    
    Args:
        batch: Input tensor of shape [batch_size, seq_len]
        eos_id: End of sequence token ID
        attn_mask_type: Type of attention mask ("causal", "block_causal", "document_causal")
        max_tokens: Maximum number of tokens to visualize (for readability)
        batch_idx: Which sample in the batch to visualize
        
    Returns:
        String representation of the attention mask
    """
    import numpy as np
    
    seq_len = min(batch.shape[1], max_tokens)
    sample = batch[batch_idx, :seq_len]
    
    # Create a visualization grid
    mask_grid = np.zeros((seq_len, seq_len), dtype=str)
    
    # Get the appropriate mask function
    if attn_mask_type == "causal":
        # Simple causal mask
        for q_idx in range(seq_len):
            for kv_idx in range(seq_len):
                if q_idx >= kv_idx:
                    mask_grid[q_idx, kv_idx] = "■"
                else:
                    mask_grid[q_idx, kv_idx] = "·"
    
    elif attn_mask_type in ["block_causal", "document_causal"]:
        # Compute document/block boundaries
        eos_mask = sample == eos_id
        doc_boundaries = torch.zeros_like(sample, dtype=torch.int32)
        doc_boundaries[0] = 1
        if seq_len > 1:
            doc_boundaries[1:] = eos_mask[:-1].int()
        document_ids = doc_boundaries.cumsum(dim=0)
        
        for q_idx in range(seq_len):
            for kv_idx in range(seq_len):
                same_doc = document_ids[q_idx] == document_ids[kv_idx]
                causal = q_idx >= kv_idx
                if same_doc and causal:
                    mask_grid[q_idx, kv_idx] = "■"
                else:
                    mask_grid[q_idx, kv_idx] = "·"
    
    # Build the visualization string
    output = []
    output.append(f"\n=== Attention Mask Visualization ({attn_mask_type}) ===")
    output.append(f"Batch index: {batch_idx}, Sequence length: {seq_len}")
    output.append(f"■ = can attend, · = cannot attend")
    
    # Show token IDs and document boundaries
    output.append("\nToken IDs: " + " ".join(f"{t:3d}" for t in sample.tolist()))
    if attn_mask_type in ["block_causal", "document_causal"]:
        output.append("Doc IDs:   " + " ".join(f"{d:3d}" for d in document_ids.tolist()))
        output.append("EOS marks: " + " ".join("EOS" if t == eos_id else "   " for t in sample.tolist()))
    
    # Show the attention mask grid
    output.append("\nAttention mask (queries are rows, keys are columns):")
    output.append("     " + "".join(f"{i:3d}" for i in range(seq_len)))
    for q_idx in range(seq_len):
        row = f"{q_idx:3d}: " + "  ".join(mask_grid[q_idx])
        output.append(row)
    
    # Show document boundaries if applicable
    if attn_mask_type in ["block_causal", "document_causal"]:
        output.append("\nDocument structure:")
        current_doc = 0
        doc_start = 0
        for i in range(seq_len):
            if i > 0 and document_ids[i] != current_doc:
                output.append(f"  Document {current_doc}: tokens {doc_start}-{i-1}")
                current_doc = document_ids[i].item()
                doc_start = i
        output.append(f"  Document {current_doc}: tokens {doc_start}-{seq_len-1}")
    
    return "\n".join(output)


def init_attention_mask(batch: torch.Tensor, eos_id: int | None) -> None:
    FlexAttention.init_attention_mask(batch, eos_id)
