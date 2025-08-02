#!/usr/bin/env python3
"""
Test script to visualize document masking patterns.
"""

import sys
import os
# Add the torchtitan directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

import torch
from torchtitan.models.attention import visualize_attention_mask


def test_document_masking():
    """Test and visualize different masking patterns."""
    
    # Example 1: Simple document masking with 3 documents
    print("\n" + "="*80)
    print("Example 1: Three documents separated by EOS tokens")
    print("="*80)
    
    eos_id = 2
    # Create a sample batch with 3 documents
    # Document 1: tokens 10-14, Document 2: tokens 20-23, Document 3: tokens 30-32
    batch = torch.tensor([[
        10, 11, 12, 13, 14, eos_id,  # Document 1
        20, 21, 22, 23, eos_id,       # Document 2  
        30, 31, 32, eos_id,           # Document 3
        40, 41, 42                    # Document 4 (no EOS at end)
    ]])
    
    # Visualize causal mask
    print(visualize_attention_mask(batch, eos_id, "causal", max_tokens=20))
    
    # Visualize document causal mask
    print(visualize_attention_mask(batch, eos_id, "document_causal", max_tokens=20))
    
    # Example 2: Longer documents
    print("\n" + "="*80)
    print("Example 2: Two longer documents")
    print("="*80)
    
    # Create a sample with 2 longer documents
    doc1 = list(range(100, 110))  # 10 tokens
    doc2 = list(range(200, 215))  # 15 tokens
    batch2 = torch.tensor([doc1 + [eos_id] + doc2 + [eos_id]])
    
    print(visualize_attention_mask(batch2, eos_id, "document_causal", max_tokens=30))
    
    # Example 3: Single token documents (edge case)
    print("\n" + "="*80)
    print("Example 3: Single token documents")
    print("="*80)
    
    batch3 = torch.tensor([[1, eos_id, 2, eos_id, 3, eos_id, 4, eos_id]])
    print(visualize_attention_mask(batch3, eos_id, "document_causal", max_tokens=10))
    
    # Example 4: No EOS tokens (single document)
    print("\n" + "="*80)
    print("Example 4: Single document (no EOS tokens)")
    print("="*80)
    
    batch4 = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    print(visualize_attention_mask(batch4, eos_id, "document_causal", max_tokens=10))


if __name__ == "__main__":
    test_document_masking()