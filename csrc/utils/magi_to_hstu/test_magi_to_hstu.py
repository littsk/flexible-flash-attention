#!/usr/bin/env python3
"""
Test script for magi_to_hstu CUDA kernel.
Ported from the original C++ main() test function.

Test cases from https://sandai-org.github.io/MagiAttention/blog/
"""

import torch
import magi_to_hstu_cuda


def decode_func_out(func_out, q_idx):
    """
    Decode func_out to get valid k intervals for a given q position.
    
    Function encoding rule:
        interval 0: [0, F[0])
        interval 1: [F[1], F[2])
        interval 2: [F[3], F[4])
        ...
    """
    intervals = []
    n_max_func = func_out.shape[0]
    
    f0 = func_out[0, q_idx].item()
    if f0 == -1:
        return intervals
    
    if f0 > 0:
        intervals.append((0, f0))
    
    for i in range(1, n_max_func, 2):
        start = func_out[i, q_idx].item()
        end_idx = i + 1
        if end_idx >= n_max_func:
            break
        end = func_out[end_idx, q_idx].item()
        if start == -1 or end == -1:
            break
        intervals.append((start, end))
    
    return intervals


def reconstruct_mask(func_out, seqlen_q, seqlen_k):
    """Reconstruct attention mask from func_out."""
    mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool)
    
    for q in range(seqlen_q):
        intervals = decode_func_out(func_out, q)
        for start, end in intervals:
            mask[q, start:end] = True
    
    return mask


def print_mask(mask):
    """Print attention mask in a readable format."""
    seqlen_q, seqlen_k = mask.shape
    
    # Header
    print("     ", end="")
    for k in range(seqlen_k):
        print(f"k={k:<2} ", end="")
    print()
    
    # Rows
    for q in range(seqlen_q):
        print(f"q={q:<2} ", end="")
        for k in range(seqlen_k):
            print(f"  {'1' if mask[q, k] else '0'}  ", end="")
        print()


def test_irregular_mask():
    """
    Test case (h) from MagiAttention blog: "some irregular mask"
    
    slice 0: q_range=[0,2), k_range=[0,2), mask_type=0 (full)
    slice 1: q_range=[2,4), k_range=[2,6), mask_type=0 (full)
    slice 2: q_range=[4,6), k_range=[0,2), mask_type=1 (causal)
    slice 3: q_range=[4,6), k_range=[4,6), mask_type=0 (full)
    slice 4: q_range=[6,8), k_range=[2,8), mask_type=1 (causal)
    """
    print("=" * 60)
    print("Test: Irregular Mask (case h)")
    print("=" * 60)
    
    seqlen_q = 8
    seqlen_k = 8
    n_max_func = 5
    
    q_ranges = torch.tensor([
        [0, 2], [2, 4], [4, 6], [4, 6], [6, 8]
    ], dtype=torch.int32, device="cuda")
    
    k_ranges = torch.tensor([
        [0, 2], [2, 6], [0, 2], [4, 6], [2, 8]
    ], dtype=torch.int32, device="cuda")
    
    mask_types = torch.tensor([0, 0, 1, 0, 1], dtype=torch.int32, device="cuda")
    
    # Print configuration
    print("\n=== Attention Slice Configuration ===")
    print(f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, num_slices={len(mask_types)}, n_max_func={n_max_func}\n")
    
    mask_type_names = {0: "full", 1: "causal", 2: "inverse", 3: "bi-causal"}
    for i in range(len(mask_types)):
        print(f"Slice {i}: q_range=[{q_ranges[i, 0].item()},{q_ranges[i, 1].item()}), "
              f"k_range=[{k_ranges[i, 0].item()},{k_ranges[i, 1].item()}), "
              f"mask_type={mask_types[i].item()} ({mask_type_names[mask_types[i].item()]})")
    
    # Run kernel
    func_out = magi_to_hstu_cuda.magi_to_hstu(
        q_ranges, k_ranges, mask_types, seqlen_q, seqlen_k, n_max_func
    )
    n_func = func_out.size(0)
    
    # Print func_out
    print(f"\n=== func_out (shape: {list(func_out.shape)}, n_func: {n_func}) ===")
    print("        ", end="")
    for q in range(seqlen_q):
        print(f"q={q:<3} ", end="")
    print()
    
    for f in range(n_func):
        print(f"F[{f}]:   ", end="")
        for q in range(seqlen_q):
            val = func_out[f, q].item()
            if val == -1:
                print("  -   ", end="")
            else:
                print(f"{val:<5} ", end="")
        print()
    
    # Print intervals for each q
    print("\n=== Valid k intervals for each q ===")
    for q in range(seqlen_q):
        intervals = decode_func_out(func_out, q)
        if not intervals:
            print(f"q={q}: No valid intervals")
        else:
            interval_strs = [f"[{s},{e})" for s, e in intervals]
            print(f"q={q}: {' '.join(interval_strs)}")
    
    # Reconstruct and print mask
    print("\n=== Attention Mask (reconstructed from func_out) ===")
    mask = reconstruct_mask(func_out, seqlen_q, seqlen_k)
    print_mask(mask)
    
    return func_out


def test_varlen_sliding_window_causal():
    """
    Test case (c) from MagiAttention blog: "varlen Sliding-Window Causal Mask"
    """
    print("\n" + "=" * 60)
    print("Test: Varlen Sliding-Window Causal Mask (case c)")
    print("=" * 60)
    
    seqlen_q = 8
    seqlen_k = 8
    n_max_func = 5
    
    q_ranges = torch.tensor([
        [0, 1], [1, 3], [3, 5], [5, 7], [7, 8]
    ], dtype=torch.int32, device="cuda")
    
    k_ranges = torch.tensor([
        [0, 1], [1, 3], [1, 5], [5, 7], [5, 8]
    ], dtype=torch.int32, device="cuda")
    
    mask_types = torch.tensor([1, 1, 3, 1, 3], dtype=torch.int32, device="cuda")
    
    func_out = magi_to_hstu_cuda.magi_to_hstu(
        q_ranges, k_ranges, mask_types, seqlen_q, seqlen_k, n_max_func
    )
    
    print(f"\nfunc_out shape: {list(func_out.shape)}")
    print("\n=== Attention Mask (reconstructed from func_out) ===")
    mask = reconstruct_mask(func_out, seqlen_q, seqlen_k)
    print_mask(mask)
    
    return func_out


def test_varlen_causal():
    """
    Test case (d) from MagiAttention blog: "var len causal mask"
    """
    print("\n" + "=" * 60)
    print("Test: Varlen Causal Mask (case d)")
    print("=" * 60)
    
    seqlen_q = 8
    seqlen_k = 8
    n_max_func = 5
    
    q_ranges = torch.tensor([
        [0, 1], [1, 5], [5, 8]
    ], dtype=torch.int32, device="cuda")
    
    k_ranges = torch.tensor([
        [0, 1], [1, 5], [5, 8]
    ], dtype=torch.int32, device="cuda")
    
    mask_types = torch.tensor([1, 1, 1], dtype=torch.int32, device="cuda")
    
    func_out = magi_to_hstu_cuda.magi_to_hstu(
        q_ranges, k_ranges, mask_types, seqlen_q, seqlen_k, n_max_func
    )
    
    print(f"\nfunc_out shape: {list(func_out.shape)}")
    print("\n=== Attention Mask (reconstructed from func_out) ===")
    mask = reconstruct_mask(func_out, seqlen_q, seqlen_k)
    print_mask(mask)
    
    return func_out


def test_varlen_block_causal():
    """
    Test case (g) from MagiAttention blog: "var len block causal mask"
    """
    print("\n" + "=" * 60)
    print("Test: Varlen Block Causal Mask (case g)")
    print("=" * 60)
    
    seqlen_q = 8
    seqlen_k = 8
    n_max_func = 5
    
    q_ranges = torch.tensor([
        [0, 2], [2, 4], [4, 6], [6, 8]
    ], dtype=torch.int32, device="cuda")
    
    k_ranges = torch.tensor([
        [0, 2], [0, 4], [4, 6], [0, 8]
    ], dtype=torch.int32, device="cuda")
    
    mask_types = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device="cuda")
    
    func_out = magi_to_hstu_cuda.magi_to_hstu(
        q_ranges, k_ranges, mask_types, seqlen_q, seqlen_k, n_max_func
    )
    
    print(f"\nfunc_out shape: {list(func_out.shape)}")
    print("\n=== Attention Mask (reconstructed from func_out) ===")
    mask = reconstruct_mask(func_out, seqlen_q, seqlen_k)
    print_mask(mask)
    
    return func_out


def test_inverse_mask():
    """
    Test case for inverse mask (mask_type=2)
    """
    print("\n" + "=" * 60)
    print("Test: Inverse Mask")
    print("=" * 60)
    
    seqlen_q = 8
    seqlen_k = 8
    n_max_func = 5
    
    q_ranges = torch.tensor([[4, 8]], dtype=torch.int32, device="cuda")
    k_ranges = torch.tensor([[0, 8]], dtype=torch.int32, device="cuda")
    mask_types = torch.tensor([2], dtype=torch.int32, device="cuda")
    
    func_out = magi_to_hstu_cuda.magi_to_hstu(
        q_ranges, k_ranges, mask_types, seqlen_q, seqlen_k, n_max_func
    )
    
    print(f"\nfunc_out shape: {list(func_out.shape)}")
    print("\n=== Attention Mask (reconstructed from func_out) ===")
    mask = reconstruct_mask(func_out, seqlen_q, seqlen_k)
    print_mask(mask)
    
    return func_out


def test_bicausal_mask():
    """
    Test case for bi-causal mask (mask_type=3)
    """
    print("\n" + "=" * 60)
    print("Test: Bi-Causal Mask")
    print("=" * 60)
    
    seqlen_q = 8
    seqlen_k = 8
    n_max_func = 5
    
    q_ranges = torch.tensor([[4, 8]], dtype=torch.int32, device="cuda")
    k_ranges = torch.tensor([[0, 8]], dtype=torch.int32, device="cuda")
    mask_types = torch.tensor([3], dtype=torch.int32, device="cuda")
    
    func_out = magi_to_hstu_cuda.magi_to_hstu(
        q_ranges, k_ranges, mask_types, seqlen_q, seqlen_k, n_max_func
    )
    
    print(f"\nfunc_out shape: {list(func_out.shape)}")
    print("\n=== Attention Mask (reconstructed from func_out) ===")
    mask = reconstruct_mask(func_out, seqlen_q, seqlen_k)
    print_mask(mask)
    
    return func_out


if __name__ == "__main__":
    print("Testing magi_to_hstu CUDA kernel\n")
    
    test_irregular_mask()
    test_varlen_sliding_window_causal()
    test_varlen_causal()
    test_varlen_block_causal()
    test_inverse_mask()
    test_bicausal_mask()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

