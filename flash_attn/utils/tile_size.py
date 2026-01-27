"""
Tile size utilities for Flash Attention.

This module provides functions to query the correct tile sizes (kBlockM, kBlockN)
for forward and backward passes based on GPU architecture and configuration.

The tile sizes must match between:
1. create_block_mask / create_q2k_csr_sparse_from_func / create_k2q_csr_sparse_from_func
2. The attention kernel tile sizes

Users should NOT manually set Q_BLOCK_SIZE / KV_BLOCK_SIZE. Instead, use these
helper functions to get the correct values automatically.

IMPORTANT: DSL and C++ backends have DIFFERENT tile sizes!
- C++ backend: Tile sizes come from hopper/tile_size.h (varies by headdim, causal, etc.)
- DSL backend: Tile sizes are fixed (see get_fwd_tile_sizes_dsl / get_bwd_tile_sizes_dsl)

Use the appropriate function based on your backend:
- C++ backend (hopper): get_fwd_tile_sizes() / get_bwd_tile_sizes()
- DSL backend (cute):   get_fwd_tile_sizes_dsl() / get_bwd_tile_sizes_dsl()

IMPLEMENTATION NOTE:
The C++ extension (create_block_mask_cuda) calls hopper/tile_size.h directly,
which is the SINGLE SOURCE OF TRUTH for C++ tile sizes. This ensures tile sizes
stay in sync with actual kernel implementations.

If the C++ extension is not available, a Python fallback is used (which may
become stale if C++ code is modified).
"""

from typing import Tuple
import torch
import warnings

# Try to import C++ extension as primary source of tile sizes
# The C++ extension calls hopper/tile_size.h directly (single source of truth)
_USE_CPP_TILE_SIZES = False
_CPP_IMPORT_WARNING_SHOWN = False

try:
    import create_block_mask_cuda
    _USE_CPP_TILE_SIZES = True
except ImportError:
    pass


def _warn_cpp_not_available():
    """Warn user that C++ extension is not available and fallback is being used."""
    global _CPP_IMPORT_WARNING_SHOWN
    if not _CPP_IMPORT_WARNING_SHOWN:
        warnings.warn(
            "create_block_mask_cuda C++ extension not installed. "
            "Using Python fallback for tile sizes, which may become stale if C++ code changes. "
            "To use the single source of truth (hopper/tile_size.h), install the extension:\n"
            "  cd csrc/utils/create_block_mask && pip install -e .",
            UserWarning,
            stacklevel=3
        )
        _CPP_IMPORT_WARNING_SHOWN = True


def get_arch() -> int:
    """Get GPU architecture as int (80, 86, 89, 90, 100).
    
    Returns:
        Architecture number (e.g., 80 for SM80, 90 for SM90, 100 for SM100)
    """
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _get_fwd_tile_sizes_sm90_fallback(
    headdim: int,
    headdim_v: int = None,
    is_causal: bool = False,
    is_local: bool = False,
    is_arbitrary: bool = False,
    paged_kv_non_TMA: bool = False,
    element_size: int = 2,  # 2 for bf16/fp16, 1 for fp8
    v_colmajor: bool = False,
    softcap: bool = False,
) -> Tuple[int, int]:
    """Get forward pass tile sizes (kBlockM, kBlockN) for SM90.
    
    Mirrors tile_size_fwd_sm90 from hopper/tile_size.h
    
    Args:
        headdim: Head dimension
        headdim_v: Value head dimension (default: same as headdim)
        is_causal: Whether using causal attention
        is_local: Whether using local attention
        is_arbitrary: Whether using arbitrary mask
        paged_kv_non_TMA: Whether using paged KV without TMA
        element_size: Element size in bytes (2 for bf16/fp16)
        v_colmajor: Whether V is column major
        softcap: Whether using softcap
    
    Returns:
        Tuple of (kBlockM, kBlockN)
    """
    if headdim_v is None:
        headdim_v = headdim
        
    if element_size == 2:
        if headdim <= 64:
            if headdim_v == 512:
                return (64, 64)
            elif headdim_v == 256:
                return (128, 96)
            else:
                use_blockN_128 = is_causal or is_local or is_arbitrary or paged_kv_non_TMA
                return (192, 128 if use_blockN_128 else 192)
        elif headdim <= 96:
            return (192, 128 if is_local or is_arbitrary or paged_kv_non_TMA else 144)
        elif headdim <= 128:
            use_blockN_128 = is_causal or is_local or is_arbitrary or paged_kv_non_TMA
            return (128, 128 if use_blockN_128 else 176)
        elif headdim <= 192:
            if paged_kv_non_TMA or is_local or is_arbitrary:
                return (128, 96)
            elif headdim_v <= 128:
                return (128, 128)
            else:
                return (128, 112)
        else:  # headdim <= 256
            return (128, 64 if is_local or is_arbitrary else 80)
    else:  # FP8
        if headdim <= 64:
            return (192, 160)
        elif headdim <= 96:
            return (192, 128)
        elif headdim <= 128:
            if paged_kv_non_TMA:
                return (128, 160)
            elif v_colmajor or (softcap and (is_local or is_arbitrary)):
                return (128, 192)
            else:
                return (128, 224)
        elif headdim <= 192:
            if (paged_kv_non_TMA or softcap) and (is_local or is_arbitrary):
                return (128, 128)
            else:
                return (128, 160)
        else:  # headdim <= 256
            return (128, 64 if is_local or is_arbitrary else 128)


def _get_fwd_tile_sizes_sm8x_fallback(
    arch: int,
    headdim: int,
    headdim_v: int = None,
    is_causal: bool = False,
    is_local: bool = False,
    is_arbitrary: bool = False,
    paged_kv: bool = False,
    varlen_and_split: bool = False,
    softcap: bool = False,
    append_kv: bool = False,
    element_size: int = 2,
) -> Tuple[int, int]:
    """Get forward pass tile sizes (kBlockM, kBlockN) for SM80/SM86/SM89.
    
    Mirrors tile_size_fwd_sm8x from hopper/tile_size.h
    
    Args:
        arch: Architecture (80, 86, or 89)
        headdim: Head dimension
        headdim_v: Value head dimension (default: same as headdim)
        is_causal: Whether using causal attention
        is_local: Whether using local attention
        is_arbitrary: Whether using arbitrary mask
        paged_kv: Whether using paged KV
        varlen_and_split: Whether using variable length and split
        softcap: Whether using softcap
        append_kv: Whether appending KV
        element_size: Element size in bytes (2 for bf16/fp16)
    
    Returns:
        Tuple of (kBlockM, kBlockN)
    """
    if headdim_v is None:
        headdim_v = headdim
        
    sm86_or_89 = arch == 86 or arch == 89
    
    if element_size == 2:
        if headdim <= 64:
            if varlen_and_split:
                return (128, 80)
            elif is_local or is_arbitrary:
                return (128, 96)
            else:
                return (128, 112)
        elif headdim <= 96:
            if varlen_and_split or is_local or is_arbitrary:
                return (128, 48)
            else:
                return (128, 64)
        elif headdim <= 128:
            use_8_warps = sm86_or_89 or varlen_and_split
            if use_8_warps:
                if varlen_and_split:
                    return (128, 96 if is_local or is_arbitrary else 112)
                else:
                    return (128, 96 if is_local or is_arbitrary else 128)
            else:
                return (128, 48 if is_local or is_arbitrary else 64)
        elif headdim <= 192:
            kBlockN_64 = append_kv or is_local or is_arbitrary or varlen_and_split or paged_kv
            return (128, 64 if kBlockN_64 else 96)
        else:  # headdim <= 256
            if sm86_or_89:
                if append_kv:
                    return (128, 32)
                elif varlen_and_split or is_local or is_arbitrary:
                    return (128, 48)
                else:
                    return (128, 64)
            else:
                if append_kv:
                    return (128, 48)
                elif varlen_and_split or is_local or is_arbitrary:
                    return (128, 64)
                else:
                    return (128, 96)
    else:
        # Placeholder for FP8
        return (128, 64)


def get_fwd_tile_sizes(
    arch: int = None,
    headdim: int = 128,
    headdim_v: int = None,
    is_causal: bool = False,
    is_local: bool = False,
    is_arbitrary: bool = False,
    paged_kv: bool = False,
    varlen_and_split: bool = False,
    softcap: bool = False,
    append_kv: bool = False,
    element_size: int = 2,
) -> Tuple[int, int]:
    """Get forward pass tile sizes (kBlockM, kBlockN) based on architecture.
    
    This is the main function users should call. It automatically detects
    the GPU architecture if not specified.
    
    Note: When C++ extension is available, this calls hopper/tile_size.h
    (the SINGLE SOURCE OF TRUTH) via create_block_mask_cuda.
    
    Args:
        arch: GPU architecture (80, 86, 89, 90, 100). Auto-detected if None.
        headdim: Head dimension (default: 128)
        headdim_v: Value head dimension (default: same as headdim)
        is_causal: Whether using causal attention
        is_local: Whether using local attention
        is_arbitrary: Whether using arbitrary mask
        paged_kv: Whether using paged KV
        varlen_and_split: Whether using variable length and split
        softcap: Whether using softcap
        append_kv: Whether appending KV
        element_size: Element size in bytes (2 for bf16/fp16)
    
    Returns:
        Tuple of (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for forward pass block sparsity
    """
    if arch is None:
        arch = get_arch()
    
    # Use C++ extension if available (calls hopper/tile_size.h - single source of truth)
    if _USE_CPP_TILE_SIZES:
        return tuple(create_block_mask_cuda.get_fwd_tile_sizes(
            headdim, is_causal, is_local, is_arbitrary, arch
        ))
    
    # Python fallback (may be stale if C++ code changes)
    _warn_cpp_not_available()
    
    if headdim_v is None:
        headdim_v = headdim
    
    if arch >= 100:
        # SM100 (Blackwell): Use 256x128 as default
        # Note: This may vary based on configuration, update as needed
        return (256, 128)
    elif arch >= 90:
        return _get_fwd_tile_sizes_sm90_fallback(
            headdim=headdim,
            headdim_v=headdim_v,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            paged_kv_non_TMA=paged_kv,
            element_size=element_size,
            softcap=softcap,
        )
    else:
        return _get_fwd_tile_sizes_sm8x_fallback(
            arch=arch,
            headdim=headdim,
            headdim_v=headdim_v,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            paged_kv=paged_kv,
            varlen_and_split=varlen_and_split,
            softcap=softcap,
            append_kv=append_kv,
            element_size=element_size,
        )


def _get_bwd_tile_sizes_sm90_fallback(
    headdim: int,
    is_causal: bool = False,
    is_local: bool = False,
    is_arbitrary: bool = False,
    has_softcap: bool = False,
) -> Tuple[int, int]:
    """Get backward pass tile sizes (kBlockM, kBlockN) for SM90.
    
    Mirrors the dispatch logic in flash_bwd_launch_template.h for SM90.
    
    Returns:
        Tuple of (kBlockM, kBlockN)
    """
    if headdim <= 64:
        if (is_causal and has_softcap) or is_arbitrary:
            return (96, 128)
        else:
            return (128, 128)
    elif headdim <= 96:
        return (64, 128)
    elif headdim <= 128:
        if is_causal or is_local or has_softcap or is_arbitrary:
            return (64, 128)
        else:
            return (80, 128)
    elif headdim <= 192:
        return (64, 96)
    else:  # headdim <= 256
        return (64, 80)


def _get_bwd_tile_sizes_sm8x_fallback(
    arch: int,
    headdim: int,
    is_causal: bool = False,
    is_local: bool = False,
    is_arbitrary: bool = False,
    has_softcap: bool = False,
) -> Tuple[int, int]:
    """Get backward pass tile sizes (kBlockM, kBlockN) for SM80/SM86/SM89.
    
    Mirrors the dispatch logic in flash_bwd_launch_template.h for SM80/86/89.
    
    Returns:
        Tuple of (kBlockM, kBlockN)
    """
    if arch == 86 or arch == 89:
        if headdim <= 64:
            return (64, 128)
        elif headdim <= 96:
            return (64, 128)
        elif headdim <= 128:
            return (64, 96)
        elif headdim <= 192:
            return (64, 64)
        else:  # headdim <= 256
            return (32, 64)
    else:  # SM80
        if headdim <= 64:
            return (128, 128)
        elif headdim <= 96:
            return (64, 128)
        elif headdim <= 128:
            return (64, 128)
        elif headdim <= 192:
            return (64, 80)
        else:  # headdim <= 256
            return (64, 64)


def get_bwd_tile_sizes(
    arch: int = None,
    headdim: int = 128,
    is_causal: bool = False,
    is_local: bool = False,
    is_arbitrary: bool = False,
    has_softcap: bool = False,
) -> Tuple[int, int]:
    """Get backward pass tile sizes (kBlockM, kBlockN) based on architecture.
    
    This is the main function users should call. It automatically detects
    the GPU architecture if not specified.
    
    Note: When C++ extension is available, this calls hopper/tile_size.h
    (the SINGLE SOURCE OF TRUTH) via create_block_mask_cuda.
    
    Args:
        arch: GPU architecture (80, 86, 89, 90, 100). Auto-detected if None.
        headdim: Head dimension (default: 128)
        is_causal: Whether using causal attention
        is_local: Whether using local attention
        is_arbitrary: Whether using arbitrary mask
        has_softcap: Whether using softcap
    
    Returns:
        Tuple of (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for backward pass block sparsity
    """
    if arch is None:
        arch = get_arch()
    
    # Use C++ extension if available (calls hopper/tile_size.h - single source of truth)
    if _USE_CPP_TILE_SIZES:
        return tuple(create_block_mask_cuda.get_bwd_tile_sizes(
            headdim, is_causal, is_local, is_arbitrary, has_softcap, arch
        ))
    
    # Python fallback (may be stale if C++ code changes)
    _warn_cpp_not_available()
    
    if arch >= 100:
        # SM100 (Blackwell): Use 128x128 as default for backward
        return (128, 128)
    elif arch >= 90:
        return _get_bwd_tile_sizes_sm90_fallback(
            headdim=headdim,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            has_softcap=has_softcap,
        )
    else:
        return _get_bwd_tile_sizes_sm8x_fallback(
            arch=arch,
            headdim=headdim,
            is_causal=is_causal,
            is_local=is_local,
            is_arbitrary=is_arbitrary,
            has_softcap=has_softcap,
        )


def validate_tile_sizes(
    q_block_size: int,
    kv_block_size: int,
    expected_q_block_size: int,
    expected_kv_block_size: int,
    pass_type: str = "forward",
) -> None:
    """Validate that provided tile sizes match expected kernel tile sizes.
    
    Args:
        q_block_size: User-provided Q block size
        kv_block_size: User-provided KV block size
        expected_q_block_size: Expected Q block size from kernel
        expected_kv_block_size: Expected KV block size from kernel
        pass_type: Either "forward" or "backward"
    
    Raises:
        ValueError: If tile sizes don't match
    """
    if q_block_size != expected_q_block_size or kv_block_size != expected_kv_block_size:
        raise ValueError(
            f"Block sparsity tile size mismatch for {pass_type} pass!\n"
            f"  Provided: Q_BLOCK_SIZE={q_block_size}, KV_BLOCK_SIZE={kv_block_size}\n"
            f"  Expected: Q_BLOCK_SIZE={expected_q_block_size}, KV_BLOCK_SIZE={expected_kv_block_size}\n"
            f"  Use get_fwd_tile_sizes() or get_bwd_tile_sizes() to get correct values."
        )


# =============================================================================
# DSL Backend Tile Sizes (CUTE DSL / flash_attn.cute.interface)
# =============================================================================
# DSL backend uses FIXED tile sizes defined in flash_attn/cute/interface.py
# These are different from C++ backend tile sizes in hopper/tile_size.h
#
# DSL Forward:  (128, 128) for all configurations with block sparsity
# DSL Backward: (64, 128) for SM90 with arbitrary, (128, 128) for SM100
# =============================================================================

def get_fwd_tile_sizes_dsl(
    arch: int = None,
) -> Tuple[int, int]:
    """Get forward pass tile sizes for DSL backend (flash_attn.cute.interface).
    
    DSL backend uses fixed tile sizes for block sparsity mode:
    - Forward: (128, 128) for all architectures
    
    This is different from C++ backend which varies by headdim.
    
    Args:
        arch: GPU architecture (auto-detected if None). Currently unused but
              kept for API consistency.
    
    Returns:
        Tuple of (Q_BLOCK_SIZE, KV_BLOCK_SIZE) = (128, 128)
    """
    # DSL forward pass always uses (128, 128) for block sparsity mode
    # See flash_attn/cute/interface.py:
    #   m_block_size: int = 128,
    #   n_block_size: int = 128,
    # Note: n_block_size can be 192 when NOT using block_sparsity, but
    # for arbitrary mask (which requires block_sparsity), it's always 128
    return (128, 128)


def get_bwd_tile_sizes_dsl(
    arch: int = None,
    is_causal: bool = False,
    is_arbitrary: bool = False,
) -> Tuple[int, int]:
    """Get backward pass tile sizes for DSL backend (flash_attn.cute.interface).
    
    DSL backend backward tile sizes from flash_attn/cute/interface.py:
    - SM90: m_block_size = 64 if (causal or arbitrary) else 80, n_block_size = 128
    - SM100: m_block_size = 128, n_block_size = 128
    
    Args:
        arch: GPU architecture (auto-detected if None)
        is_causal: Whether using causal attention
        is_arbitrary: Whether using arbitrary mask
    
    Returns:
        Tuple of (Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    """
    if arch is None:
        arch = get_arch()
    
    # From interface.py _flash_attn_bwd:
    # if compute_capability == 9:
    #     m_block_size = 64 if (causal or arbitrary) else 80
    #     n_block_size = 128
    # else:  # SM100
    #     m_block_size = 128
    #     n_block_size = 128
    
    if arch >= 100:
        return (128, 128)
    elif arch >= 90:
        m_block_size = 64 if (is_causal or is_arbitrary) else 80
        return (m_block_size, 128)
    else:
        # DSL doesn't support SM8x, but provide fallback
        return (64, 128)


def get_tile_sizes_by_backend(
    backend: str,
    pass_type: str = "forward",
    arch: int = None,
    headdim: int = 128,
    is_causal: bool = False,
    is_local: bool = False,
    is_arbitrary: bool = False,
    has_softcap: bool = False,
) -> Tuple[int, int]:
    """Get tile sizes based on backend type.
    
    This is a convenience function that selects the appropriate tile size
    function based on the backend.
    
    Args:
        backend: Either "cute" (DSL) or "hopper" (C++)
        pass_type: Either "forward" or "backward"
        arch: GPU architecture (auto-detected if None)
        headdim: Head dimension (only used for C++ backend)
        is_causal: Whether using causal attention
        is_local: Whether using local attention
        is_arbitrary: Whether using arbitrary mask
        has_softcap: Whether using softcap (only for backward)
    
    Returns:
        Tuple of (Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    Example:
        # For DSL backend
        fwd_q, fwd_kv = get_tile_sizes_by_backend("cute", "forward")
        bwd_q, bwd_kv = get_tile_sizes_by_backend("cute", "backward", is_arbitrary=True)
        
        # For C++ backend
        fwd_q, fwd_kv = get_tile_sizes_by_backend("hopper", "forward", headdim=128, is_arbitrary=True)
        bwd_q, bwd_kv = get_tile_sizes_by_backend("hopper", "backward", headdim=128, is_arbitrary=True)
    """
    if arch is None:
        arch = get_arch()
    
    if backend == "cute":
        # DSL backend: fixed tile sizes
        if pass_type == "forward":
            return get_fwd_tile_sizes_dsl(arch=arch)
        else:
            return get_bwd_tile_sizes_dsl(arch=arch, is_causal=is_causal, is_arbitrary=is_arbitrary)
    else:
        # C++ backend (hopper): variable tile sizes from tile_size.h
        if pass_type == "forward":
            return get_fwd_tile_sizes(
                arch=arch, headdim=headdim, is_causal=is_causal,
                is_local=is_local, is_arbitrary=is_arbitrary
            )
        else:
            return get_bwd_tile_sizes(
                arch=arch, headdim=headdim, is_causal=is_causal,
                is_local=is_local, is_arbitrary=is_arbitrary, has_softcap=has_softcap
            )
