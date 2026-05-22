/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

// ============================================================================
// SINGLE SOURCE OF TRUTH FOR TILE SIZES (SM80, SM86/89, SM90)
// ============================================================================
// This file contains all tile size configurations for both forward and backward
// passes. When modifying tile sizes, ONLY change this file.
//
// For SM100 (Blackwell), tile sizes are fixed:
//   - Forward:  (q_stage * 128, 128), typically q_stage = 1 or 2 -> (128, 128) or (256, 128)
//   - Backward: (128, 128)
//
// Other code that needs tile sizes should include this header and call these
// functions, rather than duplicating the logic.
// ============================================================================


// ============================================================================
// Forward Pass Tile Sizes
// ============================================================================

// Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, int headdim_v, bool is_causal, bool is_local, bool is_arbitary, int element_size=2,
        bool v_colmajor=false, bool paged_kv_non_TMA=false, bool softcap=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            // return {same_hdim ? 192 : 64, same_hdim ? 128 : 64, same_hdim, same_hdim};
            // With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
            // https://github.com/NVIDIA/cutlass/blob/833f6990e031b48b4cd2fcf55e0849c51ef6bac2/include/cute/container/tuple.hpp#L131
            if (headdim_v == 512) {
                return {64, 64, false, false};
            } else if (headdim_v == 256) {
                return {128, 96, true, false};
            } else {
                // Switch to tile size 192 x 192 for now
                bool const use_blockN_128 = is_causal || is_local || is_arbitary || paged_kv_non_TMA;
                return {192, use_blockN_128 ? 128 : 192, use_blockN_128, true};
            }
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local || is_arbitary ? 192 : 176, true, false};
        } else if (headdim <= 96) {
            return {192, is_local || is_arbitary || paged_kv_non_TMA ? 128 : 144, false, true};
        } else if (headdim <= 128) {
            bool const use_blockN_128 = is_causal || is_local || is_arbitary || paged_kv_non_TMA;
            return {128, use_blockN_128 ? 128 : 176, true, true};
            // {128, 192, true, false} and {192, 128, false, true} are quite good too
            // 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if !MmaPV_is_RS
        } else if (headdim <= 192) {
            return {128, paged_kv_non_TMA || is_local || is_arbitary ? 96 : (headdim_v <= 128 ? 128 : 112), true, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, is_local || is_arbitary ? 64 : 80, true, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 160, true, true};
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            return {128, paged_kv_non_TMA ? 160 : (v_colmajor || (softcap && (is_local || is_arbitary)) ? 192 : 224), true, true};
        } else if (headdim <= 192) {
            return {128, (paged_kv_non_TMA || softcap) && (is_local || is_arbitary) ? 128 : 160, true, true};
        } else {
            return {128, is_local || is_arbitary ? 64 : 128, true, !paged_kv_non_TMA};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, bool is_arbitrary, int element_size=2,
        bool paged_kv=false, bool varlen_and_split=false,
        bool softcap=false, bool append_kv=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, varlen_and_split ? 80 : (is_local || is_arbitrary ? 96 : 112), 4, 1, false};
        } else if (headdim <= 96) {
            return {128, varlen_and_split || is_local || is_arbitrary ? 48 : 64, 4, 1, false};
        } else if (headdim <= 128) {
            bool const use_8_warps = sm86_or_89 | varlen_and_split;
            return {128, use_8_warps ? (varlen_and_split ? (is_local || is_arbitrary ? 96 : 112) : (is_local || is_arbitrary ? 96 : 128)) : (is_local || is_arbitrary ? 48 : 64), use_8_warps ? 8 : 4, 1, use_8_warps};
        } else if (headdim <= 192) {
            bool const kBlockN_64 = append_kv || is_local || is_arbitrary || varlen_and_split || paged_kv;
            return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
        } else {
            return {128, sm86_or_89 ? (append_kv ? 32 : (varlen_and_split || is_local || is_arbitrary ? 48 : 64)) : (append_kv ? 48 : (varlen_and_split || is_local || is_arbitrary ? 64 : 96)), 8, 1, sm86_or_89 && !append_kv};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
}


// ============================================================================
// Backward Pass Tile Sizes and Configuration
// ============================================================================

// Return {kBlockM, kBlockN, Stages_dO, Stages_dS, SdP_swapAB, dKV_swapAB, dQ_swapAB,
//         NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs}
// Used by flash_bwd_launch_template.h run_mha_bwd_ function
constexpr std::tuple<int, int, int, int, bool, bool, bool, int, int, int, int, bool>
tile_size_bwd_sm90(int headdim, bool is_causal, bool is_local, bool is_arbitrary, bool has_softcap) {
    if (headdim <= 64) {
        // hdim64 arbitrary has small 128x128 spills, but the larger tile is faster on H100.
        // Keep causal+softcap on 96x128 to avoid its 128x128 spill regression.
        int kBlockM = (is_causal && has_softcap) ? 96 : 128;
        bool dQ_swapAB = (kBlockM < 128);
        return {kBlockM, 128, /*Stages_dO=*/2, /*Stages_dS=*/2, /*SdP_swapAB=*/true, /*dKV_swapAB=*/false, dQ_swapAB,
                /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/1, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/false};
    } else if (headdim <= 96) {
        // hdim96: always 64x128
        return {64, 128, /*Stages_dO=*/2, /*Stages_dS=*/2, /*SdP_swapAB=*/true, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/1, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/1, /*V_in_regs=*/true};
    } else if (headdim <= 128) {
        // hdim128: 64x128 for causal/local/softcap/arbitrary, else 80x128
        int kBlockM = (is_causal || is_local || has_softcap || is_arbitrary) ? 64 : 80;
        bool dQ_swapAB = (kBlockM == 80);
        return {kBlockM, 128, /*Stages_dO=*/2, /*Stages_dS=*/2, /*SdP_swapAB=*/true, /*dKV_swapAB=*/false, dQ_swapAB,
                /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/1, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/1, /*V_in_regs=*/false};
    } else if (headdim <= 192) {
        // hdim192: always 64x96
        return {64, 96, /*Stages_dO=*/1, /*Stages_dS=*/1, /*SdP_swapAB=*/false, /*dKV_swapAB=*/true, /*dQ_swapAB=*/false,
                /*NumMmaWarpGroups=*/3, /*AtomLayoutMSdP=*/1, /*AtomLayoutNdKV=*/1, /*AtomLayoutMdQ=*/1, /*V_in_regs=*/false};
    } else {
        // hdim256: always 64x80
        return {64, 80, /*Stages_dO=*/1, /*Stages_dS=*/1, /*SdP_swapAB=*/false, /*dKV_swapAB=*/true, /*dQ_swapAB=*/true,
                /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/1, /*AtomLayoutNdKV=*/1, /*AtomLayoutMdQ=*/1, /*V_in_regs=*/false};
    }
}

// Return {kBlockM, kBlockN, Stages_dO, Stages_dS, SdP_swapAB, dKV_swapAB, dQ_swapAB,
//         NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs}
// Used by flash_bwd_launch_template.h run_mha_bwd_ function
constexpr std::tuple<int, int, int, int, bool, bool, bool, int, int, int, int, bool>
tile_size_bwd_sm8x(bool sm86_or_89, int headdim, bool is_causal, bool is_local, bool is_arbitrary, bool has_softcap) {
    if (sm86_or_89) {
        // SM86/SM89 configurations - all have V_in_regs=true
        if (headdim <= 64) {
            return {64, 128, /*Stages_dO=*/2, /*Stages_dS=*/2, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/2, /*AtomLayoutNdKV=*/4, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/true};
        } else if (headdim <= 96) {
            return {64, 128, /*Stages_dO=*/1, /*Stages_dS=*/2, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/2, /*AtomLayoutNdKV=*/4, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/true};
        } else if (headdim <= 128) {
            return {64, 96, /*Stages_dO=*/1, /*Stages_dS=*/2, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/2, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/true};
        } else if (headdim <= 192) {
            return {64, 64, /*Stages_dO=*/1, /*Stages_dS=*/1, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/2, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/true};
        } else {
            // hdim256
            return {32, 64, /*Stages_dO=*/1, /*Stages_dS=*/1, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/2, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/1, /*V_in_regs=*/true};
        }
    } else {
        // SM80 configurations - all have V_in_regs=false
        if (headdim <= 64) {
            return {128, 128, /*Stages_dO=*/2, /*Stages_dS=*/2, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/4, /*AtomLayoutNdKV=*/4, /*AtomLayoutMdQ=*/4, /*V_in_regs=*/false};
        } else if (headdim <= 96) {
            return {64, 128, /*Stages_dO=*/2, /*Stages_dS=*/2, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/2, /*AtomLayoutNdKV=*/4, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/false};
        } else if (headdim <= 128) {
            return {64, 128, /*Stages_dO=*/2, /*Stages_dS=*/2, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/2, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/false};
        } else if (headdim <= 192) {
            return {64, 80, /*Stages_dO=*/1, /*Stages_dS=*/2, /*SdP_swapAB=*/false, /*dKV_swapAB=*/true, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/4, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/false};
        } else {
            // hdim256
            return {64, 64, /*Stages_dO=*/1, /*Stages_dS=*/1, /*SdP_swapAB=*/false, /*dKV_swapAB=*/false, /*dQ_swapAB=*/false,
                    /*NumMmaWarpGroups=*/2, /*AtomLayoutMSdP=*/4, /*AtomLayoutNdKV=*/2, /*AtomLayoutMdQ=*/2, /*V_in_regs=*/false};
        }
    }
}


// ============================================================================
// SM100 (Blackwell) Tile Sizes - Fixed values
// ============================================================================

// Return {kBlockM, kBlockN}
// SM100 forward tile size is (q_stage * 128, 128), typically q_stage = 1 or 2
constexpr std::tuple<int, int> tile_size_fwd_sm100(int q_stage = 2) {
    return {q_stage * 128, 128};
}

// Return {kBlockM, kBlockN}
// SM100 backward tile size is fixed at (128, 128)
constexpr std::tuple<int, int> tile_size_bwd_sm100() {
    return {128, 128};
}
