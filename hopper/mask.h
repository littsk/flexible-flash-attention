/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const attention_chunk_divmod;
    cutlass::FastDivmod const qhead_per_khead_divmod;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    };

    // Overload without gMaskFunc for non-arbitrary mask
    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        apply<Seqlenk_mask, Causal_mask, Local_mask, false /*Arbitrary_mask*/, 0 /*kNFunc*/>(tSrS, m_block, n_block, static_cast<void*>(nullptr));
    }

    // Full version with gMaskFunc for arbitrary mask support
    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        bool Arbitrary_mask=false, int kNFunc=0,
        typename Engine, typename Layout, typename MaskFuncTensor>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block,
               MaskFuncTensor const* gMaskFunc) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(!(Arbitrary_mask && (Causal_mask || Local_mask)), "Arbitrary_mask cannot be combined with Causal_mask or Local_mask");
        static_assert(!Arbitrary_mask || kNFunc > 0, "When Arbitrary_mask is true, kNFunc must be > 0");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        if (!Seqlenk_mask && !Causal_mask && !Local_mask && !Arbitrary_mask) { return; }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (!Causal_mask && !Local_mask && !Arbitrary_mask) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else if constexpr (Arbitrary_mask) {  // Arbitrary mask based on mask function intervals
            //   gMaskFunc has shape (kNFunc, kBlockM), where kNFunc is odd
            //   For each row, stores valid column ranges:
            //   col_max[0]   = gMaskFunc(0, row_idx_local) - first valid range upper bound
            //   col_min[i]   = gMaskFunc(2*i+1, row_idx_local) - (i+1)th valid range lower bound
            //   col_max[i+1] = gMaskFunc(2*i+2, row_idx_local) - (i+1)th valid range upper bound
            static constexpr int kNumIntervals = kNFunc / 2;  // Number of additional intervals
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                int const row_idx_local = get<Row>(tScS_rowcol(m, _0{}));
                int const global_row_idx = row_idx_local + m_block * kBlockM;
                // Read col_max[0] from gMaskFunc
                int col_max_0 = (*gMaskFunc)(0, row_idx_local);
                // Read col_min[i] and col_max[i+1] for additional intervals
                int col_min[kNumIntervals > 0 ? kNumIntervals : 1];
                int col_max[kNumIntervals + 1];
                col_max[0] = col_max_0;
                #pragma unroll
                for (int i = 0; i < kNumIntervals; ++i) {
                    col_min[i] = (*gMaskFunc)(2 * i + 1, row_idx_local);
                    col_max[i + 1] = (*gMaskFunc)(2 * i + 2, row_idx_local);
                }
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const col_idx_local = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                    int const global_col_idx = col_idx_local + n_block * kBlockN + thread_col_offset;
                    // Check if global_col_idx is in valid interval [0, col_max[0]) or any [col_min[i], col_max[i+1])
                    bool value_valid = global_col_idx < col_max[0];
                    #pragma unroll
                    for (int i = 0; i < kNumIntervals; ++i) {
                        if (global_col_idx >= col_min[i] && global_col_idx < col_max[i + 1]) {
                            value_valid = true;
                        }
                    }
                    if constexpr (Seqlenk_mask) {
                        bool const out_of_bounds = (global_row_idx >= seqlen_q) || (global_col_idx >= seqlen_k);
                        if (out_of_bounds || !value_valid) { tSrS_rowcol(m, n) = -INFINITY; }
                    } else {
                        if (!value_valid) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // Causal or Local mask based on both row and col
            if constexpr (!SwapAB) {
                // If PackGQA, we split the work of compute divmod among threads in the same row
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                // Might get OOB but it's ok since we'll check it later
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const local_row_offset_right = causal_row_offset + window_size_right;
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;  // TODO: subtract thread_col_offset?
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left = row_idx + local_row_offset_left;
                        if (attention_chunk_divmod.divisor > 0) {
                            int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
                            col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
                            col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
                        }
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            } else {
                // TODO: backward does not support attention_chunk yet
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }
    };

};

} // namespace flash
