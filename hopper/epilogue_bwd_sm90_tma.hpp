/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MNK_, class Element_, class ElementAccum_, int NumEpilogueThreads_, bool Varlen_>
struct CollectiveEpilogueBwd {

    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    static constexpr int NumEpilogueThreads = NumEpilogueThreads_; // 256
    static constexpr bool Varlen = Varlen_;

    using GmemTiledCopydKVTMA = cute::SM90_TMA_REDUCE_ADD;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads); // 16
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomdKVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementAccum,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutdKVTMA = decltype(tile_to_shape(SmemLayoutAtomdKVTMA{}, select<1, 2>(TileShape_MNK{})));

    using SmemCopyAtomdKV = Copy_Atom<cute::DefaultCopy, ElementAccum>;

    struct TensorStorage : cute::aligned_struct<1024> {
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdKVTMA>, 1024> smem_dk;
        cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdKVTMA>, 1024> smem_dv;
    };

    using ShapedKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen_q, d, head, batch)
    using StridedKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using LayoutdKV = cute::Layout<ShapedKV, StridedKV>;

    using TMA_dKV = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapedKV{}, StridedKV{}),
        SmemLayoutdKVTMA{},
        select<1, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for dKV

    // Host side kernel arguments
    struct Arguments {
        ElementAccum* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        ElementAccum* ptr_dV;
        StridedKV const stride_dV;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        int const* seq_ranges = nullptr;
    };

    // Device side kernel params
    struct Params {
        ElementAccum* ptr_dK;
        ShapedKV const shape_dK;
        StridedKV const stride_dK;
        ElementAccum* ptr_dV;
        StridedKV const stride_dV;
        TMA_dKV tma_store_dK, tma_store_dV;
        int const* cu_seqlens = nullptr;
        int const* seqused = nullptr;
        int const* seq_ranges = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        if constexpr (Varlen) {
            assert (args.cu_seqlens != nullptr || args.seq_ranges != nullptr);
        }
        Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
        Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dK, args.stride_dV);
        TMA_dKV tma_store_dK = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdK,
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
        TMA_dKV tma_store_dV = make_tma_copy(
            GmemTiledCopydKVTMA{},
            mdV,
            SmemLayoutdKVTMA{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dKV
        return {args.ptr_dK, args.shape_dK, args.stride_dK, args.ptr_dV, args.stride_dV,
                tma_store_dK, tma_store_dV, args.cu_seqlens, args.seqused, args.seq_ranges};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
    }

    template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& params,
          FrgTensorO const& tdKrdK,
          FrgTensorO const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord
          ) {

        auto [n_block, bidh, bidb] = block_coord;
        Tensor sdK = make_tensor(make_smem_ptr(shared_storage.epilogue.smem_dk.data()), SmemLayoutdKVTMA{});
        Tensor sdV = make_tensor(make_smem_ptr(shared_storage.epilogue.smem_dv.data()), SmemLayoutdKVTMA{});
        auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);

        Tensor tdKrdK_out = flash::convert_type<ElementAccum>(tdKrdK);
        Tensor tdVrdV_out = flash::convert_type<ElementAccum>(tdVrdV);
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V

        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);

        // Copy from shared memory to global memory using TMA
        cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
        cutlass::arch::NamedBarrier::arrive(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

        bool const is_flex = Varlen && params.seq_ranges != nullptr;
        bool const is_varlen = (Varlen && params.cu_seqlens != nullptr) || is_flex;
        // We reshaped dQaccum to have last dimension 32, so the offset needs to be multiplied by kHeadDim / 32
        int offset, seqlen;
        if (is_flex){
            assert(params.cu_seqlens == nullptr);
            offset = params.seq_ranges[2 * bidb];
            seqlen = params.seq_ranges[2 * bidb + 1] - offset;
        }
        else{
            offset = !is_varlen ? 0 : params.cu_seqlens[bidb];
            seqlen = !is_varlen ? get<0>(params.shape_dK) : (
                params.seqused ? params.seqused[bidb] : params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb]
            );
        }

        Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor gdV = local_tile(domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
        auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
        Tensor tdKgdK = block_tma_dK.partition_D(gdK);  // (TMA, TMA_M, TMA_K)
        Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_M, TMA_K)
        Tensor tdVgdV = block_tma_dV.partition_D(gdV);  // (TMA, TMA_M, TMA_K)
        Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_M, TMA_K)


        // Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // // Repeat the partitioning with identity layouts
        // Tensor tdKVcdKV = block_tma_dK.partition_D(cdKV);
        // Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<0>(tdKgdK), size<1>(tdKgdK), size<2>(tdKgdK)),
        //                                     make_stride(          _0{},            _1{},            _0{}));
        // static constexpr int kBlockN = get<1>(TileShape_MNK{});
        // int bound = seqlen - n_block * kBlockN;
        // #pragma unroll
        // for (int n = 0; n < size<1>(tdKVpdKV); ++n) { tdKVpdKV(_0{}, n, _0{}) = get<0>(tdKVcdKV(_0{}, n, _0{})) < bound; }

        int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
        if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
                                            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
            int const lane_predicate = cute::elect_one_sync();
            if (lane_predicate) {
                cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
                cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
                tma_store_arrive();
            }
        }
    }

    CUTLASS_DEVICE void
    store_tail() {
        tma_store_wait<0>();
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord
         ) {
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        auto [n_block, bidh, bidb] = block_coord;
        bool const is_flex = params.seq_ranges != nullptr;
        bool const is_varlen = (Varlen && params.cu_seqlens != nullptr) || is_flex;
        int offset, seqlen;
        if (is_flex){
            offset = params.seq_ranges[2 * bidb];
            seqlen = params.seq_ranges[2 * bidb + 1] - offset;
        }
        else{
            offset = !is_varlen ? 0 : params.cu_seqlens[bidb];
            seqlen = !is_varlen ? get<0>(params.shape_dK) : (params.seqused ? params.seqused[bidb] : params.cu_seqlens[bidb + 1] - offset);
        }

        Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dK, params.stride_dV)(_, _, bidh, !is_varlen ? bidb : 0);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
        Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKVrdKV = make_fragment_like(tdKVgdK);
        clear(tdKVrdKV);
        // Construct identity layout for gdKV
        Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(params.shape_dK); }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdK, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdV, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
    }

};

} // namespace flash
