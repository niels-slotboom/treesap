#pragma once

#include "TensorAliasBase.hpp"
#include "util/cexprmath.hpp"

namespace treesap {

template <int comp_base_, int rank_, typename ArrayType = const amrex::Real, int dim_ = 3>
struct RankNAlias : public TensorAliasBase<RankNAlias<comp_base_, rank_, ArrayType, dim_>, comp_base_,
                                           cexprmath::power(dim_, rank_), rank_, dim_, ArrayType> {

    template <typename... TensorIndices>
        requires(sizeof...(TensorIndices) == rank_ && (std::is_same_v<int, TensorIndices> && ...))
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int constexpr get_offset(TensorIndices... tensor_indices) const {
        int indices[rank_] = {tensor_indices...};
        int result = 0;
#pragma unroll
        for (int i = 0; i < rank_; ++i) {
            result += cexprmath::power(dim_, i) * indices[i];
        }
        return result;
    }

    AMREX_FORCE_INLINE
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<int, rank_> constexpr restore_relative(int rel_comp) const {
        amrex::GpuArray<int, rank_> result;
        int temp = rel_comp;
#pragma unroll
        for (int i = 0; i < rank_; ++i) {
            result[i] = temp % dim_;
            temp /= dim_;
        }
        return result;
    }
};

// named specialisations for scalars, vectors, and up to rank 4 tensors
template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using ScalarAlias = RankNAlias<comp_base_, 0, ArrayType, dim_>;

template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using VectorAlias = RankNAlias<comp_base_, 1, ArrayType, dim_>;

template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using Rank2Alias = RankNAlias<comp_base_, 2, ArrayType, dim_>;

template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using Rank3Alias = RankNAlias<comp_base_, 3, ArrayType, dim_>;

template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using Rank4Alias = RankNAlias<comp_base_, 4, ArrayType, dim_>;

// test a few cases
static_assert(TensorAliasImplementation<ScalarAlias<0>>);
static_assert(TensorAliasImplementation<VectorAlias<0>>);
static_assert(TensorAliasImplementation<Rank2Alias<0>>);
static_assert(TensorAliasImplementation<Rank3Alias<0>>);
static_assert(TensorAliasImplementation<Rank4Alias<0>>);
static_assert(TensorAliasImplementation<RankNAlias<0, 5>>);

} // namespace treesap