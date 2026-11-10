// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides the concrete implementation `treesap::tensor::RankNAlias` of `treesap::tensor::TensorAliasBase`.
 */

#pragma once

#include "TensorAliasBase.hpp"
#include "util/cexprmath.hpp"

namespace treesap {

namespace tensor {

/**
 * @brief Concrete implementation of the CRTP base class `treesap::tensor::TensorAliasBase` for a general tensor of
 * arbitrary `rank_` without any symmetries.
 * @tparam comp_base_ First index in MultiFab component range belonging to the alias
 * @tparam rank_ Rank of the tensor, i.e. its index count
 * @tparam dim_ Dimension of the tensor indices, i.e. their value range; defaults to 3
 * @tparam ArrayType Type of the tensor components; defaults to `const amrex::Real`
 * @note Satisfies the `concept` `treesap::tensor::TensorAliasImplementation`.
 */
template <int comp_base_, int rank_, int dim_ = 3, typename ArrayType = const amrex::Real>
struct RankNAlias : public TensorAliasBase<RankNAlias<comp_base_, rank_, dim_, ArrayType>, comp_base_,
                                           cexprmath::power(dim_, rank_), rank_, dim_, ArrayType> {
    using Base = TensorAliasBase<RankNAlias<comp_base_, rank_, dim_, ArrayType>, comp_base_,
                                 cexprmath::power(dim_, rank_), rank_, dim_, ArrayType>;
    using Base::dim;
    using Base::ncomp;
    using Base::rank;

    /**
     * @brief Computes the relative component index corresponding to a given set of tensor indices.
     * @param tensor_indices The tensor indices, each taking a value in the range [0, dim_).
     * @return The relative component index in the flattened tensor representation.
     */
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr int get_relative(const IndexArray<rank_> tensor_indices) {
        int result = 0;
#pragma unroll
        for (int i = 0; i < rank_; ++i) {
            result += cexprmath::power(dim_, i) * tensor_indices[i];
        }
        return result;
    }

    /**
     * @brief Restores the tensor indices corresponding to a relative component index.
     * @param rel_comp The relative component index in the range [0, dim_^rank_).
     * @return The corresponding tensor indices, with each index in the range [0, dim_).
     */
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr IndexArray<rank_> restore_relative(int rel_comp) {
        IndexArray<rank_> result{};
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

/// @brief Alias for a rank-0 (scalar) tensor
template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using ScalarAlias = RankNAlias<comp_base_, 0, dim_, ArrayType>;

/// @brief Alias for a rank-1 (vector) tensor
template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using VectorAlias = RankNAlias<comp_base_, 1, dim_, ArrayType>;

/// @brief Alias for a rank-2 tensor
template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using Rank2Alias = RankNAlias<comp_base_, 2, dim_, ArrayType>;

/// @brief Alias for a rank-3 tensor
template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using Rank3Alias = RankNAlias<comp_base_, 3, dim_, ArrayType>;

/// @brief Alias for a rank-4 tensor
template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
using Rank4Alias = RankNAlias<comp_base_, 4, dim_, ArrayType>;

// test a few cases
static_assert(TensorAliasImplementation<ScalarAlias<0>>);
static_assert(TensorAliasImplementation<VectorAlias<0>>);
static_assert(TensorAliasImplementation<Rank2Alias<0>>);
static_assert(TensorAliasImplementation<Rank3Alias<0>>);
static_assert(TensorAliasImplementation<Rank4Alias<0>>);
static_assert(TensorAliasImplementation<RankNAlias<0, 5>>);

} // namespace tensor

} // namespace treesap