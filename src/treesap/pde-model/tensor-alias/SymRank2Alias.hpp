// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides the concrete implementation `treesap::tensor::SymRank2Alias` of `treesap::tensor::TensorAliasBase`.
 */

#pragma once

#include "TensorAliasBase.hpp"

namespace treesap {

namespace tensor {
/**
 * @brief Concrete implementation of the CRTP base class template `treesap::tensor::TensorAliasBase` for a symmetric
 * rank-2 tensor, such as e.g. the metric or extrinsic curvature. Redundant components are not aliased; only the
 * lower-triangular part is addressed.
 * @tparam comp_base_ First index in MultiFab component range belonging to the alias
 * @tparam dim_ Dimension of the tensor indices, i.e. their value range; defaults to 3
 * @tparam ArrayType Type of the tensor components; defaults to `const amrex::Real`
 * @note Satisfies the `concept` `treesap::tensor::TensorAliasImplementation`.
 */
template <int comp_base_, int dim_ = 3, typename ArrayType = const amrex::Real>
struct SymRank2Alias : public TensorAliasBase<SymRank2Alias<comp_base_, dim_, ArrayType>, comp_base_,
                                              (dim_ * (dim_ + 1)) / 2, 2, dim_, ArrayType> {
    using Base = TensorAliasBase<SymRank2Alias<comp_base_, dim_, ArrayType>, comp_base_, (dim_ * (dim_ + 1)) / 2, 2,
                                 dim_, ArrayType>;
    using Base::dim;
    using Base::ncomp;
    using Base::rank;

    /**
     * @brief Computes the relative component index corresponding to a given set of tensor indices, respecting the
     * symmetry; {a, b} and {b, a} lead to the same return value.
     * @param tensor_indices The tensor indices, each taking a value in the range [0, dim_).
     * @return The relative component index in the flattened lower-triangular representation.
     */
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr int get_relative(const IndexArray<rank> tensor_indices) {
        const int a = tensor_indices[0];
        const int b = tensor_indices[1];
        const int row = (a > b) ? a : b;
        const int col = (a > b) ? b : a;
        return (row * (row + 1)) / 2 + col;
    }

    /**
     * @brief Restores the tensor indices corresponding to a relative component index, always returning the lower
     * triangular index pair {a, b}, with a >= b.
     * @param rel_comp The relative component index in the range [0, (dim_ * (dim_ + 1)) / 2).
     * @return The corresponding tensor indices, with each index in the range [0, dim_).
     */
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr IndexArray<rank> restore_relative(int rel_comp) {
        int row = 0;
        while ((row + 1) * (row + 2) / 2 <= rel_comp) {
            ++row;
        }
        const int col = rel_comp - (row * (row + 1)) / 2;
        return {row, col};
    }
};
} // namespace tensor
} // namespace treesap