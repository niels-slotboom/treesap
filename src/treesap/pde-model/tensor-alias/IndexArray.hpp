// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Implements the `treesap::tensor::IndexArray` helper class template.
 */

#pragma once

#include "AMReX.H"

namespace treesap {

namespace tensor {
/**
 * @brief Small helper struct template to contain tensor index arrays.
 * @note This struct template is needed because `amrex::GpuArray` does not allow `constexpr` usage of `operator[]`, and
 * `std::array` is not device-safe.
 */
template <int rank_> struct IndexArray {
    static_assert(rank_ >= 0);
    int data[rank_ > 0 ? rank_ : 1]{}; ///< Holds index data.

    /// `constexpr`-accessor
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr int& operator[](int i) { return data[i]; }

    /// `constexpr`-accessor
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr const int& operator[](int i) const { return data[i]; }
};
} // namespace tensor

} // namespace treesap