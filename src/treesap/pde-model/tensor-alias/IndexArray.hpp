// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once

#include "AMReX.H"

/**
 * @brief Small helper struct to contain tensor index arrays.
 * @note This is necessary because amrex::GpuArray does not allow constexpr operator[], and std::array is not
 * device-safe.
 */
template <int rank_> struct IndexArray {
    static_assert(rank_ >= 0);
    int data[rank_ > 0 ? rank_ : 1]{};

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr int& operator[](int i) { return data[i]; }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr const int& operator[](int i) const { return data[i]; }
};
