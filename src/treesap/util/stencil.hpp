// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>

#include "stencil_impl.hpp"

namespace treesap {

/**
 * @namespace treesap::stencil
 * @brief Defines compile-time resolved derivative stencils and related operators (up to third
 * derivatives, arbitrary stencil width).
 * @note It is assumed that \f$\Delta x = \Delta y = \Delta z\f$.
 * @note Multiplication of results by \f$1/\Delta x^k\f$ with \f$k\f$ the derivative order is **UP TO
 * THE USER**.
 */
namespace stencil {
/// Computes arbitrary order derivative given template directions using grid indices (i, j, k).
template <int ngrow, typename ArrayType, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType derivative(int i, int j, int k, int comp,
                                                              const amrex::Array4<ArrayType>& arr) {
    static_assert(((Dirs >= 0 && Dirs <= 2) && ...),
                  "All direction template parameters must be between 0 and 2 (inclusive)");
    return stencil_impl::derivative_impl<ngrow, ArrayType, Dirs...>::eval(i, j, k, comp, arr);
}

/// Computes arbitrary order derivative given template directions using an IntVect index.
template <int ngrow, typename ArrayType, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType derivative(amrex::IntVect idx, int comp,
                                                              const amrex::Array4<ArrayType>& arr) {
    static_assert(((Dirs >= 0 && Dirs <= 2) && ...),
                  "All direction template parameters must be between 0 and 2 (inclusive)");
    return stencil_impl::derivative_impl<ngrow, ArrayType, Dirs...>::eval(idx[0], idx[1], idx[2], comp, arr);
}

// --- --- --- SPECIAL OPERATORS DERIVED FROM BASIC DERIVATIVE --- --- ---

// --- GRADIENT ---
/// Computes the gradient vector at a grid point.
template <int ngrow, typename ArrayType>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::GpuArray<ArrayType, 3> gradient(int i, int j, int k, int comp,
                                                                                const amrex::Array4<ArrayType>& arr) {
    return {
        derivative<ngrow, ArrayType, dir::x>(i, j, k, comp, arr),
        derivative<ngrow, ArrayType, dir::y>(i, j, k, comp, arr),
        derivative<ngrow, ArrayType, dir::z>(i, j, k, comp, arr),
    };
}

/// Computes the squared magnitude of the gradient at a grid point.
template <int ngrow, typename ArrayType>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType gradient_squared(int i, int j, int k, int comp,
                                                                    const amrex::Array4<ArrayType>& arr) {
    const auto grad = gradient<ngrow, ArrayType>(i, j, k, comp, arr);
    return grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2];
}

// --- LAPLACIAN ---
// NOTE: This only works for dx = dy = dz.
/// Computes the Laplacian operator at a grid point.
template <int ngrow, typename ArrayType>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType laplacian(int i, int j, int k, int comp,
                                                             const amrex::Array4<ArrayType>& arr) {
    return derivative<ngrow, ArrayType, dir::x, dir::x>(i, j, k, comp, arr) +
           derivative<ngrow, ArrayType, dir::y, dir::y>(i, j, k, comp, arr) +
           derivative<ngrow, ArrayType, dir::z, dir::z>(i, j, k, comp, arr);
}

// --- HESSIAN ---
/// Computes the Hessian matrix of second-order partial derivatives at a grid point.
template <int ngrow, typename ArrayType>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Array2D<ArrayType, 0, 2, 0, 2>
hessian(int i, int j, int k, int comp, const amrex::Array4<ArrayType>& arr) {
    amrex::Array2D<ArrayType, 0, 2, 0, 2> H;
    // diagonal
    H(0, 0) = derivative<ngrow, ArrayType, dir::x, dir::x>(i, j, k, comp, arr);
    H(1, 1) = derivative<ngrow, ArrayType, dir::y, dir::y>(i, j, k, comp, arr);
    H(2, 2) = derivative<ngrow, ArrayType, dir::z, dir::z>(i, j, k, comp, arr);
    // off-diagonal
    H(0, 1) = H(1, 0) = derivative<ngrow, ArrayType, dir::x, dir::y>(i, j, k, comp, arr);
    H(0, 2) = H(2, 0) = derivative<ngrow, ArrayType, dir::x, dir::z>(i, j, k, comp, arr);
    H(1, 2) = H(2, 1) = derivative<ngrow, ArrayType, dir::y, dir::z>(i, j, k, comp, arr);

    return H;
}

// --- --- --- KREISS-OLIGER DISSIPATION TERMS --- --- ---
/// Computes the Kreiss-Oliger dissipation term along a given coordinate direction.
template <int ngrow, int dir, typename ArrayType>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType kreiss_oliger(int i, int j, int k, int comp,
                                                                 const amrex::Array4<ArrayType>& arr) {
    static_assert(ngrow > 0, "Can only apply Kreiss-Oliger stencil for ngrow >= 1");
    constexpr int i_offset = (dir == stencil::dir::x) ? 1 : 0;
    constexpr int j_offset = (dir == stencil::dir::y) ? 1 : 0;
    constexpr int k_offset = (dir == stencil::dir::z) ? 1 : 0;

    constexpr int r = ngrow;
    constexpr double norm = 1.0 / static_cast<double>(cexprmath::power(2, 2 * ngrow));

    std::decay_t<ArrayType> result = -cexprmath::binom(2 * r, r) * arr(i, j, k, comp);
#pragma unroll
    for (int s = 1; s <= ngrow; ++s) {
        result += -cexprmath::sign(s) * cexprmath::binom(2 * r, r + s) *
                  (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) +
                   arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
    }
    return result * norm;
}

} // namespace stencil

} // namespace treesap