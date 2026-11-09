// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once
#include "AMReX_REAL.H"
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>

#include <cstdint>
#include <utility>

#include "cexprmath.hpp"

namespace treesap {

/**
 * Defines compile-time resolved derivative stencils (up to third derivatives, arbitrary stencil width).
 * Note: It is assumed that dx = dy = dz, and multiplication of results by 1/dx^k with k the derivative order is UP TO
 * THE USER.
 */
namespace stencil {
// Directions (x=0, y=1, z=2)
enum dir : int { x = 0, y = 1, z = 2 };

// weights for pure/unmixed derivatives
namespace weights {
// weights for first derivative stencil with width 2*ngrow + 1, at offset away from center point
template <int ngrow> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr amrex::Real first(int offset) {
    static_assert(ngrow > 0, "ngrow needs to be at least 1 to evaluate first derivatives");

    using namespace cexprmath;
    int n = ngrow;
    int k = offset;
    int k_abs = (k > 0) ? k : -k;

    if (offset == 0)
        return 0;

    std::int64_t num = sign(k_abs + 1) * power(factorial(n), 2);
    std::int64_t denom = k_abs * factorial(n - k_abs) * factorial(n + k_abs);
    double w = static_cast<double>(num) / static_cast<double>(denom);
    return static_cast<amrex::Real>((k > 0) ? w : -w);
}

// weights for second derivative stencil with width 2*ngrow + 1, at offset away from center point
template <int ngrow> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr amrex::Real second(int offset) {
    static_assert(ngrow > 0, "ngrow needs to be at least 1 to evaluate second derivatives");

    using namespace cexprmath;
    int n = ngrow;
    int k = (offset > 0) ? offset : -offset;

    if (offset == 0)
        return -2.0 * generalised_harmonic_number(n, 2);

    std::int64_t num = 2 * sign(k + 1) * power(factorial(n), 2);
    std::int64_t denom = k * k * factorial(n - k) * factorial(n + k);
    return static_cast<amrex::Real>(static_cast<double>(num) / static_cast<double>(denom));
}

// weights for third derivative stencil with width 2*ngrow + 1, at offset away from center point
template <int ngrow> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr amrex::Real third(int offset) {
    static_assert(ngrow > 1, "ngrow needs to be at least 2 to evaluate third derivatives");

    using namespace cexprmath;
    int n = ngrow;
    int k = offset;
    int k_abs = (k > 0) ? k : -k;

    if (offset == 0)
        return 0;

    std::int64_t num = 6 * sign(k_abs + 1) * power(factorial(n), 2);
    std::int64_t denom = power(k_abs, 3) * factorial(n - k_abs) * factorial(n + k_abs);
    double w = ((1.0 - static_cast<double>(k_abs * k_abs) * generalised_harmonic_number(n, 2)) *
                static_cast<double>(num) / static_cast<double>(denom));
    return static_cast<amrex::Real>((k > 0) ? w : -w);
}
} // namespace weights

// --- --- --- FUNDAMENTAL DERIVATIVE STENCIL TEMPLATES --- --- ---
/// Helper variable template to trigger static assertions for invalid derivative configurations.
template <int ngrow> constexpr bool always_false = false;

/// Forward declaration for derivative calculation.
template <int ngrow, typename ArrayType, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType derivative(int i, int j, int k, int comp,
                                                              const amrex::Array4<ArrayType>& arr);

// base template for derivative_impl struct
template <int ngrow, typename ArrayType, int... Dirs> struct derivative_impl {
    static_assert(always_false<ngrow>,
                  "No stencil::derivative implementation available for requested template parameters");
    /// Base evaluation fallback function.
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType eval(int i, int j, int k, int comp,
                                                                   const amrex::Array4<ArrayType>& arr) {
        return 0;
    }
};

// --- FIRST DERIVATIVES ---
template <int ngrow, typename ArrayType, int dir> struct derivative_impl<ngrow, ArrayType, dir> {
    /// Evaluates first derivative along a specified direction.
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType eval(int i, int j, int k, int comp,
                                                                   const amrex::Array4<ArrayType>& arr) {
        constexpr int i_offset = (dir == stencil::dir::x);
        constexpr int j_offset = (dir == stencil::dir::y);
        constexpr int k_offset = (dir == stencil::dir::z);

        std::decay_t<ArrayType> result = 0.0;
#pragma unroll
        for (int s = 1; s <= ngrow; ++s) {
            result += weights::first<ngrow>(s) * (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) -
                                                  arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
        }
        return result;
    }
};

// --- SECOND DERIVATIVES ---
template <int ngrow, typename ArrayType, int dir0, int dir1> struct derivative_impl<ngrow, ArrayType, dir0, dir1> {
    /// Evaluates second derivative (pure or mixed) along specified directions.
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType eval(int i, int j, int k, int comp,
                                                                   const amrex::Array4<ArrayType>& arr) {
        if constexpr (dir0 == dir1) { // unmixed derivative
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            std::decay_t<ArrayType> result = weights::second<ngrow>(0) * arr(i, j, k, comp);
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::second<ngrow>(s) * (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) +
                                                       arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
            }
            return result;
        } else {
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            std::decay_t<ArrayType> result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result +=
                    weights::first<ngrow>(s) * (derivative<ngrow, ArrayType, dir1>(i + s * i_offset, j + s * j_offset,
                                                                                   k + s * k_offset, comp, arr) -
                                                derivative<ngrow, ArrayType, dir1>(i - s * i_offset, j - s * j_offset,
                                                                                   k - s * k_offset, comp, arr));
            }
            return result;
        }
    }
};

// --- THIRD DERIVATIVES ---
template <int ngrow, typename ArrayType, int dir0, int dir1, int dir2>
struct derivative_impl<ngrow, ArrayType, dir0, dir1, dir2> {
    /// Evaluates third derivative (pure, mixed double-single, or mixed triple) along specified directions.
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType eval(int i, int j, int k, int comp,
                                                                   const amrex::Array4<ArrayType>& arr) {
        if constexpr (dir0 == dir1 && dir1 == dir2) { // unmixed derivative
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            std::decay_t<ArrayType> result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::third<ngrow>(s) * (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) -
                                                      arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
            }
            return result;
        } else if constexpr (dir0 == dir1 || dir0 == dir2 ||
                             dir1 == dir2) { // derivative like xxy, two in the same direction
            constexpr std::pair<int, int> udirs =
                (dir0 == dir1) ? std::pair<int, int>{dir2, dir0} : std::pair<int, int>{dir0, dir1};
            // udirs.first is the single derivative, udirs.second the double derivative
            constexpr int i_offset = (udirs.first == stencil::dir::x);
            constexpr int j_offset = (udirs.first == stencil::dir::y);
            constexpr int k_offset = (udirs.first == stencil::dir::z);
            // hence take a first derivative along udirs.first of derivative<ngrow,udirs.second,udirs.second>(...)
            std::decay_t<ArrayType> result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::first<ngrow>(s) *
                          (derivative<ngrow, ArrayType, udirs.second, udirs.second>(i + s * i_offset, j + s * j_offset,
                                                                                    k + s * k_offset, comp, arr) -
                           derivative<ngrow, ArrayType, udirs.second, udirs.second>(i - s * i_offset, j - s * j_offset,
                                                                                    k - s * k_offset, comp, arr));
            }
            return result;
        } else {
            constexpr int i_offset = (dir0 == stencil::dir::x);
            constexpr int j_offset = (dir0 == stencil::dir::y);
            constexpr int k_offset = (dir0 == stencil::dir::z);

            std::decay_t<ArrayType> result = 0.0;
#pragma unroll
            for (int s = 1; s <= ngrow; ++s) {
                result += weights::first<ngrow>(s) *
                          (derivative<ngrow, ArrayType, dir1, dir2>(i + s * i_offset, j + s * j_offset,
                                                                    k + s * k_offset, comp, arr) -
                           derivative<ngrow, ArrayType, dir1, dir2>(i - s * i_offset, j - s * j_offset,
                                                                    k - s * k_offset, comp, arr));
            }
            return result;
        }
    }
};

/// Computes arbitrary order derivative given template directions using grid indices (i, j, k).
template <int ngrow, typename ArrayType, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType derivative(int i, int j, int k, int comp,
                                                              const amrex::Array4<ArrayType>& arr) {
    static_assert(((Dirs >= 0 && Dirs <= 2) && ...),
                  "All direction template parameters must be between 0 and 2 (inclusive)");
    return derivative_impl<ngrow, ArrayType, Dirs...>::eval(i, j, k, comp, arr);
}

/// Computes arbitrary order derivative given template directions using an IntVect index.
template <int ngrow, typename ArrayType, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType derivative(amrex::IntVect idx, int comp,
                                                              const amrex::Array4<ArrayType>& arr) {
    static_assert(((Dirs >= 0 && Dirs <= 2) && ...),
                  "All direction template parameters must be between 0 and 2 (inclusive)");
    return derivative_impl<ngrow, ArrayType, Dirs...>::eval(idx[0], idx[1], idx[2], comp, arr);
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