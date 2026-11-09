// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>

#include "cexprmath.hpp"

namespace treesap {

namespace stencil {
/// Named shorthands for directions in \f$d=3\f$; `dir::x = 0`, `dir::y = 1` and `dir::z = 2`.
enum dir : int { x = 0, y = 1, z = 2 };
} // namespace stencil

/**
 * @brief Provides implementations for finite-difference stencils.
 */
namespace stencil_impl {

/**
 * @namespace treesap::stencil_impl::weights
 * @brief Defines `constexpr`-evaluable functions to compute finite-difference stencil coefficients for arbitrary width,
 * centered and unmixed first, second and third derivatives.
 *
 * The mathematical expressions for the stencil coefficients are taken from
 * [Wikipedia: Finite difference coefficient](https://en.wikipedia.org/wiki/Finite_difference_coefficient).
 */
namespace weights {
/**
 * @brief Implements weights for first-order derivative stencils with width `2 * ngrow + 1`, at offset away from center
 * point.
 * @tparam ngrow The amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`, centered around
 * `offset = 0`.
 * @param offset The offset away from the central cell.
 * @note This function is `constexpr` evaluable.
 *
 * The evaluated expression is
 * \f[
 *      \begin{cases}
 *           0, & p = 0,\\ \\
 *           \displaystyle\frac{(-1)^{p+1}(n!)^2}{p(n-p)!(n+p)!},&\quad p \neq 0,\\
 *      \end{cases}
 * \f]
 * where \f$n=\f$`ngrow` and \f$p=\f$`offset`.
 */
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

/**
 * @brief Implements weights for unmixed second-order derivative stencils with width `2 * ngrow + 1`, at `offset` away
 * from center point.
 * @tparam ngrow The amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`, centered around
 * `offset = 0`.
 * @param offset The offset away from the central cell.
 * @note This function is `constexpr` evaluable.
 *
 * The evaluated expression is
 * \f[
 *      \begin{cases}
 *           -2 H_{n,2}, & p = 0,\\ \\
 *           \displaystyle\frac{2(-1)^{p+1}(n!)^2}{p^2(n-p)!(n+p)!},&\quad p \neq 0,\\
 *      \end{cases}
 * \f]
 * where \f$n=\f$`ngrow`, \f$p=\f$`offset` and \f$H_{n,m}\f$ are the generalised harmonic numbers (see @ref
 * treesap::cexprmath::generalised_harmonic_number).
 */
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

/**
 * @brief Implements weights for unmixed third-order derivative stencils with width `2 * ngrow + 1`, at `offset` away
 * from center point.
 * @tparam ngrow The amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`, centered around
 * `offset = 0`.
 * @param offset The offset away from the central cell.
 * @note This function is `constexpr` evaluable.
 *
 * The evaluated expression is
 * \f[
 *      \begin{cases}
 *           0, & p = 0,\\ \\
 *           \displaystyle\frac{3!(-1)^{p+1}(n!)^2}{p^3(n-p)!(n+p)!}(1-p^2 H_{n,2}),&\quad p \neq 0,\\
 *      \end{cases}
 * \f]
 * where \f$n=\f$`ngrow` and \f$p=\f$`offset` and \f$H_{n,m}\f$ are the generalised harmonic numbers (see @ref
 * treesap::cexprmath::generalised_harmonic_number).
 */
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

/**
 * @brief Base template for derivative_impl struct.
 * @tparam ngrow Amount of neighbour cells in either direction away from the center.
 * @tparam ArrayType Datatype of the array to evaluate stencil on.
 * @tparam Dirs... List of direction integers along which derivatives are taken, `x = 0`, `y = 1`, `z = 2`.
 *
 * The implementation of derivative stencils is deferred to this struct template because function templates do not
 * allow partial template specialisation.
 */
template <int ngrow, typename ArrayType, int... Dirs> struct derivative_impl {
    static_assert(always_false<ngrow>,
                  "No stencil::derivative implementation available for requested template parameters");
    /**
     * @brief Base evaluation fallback function.
     * @param i First cell index.
     * @param j Second cell index.
     * @param k Third cell index.
     * @param comp One-dimensional index of the MultiFab component.
     * @param arr Reference to the array to evaluate stencil on.
     * @return Zero.
     *
     * @note This function should never be called; it is the fallback for the situation where no explicit implementation
     * of `derivative_impl::eval` exists.
     *
     * The implementation of derivative stencils is deferred to this struct template because function templates do not
     * allow partial template specialisation.
     */
    static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType eval(int i, int j, int k, int comp,
                                                                   const amrex::Array4<ArrayType>& arr) {
        return 0;
    }
};

// --- FIRST DERIVATIVES ---
/**
 * @brief Implementation struct for compile-time unrolled first-order derivative stencils.
 * @tparam ngrow Amount of neighbour cells used in either direction away from the center.
 * @tparam ArrayType Datatype of the array to evaluate stencil on.
 * @tparam dir Derivative direction.
 *
 * The implementation of derivative stencils is deferred to this struct template because function templates do not
 * allow partial template specialisation.
 */
template <int ngrow, typename ArrayType, int dir> struct derivative_impl<ngrow, ArrayType, dir> {
    /**
     * @brief Compile-time unrolled first-order derivative evaluation function.
     * @param i First cell index.
     * @param j Second cell index.
     * @param k Third cell index.
     * @param comp One-dimensional index of the MultiFab component.
     * @param arr Reference to the array to evaluate stencil on.
     * @return Finite-difference stencil approximation of the first derivative of `arr[comp]` along `dir` at `(i,j,k)`.
     *
     * The implementation of derivative stencils is deferred to this struct template because function templates do not
     * allow partial template specialisation.
     */
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
/**
 * @brief Implementation struct for compile-time unrolled second-order derivative stencils.
 * @tparam ngrow Amount of neighbour cells used in either direction away from the center.
 * @tparam ArrayType Datatype of the array to evaluate stencil on.
 * @tparam dir0 First derivative direction.
 * @tparam dir1 Second derivative direction.
 *
 * The implementation of derivative stencils is deferred to this struct template because function templates do not
 * allow partial template specialisation.
 */
template <int ngrow, typename ArrayType, int dir0, int dir1> struct derivative_impl<ngrow, ArrayType, dir0, dir1> {
    /**
     * @brief SCompile-time unrolled second-order derivative evaluation function.
     * @param i First cell index.
     * @param j Second cell index.
     * @param k Third cell index.
     * @param comp One-dimensional index of the MultiFab component.
     * @param arr Reference to the array to evaluate stencil on.
     * @return Finite-difference stencil approximation of the second derivative of `arr[comp]` along `dir0, dir1` at
     * `(i, j, k)`.
     *
     * The implementation of derivative stencils is deferred to this struct template because function templates do not
     * allow partial template specialisation.
     */
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
/**
 * @brief Implementation struct for compile-time unrolled third-order derivative stencils.
 * @tparam ngrow Amount of neighbour cells used in either direction away from the center.
 * @tparam ArrayType Datatype of the array to evaluate stencil on.
 * @tparam dir0 First derivative direction.
 * @tparam dir1 Second derivative direction.
 * @tparam dir2 Third derivative direction.

 * The implementation of derivative stencils is deferred to this struct template because function templates do not
 * allow partial template specialisation.
 */
template <int ngrow, typename ArrayType, int dir0, int dir1, int dir2>
struct derivative_impl<ngrow, ArrayType, dir0, dir1, dir2> {
    /**
     * @brief Compile-time unrolled third-order derivative evaluation function.
     * @param i First cell index.
     * @param j Second cell index.
     * @param k Third cell index.
     * @param comp One-dimensional index of the MultiFab component.
     * @param arr Reference to the array to evaluate stencil on.
     * @return Finite-difference stencil approximation of the third derivative of `arr[comp]` along `dir0, dir1, dir2`
     * at
     * `(i, j, k)`.
     *
     * The implementation of derivative stencils is deferred to this struct template because function templates do not
     * allow partial template specialisation.
     */
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
} // namespace stencil_impl

} // namespace treesap