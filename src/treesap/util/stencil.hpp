// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides finite-difference derivative stencils and Kreiss-Oliger dissipation operators for AMReX arrays.
 */

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
/**
 * @brief Evaluates a centered derivative stencil on a raw `amrex::Array4`.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam ArrayType Type of the data stored in the `arr` argument.
 * @tparam Dirs... Up to three directions along which derivatives are approximated.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @note To obtain an approximation of a derivative, the return value of this function must be multiplied by
 * \f$1/\Delta x^k\f$ where \f$k\f$ is its order.
 * @param i First center cell index.
 * @param j Second center cell index.
 * @param k Third center cell index.
 * @param comp Component index in the MultiFab component range.
 * @param arr Reference to the array to evaluate stencil on.
 * @return A finite-difference approximation of \f$\Delta x \partial_a
 * \,\verb|arr|(\verb|i, j, k, comp|)\f$, \f$\Delta x^2 \partial_a \partial_b
 * \,\verb|arr|(\verb|i, j, k, comp|)\f$ or \f$\Delta x^3 \partial_a \partial_b \partial_c\,\verb|arr|(\verb|i, j, k,
 * comp|)\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires `arr` to provide `ngrow` ghost cells. The number of directions
 * provided in `int... Dir` defines the derivative order; up to third derivatives are implemented. For implementation
 * details see `treesap::stencil_impl`.
 */
template <int ngrow, typename ArrayType, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType derivative(int i, int j, int k, int comp,
                                                              const amrex::Array4<ArrayType>& arr) {
    static_assert(((Dirs >= 0 && Dirs <= 2) && ...),
                  "All direction template parameters must be between 0 and 2 (inclusive)");
    return stencil_impl::derivative_impl<ngrow, ArrayType, Dirs...>::eval(i, j, k, comp, arr);
}

/**
 * @brief Evaluates a centered derivative stencil on a raw `amrex::Array4`.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam ArrayType Type of the data stored in the `arr` argument.
 * @tparam Dirs... Up to three directions along which derivatives are approximated.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @note To obtain an approximation of a derivative, the return value of this function must be multiplied by
 * \f$1/\Delta x^k\f$ where \f$k\f$ is its order.
 * @param idx Index vector of the center cell.
 * @param comp Component index in the MultiFab component range.
 * @param arr Reference to the array to evaluate stencil on.
 * @return A finite-difference approximation of \f$\Delta x \partial_a
 * \,\verb|arr|(\verb|idx, comp|)\f$, \f$\Delta x^2 \partial_a \partial_b
 * \,\verb|arr|(\verb|idx, comp|)\f$ or \f$\Delta x^3 \partial_a \partial_b \partial_c\,\verb|arr|(\verb|idx, comp|)\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires `arr` to provide `ngrow` ghost cells. The number of directions
 * provided in `int... Dir` defines the derivative order; up to third derivatives are implemented. For implementation
 * details see `treesap::stencil_impl`.
 */
template <int ngrow, typename ArrayType, int... Dirs>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE ArrayType derivative(amrex::IntVect idx, int comp,
                                                              const amrex::Array4<ArrayType>& arr) {
    static_assert(((Dirs >= 0 && Dirs <= 2) && ...),
                  "All direction template parameters must be between 0 and 2 (inclusive)");
    return stencil_impl::derivative_impl<ngrow, ArrayType, Dirs...>::eval(idx[0], idx[1], idx[2], comp, arr);
}

// --- --- --- KREISS-OLIGER DISSIPATION TERMS --- --- ---
/**
 * @brief Computes the Kreiss-Oliger (KO) dissipation term along a given coordinate direction `dir`.
 * @tparam ngrow Order of the KO operator.
 * @tparam dir Coordinate direction along which KO operator is evaluated.
 * @tparam ArrayType Type of the data stored in the `arr` argument.
 * @note The KO stencils are unrolled at compile time and do not produce runtime call overhead.
 * @note To obtain the value of the KO operator evaluation, the return value of this function must be multiplied by
 * \f$1/\Delta x\f$ independent of its order.
 * @param i First center cell index.
 * @param j Second center cell index.
 * @param k Third center cell index.
 * @param comp Component index in the MultiFab component range.
 * @param arr Reference to the array to evaluate stencil on.
 * @return Value of the KO operator evaluation \f$\Delta x \mathcal{D}_\text{KO}\verb|arr|(\verb|i, j, k, comp|)\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires `arr` to provide `ngrow` ghost cells. In this context, the
 * Kreiss-Oliger dissipation term of order \f$r = \verb|ngrow|\f$ is normalised as
 * \f[
 *      \mathcal{D}_\text{KO}^r = (-1)^{r+1}\frac{\Delta x^{2r-1}}{2^{2r}}(D_+ D_-)^r,
 * \f]
 * where \f$D_+\f$ and \f$D_-\f$ are the forward and backward difference operators, respectively, given by
 * \f[
 *      D_\pm f(x) = \pm \frac{f(x\pm \Delta x)-f(x)}{\Delta x}.
 * \f]
 */
template <int ngrow, int dir, typename ArrayType>
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType kreiss_oliger(int i, int j, int k, int comp,
                                                                 const amrex::Array4<ArrayType>& arr) {
    using namespace cexprmath;

    static_assert(ngrow > 0, "Can only apply Kreiss-Oliger stencil for ngrow >= 1");
    constexpr int i_offset = (dir == stencil::dir::x) ? 1 : 0;
    constexpr int j_offset = (dir == stencil::dir::y) ? 1 : 0;
    constexpr int k_offset = (dir == stencil::dir::z) ? 1 : 0;

    constexpr int r = ngrow;
    constexpr double norm = 1.0 / static_cast<double>(power(2, 2 * ngrow));

    std::decay_t<ArrayType> result = -binom(2 * r, r) * arr(i, j, k, comp);
#pragma unroll
    for (int s = 1; s <= ngrow; ++s) {
        result += -alternating_sign(s) * binom(2 * r, r + s) *
                  (arr(i + s * i_offset, j + s * j_offset, k + s * k_offset, comp) +
                   arr(i - s * i_offset, j - s * j_offset, k - s * k_offset, comp));
    }
    return result * norm;
}

} // namespace stencil

} // namespace treesap