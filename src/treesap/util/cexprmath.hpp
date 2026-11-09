// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once

#include "AMReX_REAL.H"
#include <AMReX.H>

namespace treesap {

/**
 * @namespace treesap::cexprmath
 * @brief Implements math functions that can be evaluated at compile time.
 *
 * Provides `constexpr` implementations of the factorial function, binomial coefficients, positive integer powers, the
 * \f$(-1)^n\f$ sign function, and generalised harmonic numbers \f$H_{n,m}\f$. This namespace is primarily used in the
 * stencil coefficient implementations in `util/stencil.hpp`.
 */
namespace cexprmath {
/**
 * @brief Computes the factorial \f$n!\f$.
 * @param n The value to compute the factorial of.
 * @return The factorial of `n`.
 * @note This function is `constexpr` evaluable.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr std::int64_t factorial(int n) {
    std::int64_t res = 1;
    for (int i = 2; i <= n; ++i)
        res *= i;
    return res;
}

/**
 * @brief Computes the binomial coefficient \f$\binom{n}{k} = \frac{n!}{k!(n-k)!}\f$.
 * @param n The upper binomial coefficient variable.
 * @param k The lower binomial coefficient variable.
 * @return The binomial coefficient \f$\binom{n}{k} = \frac{n!}{k!(n-k)!}\f$.
 * @note This function is `constexpr` evaluable.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr std::int64_t binom(int n, int k) {
    if (k < 0 || k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    if (k > n / 2)
        k = n - k;

    std::int64_t res = 1;
    for (int i = 1; i <= k; ++i) {
        res = (res * (n - i + 1)) / i;
    }
    return res;
}

/**
 * @brief Computes the \f$k\f$-th power of \f$n\f$.
 * @param n The base variable.
 * @param k The exponent variable.
 * @return The value of the expression \f$n^k\f$.
 * @note This function is `constexpr` evaluable.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr std::int64_t power(std::int64_t n, int k) {
    std::int64_t result = 1;
    while (k > 0) {
        if (k & 1)
            result *= n;
        n *= n;
        k >>= 1;
    }
    return result;
}

/**
 * @brief Computes the sign factor \f$(-1)^k\f$.
 * @param k The exponent variable.
 * @return The value of the expression \f$(-1)^k\f$.
 * @note This function is `constexpr` evaluable.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr int sign(int k) { return (k & 1) ? -1 : 1; }

/**
 * @brief Computes the generalised harmonic number \f$H_{n,m}\f$.
 * @param n The amount of terms being summed.
 * @param k The exponent of the reciprocal integers.
 * @return The generalised harmonic number \f$H_{n,m}\f$.
 * @note This function is `constexpr` evaluable.
 * The generalised harmonic numbers are defined as
 * \f$$
 *      H_{n,m} = \sum_{k=1}^n \frac{1}{k^m}.
 * \f$$
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr double generalised_harmonic_number(int n, int m) {
    double sum = 0.0;
    for (int k = 1; k <= n; k++) {
        sum += 1.0 / static_cast<double>(power(k, m));
    }
    return sum;
}
} // namespace cexprmath
} // namespace treesap