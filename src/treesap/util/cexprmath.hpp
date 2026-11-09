// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once

#include "AMReX_REAL.H"
#include <AMReX.H>

namespace treesap {
/// Implements math functions that can be evaluated at compile time.
namespace cexprmath {
/// Computes the factorial of n.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr std::int64_t factorial(int n) {
    std::int64_t res = 1;
    for (int i = 2; i <= n; ++i)
        res *= i;
    return res;
}

/// Computes binomial coefficient n choose k.
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

/// Computes base n raised to power k.
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

/// Returns sign factor (-1)^k.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr int sign(int k) { return (k & 1) ? -1 : 1; }

/// Computes generalized harmonic number H_(n,m).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr double generalised_harmonic_number(int n, int m) {
    double sum = 0.0;
    for (int k = 1; k <= n; k++) {
        sum += 1.0 / static_cast<double>(power(k, m));
    }
    return sum;
}
} // namespace cexprmath
} // namespace treesap