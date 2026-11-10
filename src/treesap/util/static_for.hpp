// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides compile-time loop unrolling utilities for AMReX kernels and constexpr execution.
 */

#pragma once

#include "AMReX_Gpu.H"
#include <utility>

namespace treesap {

/**
 * @brief Unrolls a loop at compile time over the half-open interval `[Start, End)`.
 * @tparam Start Inclusive lower bound of the loop index.
 * @tparam End Exclusive lower bound of the loop index.
 * @tparam F Callable type templated on the loop index integer, typically a lambda template `[]<int i>(){...}`.
 * @param f Instance of F.
 *
 * The counter variable is the integer template argument of `template<int i>F`.
 *
 * Example Usage:
 * @code
 * static_for<2,5>([=]<int i>(){
 *   std::cout << "Iteration " << i << " of static_for" << std::endl;
 * });
 * @endcode
 * This unrolls to:
 * @code
 * std::cout << "Iteration " << 2 << " of static_for" << std::endl;
 * std::cout << "Iteration " << 3 << " of static_for" << std::endl;
 * std::cout << "Iteration " << 4 << " of static_for" << std::endl;
 * @endcode
 */
template <int Start, int End, typename F> AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr void static_for(F&& f) {
    static_assert(Start <= End, "Start must be less than or equal to End.");
    [&]<auto... Is>(std::integer_sequence<decltype(Start), Is...>) {
        (f.template operator()<Start + Is>(), ...);
    }(std::make_integer_sequence<decltype(Start), End - Start>{});
}

/**
 * @brief Unrolls a loop at compile time over the half-open interval `[0, End)`.
 * @tparam End Exclusive lower bound of the loop index.
 * @tparam F Callable type templated on the loop index integer, typically a lambda template `[]<int i>(){...}`.
 * @param f Instance of F.
 *
 * The counter variable is the integer template argument of `template<int i>F`.
 *
 * Example Usage:
 * @code
 * static_for<3>([=]<int i>(){
 *   std::cout << "Iteration " << i << " of static_for" << std::endl;
 * });
 * @endcode
 * This unrolls to:
 * @code
 * std::cout << "Iteration " << 0 << " of static_for" << std::endl;
 * std::cout << "Iteration " << 1 << " of static_for" << std::endl;
 * std::cout << "Iteration " << 2 << " of static_for" << std::endl;
 * @endcode
 */
template <int End, typename F> AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr void static_for(F&& f) {
    static_for<0, End>(std::forward<F>(f));
}

} // namespace treesap