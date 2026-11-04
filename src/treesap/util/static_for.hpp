#pragma once

#include "AMReX_Gpu.H"
#include <utility>

/**
 * @brief Unrolls a loop at compile time over the half-open interval [Start, End).
 */
template <int Start, int End, typename F> AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr void static_for(F&& f) {
    static_assert(Start <= End, "Start must be less than or equal to End.");
    [&]<auto... Is>(std::integer_sequence<decltype(Start), Is...>) {
        (f.template operator()<Start + Is>(), ...);
    }(std::make_integer_sequence<decltype(Start), End - Start>{});
}

/**
 * @brief Unrolls a loop at compile time over the half-open interval [0, End).
 */
template <int End, typename F> AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr void static_for(F&& f) {
    static_for<0, End>(std::forward<F>(f));
}