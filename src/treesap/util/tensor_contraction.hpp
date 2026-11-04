#pragma once

#include <AMReX.H>
#include <utility>

/**
 * @brief Computes the contraction of a callable over the half-open interval [Start, End).
 * @tparam F Callable type with a templated operator().
 * @tparam Start Lower bound of the interval (inclusive).
 * @tparam End Upper bound of the interval (exclusive).
 * @param f Callable object to evaluate at each index.
 * @return Accumulated result of the contraction.
 */
template <typename F, int Start, int End> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
    static_assert(Start <= End, "Start must be less than or equal to End.");
    return [&]<auto... Is>(std::integer_sequence<decltype(Start), Is...>) {
        using return_t = decltype(f.template operator()<Start>());
        return_t result{};
        ((result += f.template operator()<Start + Is>()), ...);
        return result;
    }(std::make_integer_sequence<decltype(Start), End - Start>{});
}

/**
 * @brief Computes the contraction of a callable over the half-open interval [0, End).
 * @tparam F Callable type with a templated operator().
 * @tparam End Upper bound of the interval (exclusive), defaults to 3.
 * @param f Callable object to evaluate at each index.
 * @return Accumulated result of the contraction.
 */
template <typename F, int End = 3> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
    return contract<F, 0, End>(std::forward<F>(f));
}