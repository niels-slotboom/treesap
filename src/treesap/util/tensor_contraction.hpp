#pragma once

#include <AMReX.H>
#include <utility>

#pragma once

#include <AMReX.H>
#include <utility>

/**
 * @brief Computes the contraction of a callable over the half-open interval [Start, End).
 * @tparam Start Lower bound of the interval (inclusive).
 * @tparam End Upper bound of the interval (exclusive).
 * @tparam F Callable type with a templated operator().
 * @param f Callable object to evaluate at each index.
 * @return Accumulated result of the contraction.
 */
template <int Start, int End, typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
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
 * @tparam End Upper bound of the interval (exclusive).
 * @tparam F Callable type with a templated operator().
 * @param f Callable object to evaluate at each index.
 * @return Accumulated result of the contraction.
 */
template <int End, typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
    return contract<0, End>(std::forward<F>(f));
}

/**
 * @brief Computes the contraction of a callable over the half-open interval [0, 3).
 * @tparam F Callable type with a templated operator().
 * @param f Callable object to evaluate at each index.
 * @return Accumulated result of the contraction.
 */
template <typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
    return contract<0, 3>(std::forward<F>(f));
}
