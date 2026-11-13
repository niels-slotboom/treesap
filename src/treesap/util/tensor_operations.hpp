// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides constexpr contraction and index symmetrization utilities for tensor calculus operations.
 */

#pragma once

#include <AMReX.H>
#include <utility>

namespace treesap {

namespace tensor {

/**
 * @brief Helper function declaration for return type inference of variadic lambda templates and other callables.
 * @note This function is not to be called or used otherwise by a TreESaP user. It does not have a definition and is
 * meant to only be used in `decltype` arguments.
 */
template <typename F, int... Is>
AMREX_GPU_HOST_DEVICE constexpr auto contract_return_type(F&& f, std::integer_sequence<int, Is...>)
    -> decltype(f.template operator()<Is...>());

/**
 * @brief Performs a compile-time unrolled tensor contraction over the template parameter integer(s) of `f`.
 * @tparam rank Amount of indices being contracted.
 * @tparam dim Dimension of the indices; Index values range \f$0,...,\verb|dim|\f$.
 * @tparam F Type of the callable to be summed. Must return a summable type and be templated over `rank` integers.
 * @param f Instance of the callable; typically a lambda.
 * @note The indices of `F` are intentionally template parameters to make them known at compile time and hence usable
 * e.g. as direction parameters for `treesap::stencil::derivative`.
 *
 * Example usage, where `T` and `S` are some rank-2 tensor objects:
 * @code
 * contract<2,2>([]<int a, int b>(){return T(a,b) * S(a,b);});
 * @endcode
 * At compile time, this expands to
 * @code
 * T(0,0) * S(0,0) + T(0,1) * S(0,1) + T(1,0) * S(1,0) + T(1,1) * S(1,1);
 * @endcode
 */
template <int rank, int dim, typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
    static_assert(rank >= 1);
    return [&]<int... Is>(std::integer_sequence<int, Is...>) {
        using ReturnType = decltype(contract_return_type(f, std::make_integer_sequence<int, rank>{}));
        ReturnType result{};

        if constexpr (rank == 1) { // sum over f directly
            ((result += f.template operator()<Is>()), ...);
        } else { // sum over sums recursively
            ((result +=
              contract<rank - 1, dim>([&]<int... Js>() -> ReturnType { return f.template operator()<Is, Js...>(); })),
             ...);
        }

        return result;
    }(std::make_integer_sequence<int, dim>{});
}

/**
 * @brief Performs a compile-time unrolled tensor contraction over the template parameter integer(s) of `f`.
 * @tparam rank Amount of indices being contracted.
 * @tparam F Type of the callable to be summed. Must return a summable type and be templated over `rank` integers.
 * @param f Instance of the callable; typically a lambda.
 * @note This function defines a default for the more general `treesap::tensor::contract<int rank, int dim, typename F>`
 * function which sets the dimension to 3.
 *
 * Example usage, where `T` and `S` are some rank-2 tensor objects:
 * @code
 * contract<2>([]<int a, int b>(){return T(a,b) * S(a,b);});
 * @endcode
 * At compile time, this expands to
 * @code
 * T(0,0) * S(0,0) + T(0,1) * S(0,1) + T(0,2) * S(0,2)
 * + T(1,0) * S(1,0) + T(1,1) * S(1,1) + T(1,2) * S(1,2)
 * + T(2,0) * S(2,0) + T(2,1) * S(2,1) + T(2,2) * S(2,2);
 * @endcode
 */
template <int rank, typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
    return contract<rank, 3, F>(std::forward<F>(f));
}

/**
 * @brief Performs a compile-time unrolled tensor contraction over the template parameter integer of `f`.
 * @tparam F Type of the callable to be summed. Must return a summable type and be templated over a single `int`
 * parameter.
 * @param f Instance of the callable; typically a lambda.
 * @note This function defines a default for the more general `treesap::tensor::contract<int rank, int dim, typename F>`
 * function which sets the rank to 1 and dimension to 3.
 *
 * Example usage, where `X` and `Y` are some rank-1 tensor objects/vectors:
 * @code
 * contract([]<int a, int b>(){return X(a) * Y(a);});
 * @endcode
 * At compile time, this expands to
 * @code
 * X(0) * Y(0) + X(1) * Y(1) + X(2) * Y(2);
 * @endcode
 */
template <typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto contract(F&& f) {
    return contract<1, 3, F>(std::forward<F>(f));
}

/**
 * @brief Symmetrizes `f<a,b>` over the indices `a` and `b`.
 * @tparam a First index value.
 * @tparam b Second index value.
 * @tparam F Callable type with operator template over two integer indices.
 * @param f Callable object to symmetrize.
 * @return One half of the sum of both index permutations.
 *
 * Example usage:
 * @code
 * auto result = symmetrise<a, b>([&]<int a_, int b_>() {
 *     return tensor(a_, b_);
 * });
 * @endcode
 * Expands to `1/2 * (tensor(a,b) + tensor(b,a))`
 */
template <int a, int b, typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto symmetrise(F&& f) {
    return 0.5 * (f.template operator()<a, b>() + f.template operator()<b, a>());
}

/**
 * @brief Anti-symmetrizes `f<a,b>` over the indices `a` and `b`.
 * @tparam a First index value.
 * @tparam b Second index value.
 * @tparam F Callable type with operator template over two integer indices.
 * @param f Callable object to anti-symmetrize.
 * @return One half of the alternating difference of both index permutations.
 *
 * Example usage:
 * @code
 * auto result = antisymmetrise<a, b>([&]<int a_, int b_>() {
 *     return tensor(a_, b_);
 * });
 * @endcode
 * Expands to `1/2 * (tensor(a,b) - tensor(b,a))`
 */
template <int a, int b, typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto antisymmetrise(F&& f) {
    if constexpr (a == b) // dont waste memory IO and compute if indices are the same
        return 0.0;
    return 0.5 * (f.template operator()<a, b>() - f.template operator()<b, a>());
}

/**
 * @brief Symmetrizes `f<a, b, c>` over the three indices `a`, `b`, and `c`.
 * @tparam a First index value.
 * @tparam b Second index value.
 * @tparam c Third index value.
 * @tparam F Callable type with operator template over three integer indices.
 * @param f Callable object to symmetrize.
 * @return One sixth of the sum of all six index permutations.
 *
 * Example usage:
 * @code
 * auto result = symmetrise<a, b, c>([&]<int a_, int b_, int c_>() {
 *     return tensor(a_, b_, c_);
 * });
 * @endcode
 */
template <int a, int b, int c, typename F> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto symmetrise(F&& f) {
    return (1.0 / 6.0) *
           (f.template operator()<a, b, c>() + f.template operator()<b, c, a>() + f.template operator()<c, a, b>() +
            f.template operator()<a, c, b>() + f.template operator()<b, a, c>() + f.template operator()<c, b, a>());
}

/**
 * @brief Anti-symmetrizes `f<a, b, c>` over the three indices `a`, `b`, and `c`.
 * @tparam a First index value.
 * @tparam b Second index value.
 * @tparam c Third index value.
 * @tparam F Callable type with operator template over three integer indices.
 * @param f Callable object to anti-symmetrize.
 * @return One sixth of the alternating sum of all six index permutations.
 *
 * Example usage:
 * @code
 * auto result = antisymmetrise<a, b, c>([&]<int a_, int b_, int c_>() {
 *     return tensor(a_, b_, c_);
 * });
 * @endcode
 */
template <int a, int b, int c, typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto antisymmetrise(F&& f) {
    if constexpr (a == b || b == c || c == a) // dont waste memory IO and compute if any two indices are the same
        return 0.0;
    return (1.0 / 6.0) *
           (f.template operator()<a, b, c>() + f.template operator()<b, c, a>() + f.template operator()<c, a, b>() -
            f.template operator()<a, c, b>() - f.template operator()<b, a, c>() - f.template operator()<c, b, a>());
}

/**
 * @brief Sums f over cyclic permutations of three indices at a, b, and c.
 * @tparam a First index value.
 * @tparam b Second index value.
 * @tparam c Third index value.
 * @tparam F Callable type with operator template over three integer indices.
 * @param f Callable object to evaluate.
 * @return One third of the sum of the three cyclic permutations.
 *
 * Example usage:
 * @code
 * auto result = cyclic_permute<a, b, c>([&]<int a_, int b_, int c_>() {
 *     return tensor(a_, b_, c_);
 * });
 * @endcode
 * Expands to `1/3 * (tensor(a,b,c) + tensor(b,c,a) + tensor(c,a,b))`
 */
template <int a, int b, int c, typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr auto cyclic_permutation_sum(F&& f) {
    return (1.0 / 3.0) *
           (f.template operator()<a, b, c>() + f.template operator()<b, c, a>() + f.template operator()<c, a, b>());
}

} // namespace tensor

} // namespace treesap