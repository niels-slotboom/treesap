// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides `treesap::tensor::TensorAliasBase` base class and `concept`
 * `treesap::tensor::TensorAliasImplementation`.
 */

#pragma once

#include "AMReX_Array4.H"
#include "AMReX_Extension.H"

#include <concepts>
#include <type_traits>

#include "IndexArray.hpp"

namespace treesap {
/**
 * @namespace
 * @brief Provides the compile-time resolved tensor aliasing system and related utilities.
 */
namespace tensor {

/**
 * @brief Concept that a derived class Derived must satisfy to inherit from TensorAliasBase.
 *
 * The following properties are checked:
 * - Must provide a `Derived::Base` type alias for the `treesap::tensor::TensorAliasBase` class
 * - Must have `static constexpr int` members `ncomp`, `rank` and `dim`
 * - Must have `constexpr`-evaluable static functions restore_relative() and get_relative()
 */
template <typename Derived>
concept TensorAliasImplementation = requires { // check for ncomp, rank, dim
    typename Derived::Base;                    // required alias to the base class TensorAliasBase<Derived,...>

    typename std::integral_constant<int, Derived::ncomp>;
    requires Derived::ncomp > 0;

    typename std::integral_constant<int, Derived::rank>;
    requires Derived::rank >= 0;

    typename std::integral_constant<int, Derived::dim>;
    requires Derived::dim > 0;
} && requires(int comp, IndexArray<Derived::rank> tensor_indices) {
    // check for static restore_relative() and get_relative()
    { Derived::restore_relative(comp) } -> std::same_as<IndexArray<Derived::rank>>;
    { Derived::get_relative(tensor_indices) } -> std::same_as<int>;
} && requires { // check if get_relative() is constexpr
    typename std::integral_constant<int, Derived::get_relative(IndexArray<Derived::rank>{})>;
} && requires { // check if restore_relative() is constexpr
    requires[]() consteval {
        auto result = Derived::restore_relative(0); // attempt to evaluate in consteval context
        return true;
    }
    ();
};

/**
 * @brief CRTP Base class template for aliasing a tensor onto a MultiFab component range
 * @tparam Derived Type of the implementing class
 * @tparam comp_base_ First index in the MultiFab component range belonging to the alias
 * @tparam ncomp_ Number of components being aliased
 * @tparam rank_ Rank of the tensor, i.e. its index count
 * @tparam dim_ Dimension of the tensor indices, i.e. their value range
 * @tparam ArrayType Type of the tensor components, typically `const amrex::Real`
 */
template <typename Derived, int comp_base_, int ncomp_, int rank_, int dim_, typename ArrayType>
struct TensorAliasBase {
    /// absolute component in the  MultiFab component range where the tensor alias' memory starts
    static constexpr int comp_base = comp_base_;

    /// number of components the alias spans
    static constexpr int ncomp = ncomp_;

    /// rank of the tensor; this is the amount of indices it has
    static constexpr int rank = rank_;

    /// dimension of the tensor space; the range of the values indices can take
    static constexpr int dim = dim_;

    /// value for `comp_base` of the next tensor in the MultiFab component range
    static constexpr int next = comp_base_ + ncomp_;

    // assert sensible values for the above compile-time constants
    static_assert(ncomp > 0);
    static_assert(rank >= 0);
    static_assert(dim > 0);

    /// reference to the underlying MultiFab memory chunk
    const amrex::Array4<ArrayType>& arr;

    /// the type of the data stored in arr
    using arr_t = ArrayType;

    /**
     * @brief Checks if an absolute index in the MultiFab component range falls within the valid range for this tensor
     * alias.
     * @param absolute_comp The absolute component index in the flat MultiFab component range.
     * @return True if the component is within range, false otherwise.
     * @note This function is static to ensure `constexpr`-evaluability, since it does not depend on instance state. It
     * can be called using `.contains()` regardless.
     */
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr bool contains(int absolute_comp) {
        const int comp_rel_to_base = absolute_comp - comp_base;
        return comp_rel_to_base >= 0 && comp_rel_to_base < ncomp;
    }

    /**
     * @brief Computes the absolute index in the MultiFab component range for given tensor indices.
     * @param tensor_indices The tensor index values (empty for rank 0 scalars).
     * @return The absolute component index in the underlying array.
     * @note This function is static to ensure `constexpr`-evaluability, since it does not depend on instance state. It
     * can be called using `.get_absolute()` regardless.
     */
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr int get_absolute(const IndexArray<rank> tensor_indices) {
        return comp_base + Derived::get_relative(tensor_indices);
    }

    /**
     * @brief Restores tensor indices from an absolute index in the MultiFab component range.
     * @param absolute_comp The absolute component index.
     * @return The corresponding tensor indices.
     * @note This function is static to ensure `constexpr`-evaluability, since it does not depend on instance state. It
     * can be called using `.restore_absolute()` regardless.
     */
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr IndexArray<rank> restore_absolute(int absolute_comp) {
        return Derived::restore_relative(absolute_comp - comp_base);
    }

    /**
     * @brief Accesses a specific array element using spatial and tensor indices.
     * @param i Grid cell index in \f$x\f$-direction.
     * @param j Grid cell index in \f$y\f$-direction.
     * @param k Grid cell index in \f$z\f$-direction.
     * @param tensor_indices The tensor index values ({} for rank 0 scalars).
     * @return Reference to the array element.
     */
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType& operator()(int i, int j, int k,
                                                                   const IndexArray<rank> tensor_indices) const {
        return arr(i, j, k, get_absolute(tensor_indices));
    }
};

} // namespace tensor

} // namespace treesap