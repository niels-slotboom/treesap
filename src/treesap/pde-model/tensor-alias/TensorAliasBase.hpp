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
 * @namespace treesap::tensor
 * @brief Provides the compile-time resolved tensor aliasing system and related utilities.
 */
namespace tensor {
namespace detail {

/**
 * @brief Compile-time check that `Derived::get_relative()` and `Derived::restore_relative()` form a consistent
 * mapping between tensor indices and relative component numbers.
 * @internal
 *
 * The following is verified:
 * 1. `get_relative(restore_relative(c)) == c` for every component `c` in `[0, ncomp)`.
 * 2. `get_relative(idx)` lies in `[0, ncomp)` for every index combination `idx` in `[0, dim)^rank`. This also holds
 *    for tensors with symmetries, where `ncomp < dim^rank` and several index combinations map to the same component.
 *
 * For rank 0 (scalars) there is exactly one, empty, index combination, so `ncomp` must be 1 and the single mapping
 * `get_relative({})` must yield 0.
 *
 * @tparam Derived Implementation of `TensorAliasBase` to check.
 * @return `true` if the mapping is consistent, `false` otherwise.
 */
template <typename Derived> constexpr bool alias_is_consistent() {
    constexpr int rank = Derived::rank, dim = Derived::dim, ncomp = Derived::ncomp;

    // restore -> get must be the identity on [0, ncomp)
    for (int c = 0; c < ncomp; ++c) {
        if (Derived::get_relative(Derived::restore_relative(c)) != c)
            return false;
    }

    IndexArray<rank> idx{};

    if constexpr (rank == 0) {
        // a scalar has exactly one (empty) index combination and therefore exactly one component
        return ncomp == 1 && Derived::get_relative(idx) == 0;
    } else {
        // iterate over all dim^rank index combinations (odometer-style); each must land in [0, ncomp)
        while (true) {
            const int c = Derived::get_relative(idx);
            if (c < 0 || c >= ncomp)
                return false;

            int pos = rank - 1;
            while (pos >= 0 && ++idx[pos] == dim) {
                idx[pos] = 0;
                --pos;
            }
            if (pos < 0)
                break; // wrapped around completely: all combinations visited
        }
        return true;
    }
}

} // namespace detail

/**
 * @brief Concept that a derived class Derived must satisfy to inherit from TensorAliasBase.
 *
 * The following properties are checked:
 * - Must provide a `Derived::Base` type alias for the `treesap::tensor::TensorAliasBase` class, and `Derived` must
 *   publicly derive from it
 * - Must have `static constexpr int` members `ncomp`, `rank` and `dim`
 * - Must have `constexpr`-evaluable static functions restore_relative() and get_relative()
 * - restore_relative() and get_relative() must be mutually consistent, checked at compile time:
 *   - `get_relative(restore_relative(c)) == c` for all components `c` in `[0, ncomp)`
 *   - `get_relative(idx)` is in `[0, ncomp)` for all index combinations `idx` in `[0, dim)^rank`
 *   - for rank 0 (scalars), `ncomp` must be 1 and `get_relative({})` must be 0
 */
template <typename Derived>
concept TensorAliasImplementation = requires { // check for ncomp, rank, dim
    typename Derived::Base;                    // required alias to the base class TensorAliasBase<Derived,...>
    requires std::derived_from<Derived, typename Derived::Base>; // Base must actually be a base of Derived

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
} && requires { requires detail::alias_is_consistent<Derived>(); };

/**
 * @brief CRTP Base class template for aliasing a tensor onto a MultiFab component range
 * @tparam Derived Type of the implementing class
 * @tparam comp_base_ First index in the MultiFab component range belonging to the alias
 * @tparam ncomp_ Number of components being aliased
 * @tparam rank_ Rank of the tensor, i.e. its index count
 * @tparam dim_ Dimension of the tensor indices, i.e. their value range
 * @tparam ArrayType Type of the tensor components, typically `const amrex::Real`
 *
 * When implementing a custom `Derived` struct, it is recommended to follow the file structure
 * @code
 * struct MyTensor : TensorAliasBase<MyTensor, ...> {  ...  };
 * static_assert(treesap::tensor::TensorAliasImplementation<MyTensor>);
 * @endcode to ensure that the implementation provides a valid interface.
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
     * @brief Lightweight, by-value snapshot of all tensor components of the alias at a single grid cell.
     *
     * Holds a local copy of the `ncomp` components that the alias spans, so they can be loaded from the MultiFab once
     * and then reused in subsequent computations without repeated memory accesses. Components are addressed with
     * tensor indices exactly as in the parent alias, i.e. `temp({i, j})` rather than by raw component number.
     *
     * Instances are typically obtained via `TensorAliasBase::load(int, int, int)` and live purely in registers
     * or local memory; modifying a `Temporary` does not write back to the underlying MultiFab.
     */
    struct Temporary {
        using TemporaryArrayType = std::remove_cv_t<ArrayType>;

        /// @brief Mutable access to the stored component for the given tensor indices.
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr TemporaryArrayType&
        operator()(const IndexArray<rank>& tensor_indices) {
            return data[Derived::get_relative(tensor_indices)];
        }

        /// @brief Read-only access to the stored component for the given tensor indices.
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr const TemporaryArrayType&
        operator()(const IndexArray<rank>& tensor_indices) const {
            return data[Derived::get_relative(tensor_indices)];
        }

        /// Local copy of the aliased components, stored in relative (alias-local) component order.
        TemporaryArrayType data[ncomp]{};
    };

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
    static AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE constexpr int get_absolute(const IndexArray<rank>& tensor_indices) {
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
                                                                   const IndexArray<rank>& tensor_indices) const {
        return arr(i, j, k, get_absolute(tensor_indices));
    }

    /**
     * @brief Returns a `TensorAliasBase::Temporary` object containing a copy of the data in the
     * aliased array at `(i,j,k)`.
     * @param i Grid cell index in \f$x\f$-direction.
     * @param j Grid cell index in \f$y\f$-direction.
     * @param k Grid cell index in \f$z\f$-direction.
     * @return `TensorAliasBase::Temporary` object containing a copy of the data in the aliased array at `(i,j,k)`.
     */
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE Temporary load(int i, int j, int k) const {
        Temporary result{};
#pragma unroll
        for (int comp = 0; comp < ncomp; ++comp) {
            result.data[comp] = arr(i, j, k, comp_base + comp);
        }
        return result;
    }
};

} // namespace tensor

} // namespace treesap