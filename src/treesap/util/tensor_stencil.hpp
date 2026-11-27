
// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides derivative wrappers for tensor alias implementations over structured grid stencils.
 */

#pragma once

#include "pde-model/tensor-alias/TensorAliasBase.hpp"
#include "stencil_impl.hpp"

namespace treesap {
namespace stencil {

/**
 * @brief Evaluates a centered first derivative stencil on a tensor alias.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam dir Derivative direction.
 * @tparam AliasType Type of the tensor alias.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @param tensor Reference to the tensor alias to evaluate stencil on.
 * @param tensor_indices The indices of the tensor component to be differentiated.
 * @param idx Index vector of the stencil center cell.
 * @param geom GPU-safe domain geometry data used to determine grid scaling factors.
 * @return A finite-difference approximation of \f$\partial_a\verb|tensor|_{b...}\f$ evaluated at the cell indexed by
 * \p idx where \f$a=\verb|dir|\f$ and \f$(b,...) = \verb|tensor_indices|\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires the aliased array to provide `ngrow` ghost cells. For
 * implementation details see `treesap::stencil_impl`.
 */
template <int ngrow, int dir, typename AliasType>
    requires(tensor::TensorAliasImplementation<AliasType>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(const AliasType& tensor,
                                                         const tensor::IndexArray<AliasType::rank>& tensor_indices,
                                                         const amrex::IntVect& idx, const amrex::GeometryData& geom) {

    return stencil_impl::derivative_impl<ngrow, typename AliasType::Base::arr_t, dir>::eval(
        idx[dir::x], idx[dir::y], idx[dir::z], AliasType::Base::get_absolute(tensor_indices), tensor.arr, geom);
}

/**
 * @brief Evaluates a centered second derivative stencil on a tensor alias.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam dir0 First derivative direction.
 * @tparam dir1 Second derivative direction.
 * @tparam AliasType Type of the tensor alias.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @param tensor Reference to the tensor alias to evaluate stencil on.
 * @param tensor_indices The indices of the tensor component to be differentiated.
 * @param idx Index vector of the stencil center cell.
 * @param geom GPU-safe domain geometry data used to determine grid scaling factors.
 * @return A finite-difference approximation of \f$\partial_a\partial_b\verb|tensor|_{c...}\f$ evaluated at the cell
 * indexed by \p idx where \f$a=\verb|dir0|\f$, \f$b=\verb|dir1|\f$ and
 * \f$(c,...) = \verb|tensor_indices|\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires the aliased array to provide `ngrow` ghost cells. For
 * implementation details see `treesap::stencil_impl`.
 */
template <int ngrow, int dir0, int dir1, typename AliasType>
    requires(tensor::TensorAliasImplementation<AliasType>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(const AliasType& tensor,
                                                         const tensor::IndexArray<AliasType::rank>& tensor_indices,
                                                         const amrex::IntVect& idx, const amrex::GeometryData& geom) {

    return stencil_impl::derivative_impl<ngrow, typename AliasType::Base::arr_t, dir0, dir1>::eval(
        idx[dir::x], idx[dir::y], idx[dir::z], AliasType::Base::get_absolute(tensor_indices), tensor.arr, geom);
}

/**
 * @brief Evaluates a centered third derivative stencil on a tensor alias.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam dir0 First derivative direction.
 * @tparam dir1 Second derivative direction.
 * @tparam dir2 Third derivative direction.
 * @tparam AliasType Type of the tensor alias.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @param tensor Reference to the tensor alias to evaluate stencil on.
 * @param tensor_indices The indices of the tensor component to be differentiated.
 * @param idx Index vector of the stencil center cell.
 * @param geom GPU-safe domain geometry data used to determine grid scaling factors.
 * @return A finite-difference approximation of \f$\partial_a\partial_b\partial_c\verb|tensor|_{d...}\f$ evaluated
 * at the cell indexed by \p idx where \f$a=\verb|dir0|\f$, \f$b=\verb|dir1|\f$,
 * \f$c=\verb|dir2|\f$ and \f$(d,...) = \verb|tensor_indices|\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires the aliased array to provide `ngrow` ghost cells. For
 * implementation details see `treesap::stencil_impl`.
 */
template <int ngrow, int dir0, int dir1, int dir2, typename AliasType>
    requires(tensor::TensorAliasImplementation<AliasType>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(const AliasType& tensor,
                                                         const tensor::IndexArray<AliasType::rank>& tensor_indices,
                                                         const amrex::IntVect& idx, const amrex::GeometryData& geom) {
    return stencil_impl::derivative_impl<ngrow, typename AliasType::Base::arr_t, dir0, dir1, dir2>::eval(
        idx[dir::x], idx[dir::y], idx[dir::z], AliasType::Base::get_absolute(tensor_indices), tensor.arr, geom);
}

} // namespace stencil
} // namespace treesap
