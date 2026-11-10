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
 * @brief Evaluates a centered first-order derivative stencil on a tensor alias.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam dir Derivative direction.
 * @tparam Derived Type of the tensor alias.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @note To obtain an approximation of a derivative, the return value of this function must be multiplied by
 * \f$1/\Delta x^k\f$ where \f$k\f$ is its order.
 * @param i Grid cell index in \f$x\f$-direction.
 * @param j Grid cell index in \f$y\f$-direction.
 * @param k Grid cell index in \f$z\f$-direction.
 * @param tensor Reference to the tensor alias to evaluate stencil on.
 * @param tensor_indices The indices of the tensor component to be differentiated.
 * @return A finite-difference approximation of \f$\Delta x\partial_a\verb|tensor|_{b...}\verb|(i, j, k)|\f$ where
 * \f$a=\verb|dir|\f$ and
 * \f$(b,...) = \verb|tensor_indices|\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires `arr` to provide `ngrow` ghost cells. For implementation
 * details see `treesap::stencil_impl`.
 */
template <int ngrow, int dir, typename AliasType>
    requires(tensor::TensorAliasImplementation<AliasType>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(int i, int j, int k, const AliasType& tensor,
                                                         const tensor::IndexArray<AliasType::rank>& tensor_indices) {

    return stencil_impl::derivative_impl<ngrow, typename AliasType::Base::arr_t, dir>::eval(
        i, j, k, AliasType::Base::get_absolute(tensor_indices), tensor.arr);
}

/**
 * @brief Evaluates a centered second-order derivative stencil on a tensor alias.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam dir0 First derivative direction.
 * @tparam dir1 Second derivative direction.
 * @tparam Derived Type of the tensor alias.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @note To obtain an approximation of a derivative, the return value of this function must be multiplied by
 * \f$1/\Delta x^k\f$ where \f$k\f$ is its order.
 * @param i Grid cell index in \f$x\f$-direction.
 * @param j Grid cell index in \f$y\f$-direction.
 * @param k Grid cell index in \f$z\f$-direction.
 * @param tensor Reference to the tensor alias to evaluate stencil on.
 * @param tensor_indices The indices of the tensor component to be differentiated.
 * @return A finite-difference approximation of \f$\Delta x^2\partial_a\partial_b\verb|tensor|_{c...}\verb|(i, j, k)|\f$
 * where
 * \f$a=\verb|dir0|\f$, \f$b=\verb|dir1|\f$ and
 * \f$(c,...) = \verb|tensor_indices|\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires `arr` to provide `ngrow` ghost cells. For implementation
 * details see `treesap::stencil_impl`.
 */
template <int ngrow, int dir0, int dir1, typename AliasType>
    requires(tensor::TensorAliasImplementation<AliasType>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(int i, int j, int k, const AliasType& tensor,
                                                         const tensor::IndexArray<AliasType::rank>& tensor_indices) {

    return stencil_impl::derivative_impl<ngrow, typename AliasType::Base::arr_t, dir0, dir1>::eval(
        i, j, k, AliasType::Base::get_absolute(tensor_indices), tensor.arr);
}

/**
 * @brief Evaluates a centered third-order derivative stencil on a tensor alias.
 * @tparam ngrow Amount of neighbours in either direction; the stencil width is `2 * ngrow + 1`.
 * @tparam dir0 First derivative direction.
 * @tparam dir1 Second derivative direction.
 * @tparam dir2 Third derivative direction.
 * @tparam Derived Type of the tensor alias.
 * @note The derivative stencils are unrolled at compile time and do not produce runtime call overhead.
 * @note To obtain an approximation of a derivative, the return value of this function must be multiplied by
 * \f$1/\Delta x^k\f$ where \f$k\f$ is its order.
 * @param i Grid cell index in \f$x\f$-direction.
 * @param j Grid cell index in \f$y\f$-direction.
 * @param k Grid cell index in \f$z\f$-direction.
 * @param tensor Reference to the tensor alias to evaluate stencil on.
 * @param tensor_indices The indices of the tensor component to be differentiated.
 * @return A finite-difference approximation of \f$\Delta x^3\partial_a\partial_b\partial_c\verb|tensor|_{d...}\verb|(i,
 * j, k)|\f$ where
 * \f$a=\verb|dir0|\f$, \f$b=\verb|dir1|\f$, \f$c=\verb|dir2|\f$ and
 * \f$(d,...) = \verb|tensor_indices|\f$.
 *
 * The stencil width is `2 * ngrow + 1`, and requires `arr` to provide `ngrow` ghost cells. For implementation
 * details see `treesap::stencil_impl`.
 */
template <int ngrow, int dir0, int dir1, int dir2, typename AliasType>
    requires(tensor::TensorAliasImplementation<AliasType>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(int i, int j, int k, const AliasType& tensor,
                                                         const tensor::IndexArray<AliasType::rank>& tensor_indices) {

    return stencil_impl::derivative_impl<ngrow, typename AliasType::Base::arr_t, dir0, dir1, dir2>::eval(
        i, j, k, AliasType::Base::get_absolute(tensor_indices), tensor.arr);
}

} // namespace stencil
} // namespace treesap