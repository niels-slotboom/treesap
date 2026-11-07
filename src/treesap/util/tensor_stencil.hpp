#pragma once

#include "pde-model/tensor-alias/TensorAliasBase.hpp"
#include "stencil.hpp"

// this file provides forwards/wrappers for stencil.hpp functions that can be used with TensorAliasBase implementations
namespace treesap {
namespace stencil {

// forwards to stencil::derivative (first derivative)
template <int ngrow, int dir, typename Derived>
    requires(TensorAliasImplementation<Derived>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(int i, int j, int k, const Derived& tensor,
                                                         const IndexArray<Derived::rank>& tensor_indices) {

    return derivative_impl<ngrow, typename Derived::Base::arr_t, dir>::eval(
        i, j, k, Derived::Base::get_absolute(tensor_indices), tensor.arr);
}

// forwards to stencil::derivative (second derivative)
template <int ngrow, int dir0, int dir1, typename Derived>
    requires(TensorAliasImplementation<Derived>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(int i, int j, int k, const Derived& tensor,
                                                         const IndexArray<Derived::rank>& tensor_indices) {

    return derivative_impl<ngrow, typename Derived::Base::arr_t, dir0, dir1>::eval(
        i, j, k, Derived::Base::get_absolute(tensor_indices), tensor.arr);
}

// forwards to stencil::derivative (third derivative)
template <int ngrow, int dir0, int dir1, int dir2, typename Derived>
    requires(TensorAliasImplementation<Derived>)
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto derivative(int i, int j, int k, const Derived& tensor,
                                                         const IndexArray<Derived::rank>& tensor_indices) {

    return derivative_impl<ngrow, typename Derived::Base::arr_t, dir0, dir1, dir2>::eval(
        i, j, k, Derived::Base::get_absolute(tensor_indices), tensor.arr);
}

} // namespace stencil
} // namespace treesap