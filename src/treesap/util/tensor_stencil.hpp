#pragma once

#include "pde-model/tensor-alias/TensorAliasBase.hpp"
#include "stencil.hpp"

// this file provides forwards/wrappers for stencil.hpp functions that can be used with TensorAliasBase implementations
namespace treesap {
namespace stencil {

// forwards stencil::derivative
template <int ngrow, typename Derived, typename ArrayType, int... Dirs, typename... TensorIndices>
    requires(TensorAliasImplementation<Derived> && sizeof...(TensorIndices) == Derived::rank &&
             (std::is_same_v<int, TensorIndices> && ...))
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType derivative(int i, int j, int k,
                                                              const TensorAliasBase<Derived, ArrayType>& tensor,
                                                              TensorIndices... tensor_indices) {
    return derivative<ngrow, ArrayType, Dirs...>(i, j, k, tensor.get_comp(tensor_indices...), tensor.arr);
}

} // namespace stencil
} // namespace treesap