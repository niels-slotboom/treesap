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

// --- GRADIENT FORWARD ---
template <int ngrow, typename Derived, typename ArrayType, typename... TensorIndices>
    requires(TensorAliasImplementation<Derived> && sizeof...(TensorIndices) == Derived::rank &&
             (std::is_same_v<int, TensorIndices> && ...))
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::GpuArray<ArrayType, 3>
gradient(int i, int j, int k, const TensorAliasBase<Derived, ArrayType>& tensor, TensorIndices... tensor_indices) {
    return gradient<ngrow, ArrayType>(i, j, k, tensor.get_comp(tensor_indices...), tensor.arr);
}

// --- GRADIENT SQUARED FORWARD ---
template <int ngrow, typename Derived, typename ArrayType, typename... TensorIndices>
    requires(TensorAliasImplementation<Derived> && sizeof...(TensorIndices) == Derived::rank &&
             (std::is_same_v<int, TensorIndices> && ...))
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType gradient_squared(int i, int j, int k,
                                                                    const TensorAliasBase<Derived, ArrayType>& tensor,
                                                                    TensorIndices... tensor_indices) {
    return gradient_squared<ngrow, ArrayType>(i, j, k, tensor.get_comp(tensor_indices...), tensor.arr);
}

// --- LAPLACIAN FORWARD ---
template <int ngrow, typename Derived, typename ArrayType, typename... TensorIndices>
    requires(TensorAliasImplementation<Derived> && sizeof...(TensorIndices) == Derived::rank &&
             (std::is_same_v<int, TensorIndices> && ...))
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType laplacian(int i, int j, int k,
                                                             const TensorAliasBase<Derived, ArrayType>& tensor,
                                                             TensorIndices... tensor_indices) {
    return laplacian<ngrow, ArrayType>(i, j, k, tensor.get_comp(tensor_indices...), tensor.arr);
}

// --- HESSIAN FORWARD ---
template <int ngrow, typename Derived, typename ArrayType, typename... TensorIndices>
    requires(TensorAliasImplementation<Derived> && sizeof...(TensorIndices) == Derived::rank &&
             (std::is_same_v<int, TensorIndices> && ...))
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Array2D<ArrayType, 0, 2, 0, 2>
hessian(int i, int j, int k, const TensorAliasBase<Derived, ArrayType>& tensor, TensorIndices... tensor_indices) {
    return hessian<ngrow, ArrayType>(i, j, k, tensor.get_comp(tensor_indices...), tensor.arr);
}

// --- KREISS-OLIGER DISSIPATION FORWARD ---
template <int ngrow, int dir, typename Derived, typename ArrayType, typename... TensorIndices>
    requires(TensorAliasImplementation<Derived> && sizeof...(TensorIndices) == Derived::rank &&
             (std::is_same_v<int, TensorIndices> && ...))
AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType kreiss_oliger(int i, int j, int k,
                                                                 const TensorAliasBase<Derived, ArrayType>& tensor,
                                                                 TensorIndices... tensor_indices) {
    return kreiss_oliger<ngrow, dir, ArrayType>(i, j, k, tensor.get_comp(tensor_indices...), tensor.arr);
}

} // namespace stencil
} // namespace treesap