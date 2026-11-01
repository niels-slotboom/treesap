#pragma once

#include "AMReX_Array4.H"
#include "AMReX_Extension.H"
#include "util/cexprmath.hpp"

namespace treesap {

/**
 * @brief Concept enforcing that a CRTP derived class implements a compatible get_offset method.
 *
 * Verifies that the derived class provides a member function `get_offset` which accepts
 * a number of integer arguments equal to @p rank_ and returns an `int`.
 *
 * @tparam Derived The CRTP derived class type being checked.
 * @tparam rank_ The expected rank (number of dimensions/indices) of the tensor.
 */
template <typename Derived, int rank_>
concept TensorAliasImplementation = []<int... Is>(std::integer_sequence<int, Is...>) {
    return requires(Derived d) {
        { d.get_offset(Is...) } -> std::same_as<int>;
    };
}(std::make_integer_sequence<int, rank_>{});

/**
 * @brief CRTP base class template providing multidimensional tensor indexing over AMReX arrays.
 *
 * `TensorAlias` manages spatial grid indexing (`i, j, k`) combined with a component offset
 * calculated via a CRTP-derived implementation. It maps tensor components to a contiguous
 * range within an underlying `amrex::Array4`.
 *
 * @tparam Derived The CRTP derived class type. Must satisfy `TensorAliasImplementation`.
 * @tparam rank_ The tensor rank (order).
 * @tparam ArrayType The underlying data type stored in the array (defaults to `amrex::Real`).
 * @tparam dim_ The spatial or component dimension size (defaults to 3).
 */
template <typename Derived, int rank_, typename ArrayType = amrex::Real, int dim_ = 3>
    requires(TensorAliasImplementation<Derived, rank_>)
struct TensorAliasBase {
    /// The rank (order) of the tensor.
    static constexpr int rank = rank_;

    /// The dimension size of each tensor index axis.
    static constexpr int dim = dim_;

    /// Total number of tensor components (`dim^rank`).
    static constexpr int ncomp = cexprmath::power(dim, rank);

    /**
     * @brief Accesses a specific tensor component at a given spatial grid location.
     *
     * @tparam TensorIndices Variadic pack of integer tensor index types.
     * @param i Spatial index in the x-direction.
     * @param j Spatial index in the y-direction.
     * @param k Spatial index in the z-direction.
     * @param tensor_indices The individual tensor component indices (must match @p rank in count).
     * @return amrex::Real& Reference to the targeted element in the array.
     */
    template <typename... TensorIndices>
        requires(sizeof...(TensorIndices) == rank && (std::is_same_v<int, TensorIndices> && ...))
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::Real& operator()(int i, int j, int k,
                                                                     TensorIndices... tensor_indices) {
        Derived* impl = static_cast<Derived*>(this);
        return arr(i, j, k, comp_base + impl->get_offset(tensor_indices...));
    }

    /// Reference to the underlying 4D AMReX array data structure.
    const amrex::Array4<ArrayType>& arr;

    /// Base component index where the contiguous range of @p ncomp components starts.
    const int comp_base = 0;
};

} // namespace treesap