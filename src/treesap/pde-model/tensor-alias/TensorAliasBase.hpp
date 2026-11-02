#pragma once

#include "AMReX_Array4.H"
#include "AMReX_Extension.H"

namespace treesap {

/**
 * @brief Concept that verifies a derived class implements the required tensor offset and component interface.
 * @tparam Derived The derived class type being validated.
 * @tparam rank_ The tensor rank (order).
 */
template <typename Derived, int rank_>
concept TensorAliasImplementation =
    []<int... Is>(std::integer_sequence<int, Is...>) {
        return requires(const Derived d) {
            { d.get_offset(Is...) } -> std::same_as<int>;
        };
    }(std::make_integer_sequence<int, rank_>{}) &&
    requires {
        typename std::integral_constant<int, Derived::ncomp>;
        requires Derived::ncomp > 0;
        typename std::integral_constant<int, Derived::rank>;
        requires Derived::ncomp >= 0;
        typename std::integral_constant<int, Derived::dim>;
        requires Derived::ncomp > 0;
    } && requires(const Derived d, int ncomp) {
        { d.restore_relative(ncomp) } -> std::same_as<amrex::GpuArray<int, rank_>>;
    };

/**
 * @brief Base class for mapping multi-component tensors stored in an AMReX 4D array.
 * @tparam Derived The CRTP derived class implementing index offset logic.
 * @tparam rank_ The rank of the tensor (supports rank 0 for scalars).
 * @tparam ArrayType The data type stored in the array.
 * @tparam dim_ The spatial dimension count.
 */
template <typename Derived, typename ArrayType> struct TensorAliasBase {
    static constexpr int ncomp = Derived::ncomp;
    static constexpr int rank = Derived::rank;
    static constexpr int dim = Derived::dim;

    /**
     * @brief Computes the absolute component index for given tensor indices.
     * @tparam TensorIndices Variadic integer types representing tensor indices.
     * @param tensor_indices The tensor index values (empty for rank 0 scalars).
     * @return The absolute component index in the underlying array.
     */
    template <typename... TensorIndices>
        requires(sizeof...(TensorIndices) == rank && (std::is_same_v<int, TensorIndices> && ...))
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int constexpr get_comp(TensorIndices... tensor_indices) const {
        Derived* impl = static_cast<Derived*>(this);
        if constexpr (rank == 0) {
            return comp_base + impl->get_offset();
        } else {
            return comp_base + impl->get_offset(tensor_indices...);
        }
    }

    /**
     * @brief Accesses a specific array element using spatial and tensor indices.
     * @param i The grid index in the x direction.
     * @param j The grid index in the y direction.
     * @param k The grid index in the z direction.
     * @param tensor_indices The tensor index values (omitted for rank 0 scalars).
     * @return Reference to the array element.
     */
    template <typename... TensorIndices>
        requires(sizeof...(TensorIndices) == rank && (std::is_same_v<int, TensorIndices> && ...))
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE ArrayType& operator()(int i, int j, int k,
                                                                   TensorIndices... tensor_indices) const {
        return arr(i, j, k, get_comp(tensor_indices...));
    }

    /**
     * @brief Checks if a component index falls within the valid range for this tensor alias.
     * @param comp The absolute component index to check.
     * @return True if the component is within range, false otherwise.
     */
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE bool constexpr contains(int absolute_comp) const {
        int comp_rel_to_base = absolute_comp - comp_base;
        return comp_rel_to_base >= 0 && comp_rel_to_base < Derived::ncomp;
    }

    AMREX_FORCE_INLINE
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<int, rank> constexpr restore_absolute(int absolute_comp) const {
        Derived* impl = static_cast<Derived*>(this);
        return impl->restore_relative(absolute_comp - comp_base);
    }

    /// Reference to the underlying 4D AMReX array data structure.
    const amrex::Array4<ArrayType>& arr;

    /// Base component index where the contiguous component range starts.
    const int comp_base = 0;
};

} // namespace treesap