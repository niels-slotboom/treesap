#pragma once

#include "AMReX_Array4.H"
#include "AMReX_Extension.H"

namespace treesap {

/**
 * @brief Concept that verifies a derived class implements the required tensor offset and component interface.
 * @tparam Derived The derived class type being validated.
 * @tparam rank_ The tensor rank (order).
 */
template <typename Derived>
concept TensorAliasImplementation =
    requires {
        typename std::integral_constant<int, Derived::ncomp>;
        requires Derived::ncomp > 0;
        typename std::integral_constant<int, Derived::rank>;
        requires Derived::rank >= 0;
        typename std::integral_constant<int, Derived::dim>;
        requires Derived::dim > 0;
    } &&
    []<int... Is>(std::integer_sequence<int, Is...>) {
        return requires(const Derived d) {
            { d.get_offset(Is...) } -> std::same_as<int>;
        };
    }(std::make_integer_sequence<int, Derived::rank>{}) &&
    requires(const Derived d, int ncomp) {
        { d.restore_relative(ncomp) } -> std::same_as<amrex::GpuArray<int, Derived::rank>>;
    };

template <typename Derived, int comp_base_, int ncomp_, int rank_, int dim_, typename ArrayType>
struct TensorAliasBase {
    static constexpr int comp_base = comp_base_; /// Base component index where the contiguous component range starts.
    static constexpr int ncomp = ncomp_;
    static constexpr int rank = rank_;
    static constexpr int dim = dim_;
    static constexpr int next =
        comp_base + ncomp_; /// Base component of the next tensor in the 1D AMReX multifab comp list

    /**
     * @brief Computes the absolute component index for given tensor indices.
     * @tparam TensorIndices Variadic integer types representing tensor indices.
     * @param tensor_indices The tensor index values (empty for rank 0 scalars).
     * @return The absolute component index in the underlying array.
     */
    template <typename... TensorIndices>
        requires(sizeof...(TensorIndices) == rank && (std::is_same_v<int, TensorIndices> && ...))
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int constexpr get_comp(TensorIndices... tensor_indices) const {
        const Derived* impl = static_cast<const Derived*>(this);
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
};

} // namespace treesap