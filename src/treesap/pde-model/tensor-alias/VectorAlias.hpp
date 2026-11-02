#pragma once

#include "TensorAliasBase.hpp"

namespace treesap {

template <typename ArrayType = const amrex::Real, int dim_ = 3>
struct VectorAlias : public TensorAliasBase<VectorAlias<ArrayType, dim_>, ArrayType> {
    static constexpr int ncomp = dim_;
    static constexpr int rank = 1;
    static constexpr int dim = dim_;

    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int constexpr get_offset(int a) const { return a; }

    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::GpuArray<int, 1> constexpr restore_relative(int rel_comp) const {
        return {rel_comp};
    }
};

static_assert(TensorAliasImplementation<VectorAlias<>>);

} // namespace treesap