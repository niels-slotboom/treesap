#pragma once

#include "TensorAliasBase.hpp"

namespace treesap {

template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
struct VectorAlias
    : public TensorAliasBase<VectorAlias<comp_base_, ArrayType, dim_>, comp_base_, dim_, 1, dim_, ArrayType> {
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int constexpr get_offset(int a) const { return a; }

    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::GpuArray<int, 1> constexpr restore_relative(int rel_comp) const {
        return {rel_comp};
    }
};

static_assert(TensorAliasImplementation<VectorAlias<0>>);

} // namespace treesap