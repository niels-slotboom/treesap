#pragma once

#include "TensorAliasBase.hpp"

namespace treesap {

template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
struct ScalarAlias
    : public TensorAliasBase<ScalarAlias<comp_base_, ArrayType, dim_>, comp_base_, 1, 0, dim_, ArrayType> {

    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int constexpr get_offset() const { return 0; }

    AMREX_FORCE_INLINE
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<int, 0> constexpr restore_relative(int /*rel_comp*/) const { return {}; }
};

static_assert(TensorAliasImplementation<ScalarAlias<0>>);

} // namespace treesap