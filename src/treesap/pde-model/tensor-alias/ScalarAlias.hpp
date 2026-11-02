#pragma once

#include "TensorAliasBase.hpp"

namespace treesap {

template <typename ArrayType = const amrex::Real, int dim_ = 3>
struct ScalarAlias : public TensorAliasBase<ScalarAlias<ArrayType, dim_>, 0, ArrayType, dim_> {
    static constexpr int ncomp = 1;

    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int get_offset() const { return 0; }

    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::GpuArray<int, 0> restore_relative(int /*rel_comp*/) const {
        return {};
    }
};

static_assert(TensorAliasImplementation<ScalarAlias<>, 0>);

} // namespace treesap