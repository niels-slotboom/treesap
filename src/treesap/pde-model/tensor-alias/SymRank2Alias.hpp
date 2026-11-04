#pragma once

#include "TensorAliasBase.hpp"

namespace treesap {

template <int comp_base_, typename ArrayType = const amrex::Real, int dim_ = 3>
struct SymRank2Alias : public TensorAliasBase<SymRank2Alias<comp_base_, ArrayType, dim_>, comp_base_,
                                              (dim_ * (dim_ + 1)) / 2, 2, dim_, ArrayType> {

    // example:
    // 0 1 3
    // 1 2 4
    // 3 4 5
    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE int constexpr get_offset(int a, int b) const {
        int row = (a > b) ? a : b;
        int col = (a > b) ? b : a;
        return (row * (row + 1)) / 2 + col;
    }

    AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE amrex::GpuArray<int, 2> constexpr restore_relative(int rel_comp) const {
        int row = 0;
        while ((row + 1) * (row + 2) / 2 <= rel_comp) {
            ++row;
        }
        int col = rel_comp - (row * (row + 1)) / 2;
        return {row, col};
    }
};

static_assert(TensorAliasImplementation<SymRank2Alias<0>>);

} // namespace treesap