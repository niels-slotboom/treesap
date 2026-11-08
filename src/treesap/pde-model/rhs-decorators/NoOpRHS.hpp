#pragma once
#include "../PDEModelConcepts.hpp"

namespace treesap {

// Empty example implementation of an RHS, used to test decorator implementations;
// Not to be used by a library user.
template <int ncomp_, int ngrow_> struct NoOpRHS {
    static constexpr int ncomp = ncomp_;
    static constexpr int ngrow = ngrow_;

    template <int comp>
    AMREX_GPU_HOST_DEVICE amrex::Real operator()(int i, int j, int k, amrex::Real time,
                                                 const amrex::Array4<const amrex::Real>& arr) const {
        return 0.0;
    }
};

static_assert(RHSConcept<NoOpRHS<1, 1>>);

} // namespace treesap