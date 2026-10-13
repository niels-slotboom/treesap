#pragma once

// empty example implementation of an RHS, used to test decorator implementations

#include "../PDEModelConcepts.hpp"

template <int ncomp_, int ngrow_> struct NoOpRHS {
    static_assert(ncomp_ > 0);
    static_assert(ngrow_ >= 0);

    static constexpr int ncomp = ncomp_;
    static constexpr int ngrow = ngrow_;

    AMREX_GPU_HOST_DEVICE amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                                                 const amrex::Array4<const amrex::Real>& arr) const {
        return 0.0;
    }
};

static_assert(RHSConcept<NoOpRHS<1, 1>>);
