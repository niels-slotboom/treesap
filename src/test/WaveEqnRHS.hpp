#pragma once

// empty example implementation of an RHS, used to test decorator implementations

#include "pde-model/PDEModelConcepts.hpp"
#include "util/stencil.hpp"

template <int ngrow_> struct WaveEqnRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = ngrow_;

    enum Component : int { phi = 0, dphi_dt = 1 };

    AMREX_GPU_HOST_DEVICE
    explicit WaveEqnRHS(amrex::Real dx) : inv_dx_sq(1.0 / (dx * dx)) {}

    amrex::Real inv_dx_sq;

    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                           const amrex::Array4<const amrex::Real>& arr) const {
        switch (comp) {
        case Component::phi:
            return arr(i, j, k, Component::dphi_dt);

        case Component::dphi_dt: {
            using namespace tsap::stencil;
            return inv_dx_sq * laplacian<ngrow>(i, j, k, Component::phi, arr);
        }

        default:
            return NAN;
        }
    }
};

static_assert(tsap::RHSConcept<WaveEqnRHS<1>>);