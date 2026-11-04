#pragma once

// empty example implementation of an RHS, used to test decorator implementations

#include "pde-model/PDEModelConcepts.hpp"
#include "pde-model/tensor-alias/ScalarAlias.hpp"
#include "util/tensor_stencil.hpp"

template <int ngrow_> struct WaveEqnRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = ngrow_;

    enum Component : int { phi = 0, dphi_dt = 1 };

    AMREX_GPU_HOST_DEVICE
    explicit WaveEqnRHS(amrex::Real dx) : inv_dx_sq(1.0 / (dx * dx)) {}

    amrex::Real inv_dx_sq;

    template <int comp>
    AMREX_GPU_HOST_DEVICE amrex::Real operator()(int i, int j, int k, amrex::Real time,
                                                 const amrex::Array4<const amrex::Real>& arr) const {
        using namespace treesap::stencil;

        treesap::ScalarAlias<0> phi{{arr}};
        treesap::ScalarAlias<1> pi{{arr}};

        if constexpr (phi.contains(comp)) {
            return pi(i, j, k);
        } else if constexpr (pi.contains(comp)) {
            return inv_dx_sq * laplacian<ngrow>(i, j, k, phi);
        } else {
            return NAN;
        }
    }
};

static_assert(treesap::RHSConcept<WaveEqnRHS<1>>);