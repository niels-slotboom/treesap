// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief See @ref example-02-temporaries for details on this example.
 */

#pragma once

#include "pde-model/PDEModelConcepts.hpp"        // <-- for RHSConcept
#include "pde-model/tensor-alias/RankNAlias.hpp" // <-- for ScalarAlias, VectorAlias
#include "util/tensor_operations.hpp"            // <-- for contract()
#include "util/tensor_stencil.hpp"               // <-- for derivative<>()

template <int ngrow_> struct NonLinearRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = ngrow_;

    // alias types must be defined outside of the operator() scope to be available for temporary definitions
    using PhiAliasType = treesap::tensor::ScalarAlias<0>;
    using PiAliasType = treesap::tensor::ScalarAlias<PhiAliasType::next>;
    static_assert(PiAliasType::next == ncomp); // ensure that we have the correct number of components

    amrex::Real gamma;

    AMREX_GPU_HOST_DEVICE explicit NonLinearRHS(amrex::Real gamma) : gamma(gamma) {}

    /**
     * @brief Per-grid-point scratch state for temporaries shared across component evaluations.
     *
     * Instances are managed by the integrator and passed to operator() by reference.
     */
    struct Workspace {
        PiAliasType::Temporary pi;
    };

    template <int comp>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    operator()(const amrex::IntVect idx, amrex::Real t, const amrex::Array4<const amrex::Real>& arr,
               const amrex::GeometryData& geom, Workspace& workspace) const {
        // import namespaces for more concise code
        using namespace treesap;
        using namespace tensor;
        using namespace stencil;

        // set up the aliases for field variables:
        PhiAliasType phi{{arr}}; // component range {0}
        PiAliasType pi{{arr}};   // component range {1}

        // Initialise the workspace temporary during the first component evaluation.
        if constexpr (comp == 0) {
            workspace.pi = pi.load(idx); // load pi from the underlying grid data once,
                                         // for reuse in subsequent component evaluations.
        }

        // constexpr if to decide which field comp belongs to
        if constexpr (phi.contains(comp)) {       // phi_rhs implementation
            return workspace.pi({});              // = pi(idx) as stored in temporary
        } else if constexpr (pi.contains(comp)) { // pi_rhs implementation
            VectorAlias<>::Temporary phi_grad{};  // temporary to hold grad(phi)
            static_for<3>([&]<int a>() { phi_grad({a}) = derivative<ngrow, a>(phi, {}, idx, geom); });
            amrex::Real grad_sq = contract([&]<int a>() { return phi_grad({a}) * phi_grad({a}); });
            amrex::Real laplacian = contract([&]<int a>() { return derivative<ngrow, a, a>(phi, {}, idx, geom); });

            return (laplacian - gamma * workspace.pi({}) * grad_sq);
        } else {
            return NAN; // this should never be reached
        }
    }
};

static_assert(treesap::RHSConcept<NonLinearRHS<1>>);