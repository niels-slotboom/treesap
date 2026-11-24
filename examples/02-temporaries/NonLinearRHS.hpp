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

    amrex::Real dx_inv_sq;
    amrex::Real gamma;

    AMREX_GPU_HOST_DEVICE explicit NonLinearRHS(const amrex::GeometryData& geom, amrex::Real gamma)
        : dx_inv_sq(1.0 / (geom.CellSize(0) * geom.CellSize(0))), gamma(gamma) {}

    // per-thread scratch state for temporaries used across operator() invocations for different values of `comp`.
    mutable PiAliasType::Temporary pi_temp;

    template <int comp>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, amrex::Real t,
                                                                    const amrex::Array4<const amrex::Real>& arr) const {
        // import namespaces for more concise code
        using namespace treesap;
        using namespace tensor;
        using namespace stencil;

        // set up the aliases for field variables:
        PhiAliasType phi{{arr}}; // component range {0}
        PiAliasType pi{{arr}};   // component range {1}

        // prepare scratch state temporaries if comp == 0, i.e. the first operator() evaluation
        if constexpr (comp == 0) {
            pi_temp = pi.load(i, j, k); // load pi value from memory
                                        // just once, so it can be reused for
                                        // the comp = 0 and comp = 1 evaluation.
        }

        // constexpr if to decide which field comp belongs to
        if constexpr (phi.contains(comp)) {       // phi_rhs implementation
            return pi_temp({});                   // = pi(i,j,k) as stored in temporary
        } else if constexpr (pi.contains(comp)) { // pi_rhs implementation
            VectorAlias<>::Temporary phi_grad{};  // temporary to hold grad(phi)
            static_for<3>([&]<int a>() { phi_grad({a}) = derivative<ngrow, a>(i, j, k, phi, {}); });
            amrex::Real grad_sq = contract([&]<int a>() { return phi_grad({a}) * phi_grad({a}); });
            amrex::Real laplacian = contract([&]<int a>() { return derivative<ngrow, a, a>(i, j, k, phi, {}); });
            return dx_inv_sq * (laplacian - gamma * pi_temp({}) * grad_sq);
        } else {
            return NAN; // this should never be reached
        }
    }
};

static_assert(treesap::RHSConcept<NonLinearRHS<1>>);