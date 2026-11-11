// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief See @ref example-01-wave-equation for details on this example.
 */

#pragma once

#include "pde-model/PDEModelConcepts.hpp"        // <-- for RHSConcept
#include "pde-model/tensor-alias/RankNAlias.hpp" // <-- for ScalarAlias, VectorAlias
#include "util/tensor_operations.hpp"            // <-- for contract()
#include "util/tensor_stencil.hpp"               // <-- for derivative<>()

template <int ngrow_> struct WaveEqnRHS {
    static constexpr int ncomp = 4;
    static constexpr int ngrow = ngrow_;

    amrex::Real dx_inv;

    AMREX_GPU_HOST_DEVICE explicit WaveEqnRHS(const amrex::GeometryData& geom) : dx_inv(1.0 / geom.CellSize(0)) {}

    template <int comp>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, amrex::Real t,
                                                                    const amrex::Array4<const amrex::Real>& arr) const {
        // import namespaces for more concise code
        using namespace treesap;
        using namespace tensor;
        using namespace stencil;

        // set up the aliases for field variables:
        ScalarAlias<0> phi{{arr}};      // component range {0}
        VectorAlias<phi.next> u{{arr}}; // component range {1,2,3}
        static_assert(u.next == ncomp); // ensure that we have the correct amount of components

        // constexpr if to decide which field comp belongs to
        if constexpr (phi.contains(comp)) { // phi_rhs implementation
            return dx_inv * contract([&]<int a>() { return derivative<ngrow, a>(i, j, k, u, {a}); }); // = ∂ₐuᵃ(i,j,k)

        } else if constexpr (u.contains(comp)) { // u_rhs implementation
            // restore vector index into constexpr int a so we can use it as a template argument
            constexpr auto restored = u.restore_absolute(comp);
            constexpr int a = restored[0];

            return dx_inv * derivative<ngrow, a>(i, j, k, phi, {/*no indices*/}); // = ∂ₐφ(i,j,k)
        } else {
            return NAN; // this should never be reached
        }
    }
};

static_assert(treesap::RHSConcept<WaveEqnRHS<1>>);