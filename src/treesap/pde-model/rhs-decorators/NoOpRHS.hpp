// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Implements `treesap::NoOpRHS`
 */

#pragma once
#include "../PDEModelConcepts.hpp"

namespace treesap {

/** @brief Empty example implementation of an RHS, used to test decorator implementations;
 * @note Not meant to be used by a library user.
 * @tparam ncomp_ Number of components.
 * @tparam ngrow_ Number of ghost cells.
 * @note Satisfies `treesap::RHSConcept`.
 */
template <int ncomp_, int ngrow_> struct NoOpRHS {
    static constexpr int ncomp = ncomp_;
    static constexpr int ngrow = ngrow_;

    /// Does nothing.
    template <int comp>
    AMREX_GPU_HOST_DEVICE amrex::Real operator()(int i, int j, int k, amrex::Real time,
                                                 const amrex::Array4<const amrex::Real>& arr) const {
        return 0.0;
    }
};

static_assert(RHSConcept<NoOpRHS<1, 1>>);

} // namespace treesap