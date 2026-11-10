// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides the `treesap::NoOpBC` class, which implements no-op boundary conditions.
 */

#pragma once
#include "../PDEModelConcepts.hpp"

namespace treesap {

/**
 * @brief Implementation of empty boundary conditions.
 * @tparam ncomp_ Number of components.
 * @tparam ngrow_ Number of ghost cells.
 * @note Satisfies `treesap::BCConcept`.
 *
 * When called, this functor does nothing.
 */
template <int ncomp_, int ngrow_> struct NoOpBC {
    /// Number of components.
    static constexpr int ncomp = ncomp_;

    /// Number of ghost cells.
    static constexpr int ngrow = ngrow_;

    /**
     * @brief Does nothing.
     * @param iv Grid cell index.
     * @param dest Destination array view.
     * @param dcomp Destination component index.
     * @param numcomp Number of components.
     * @param geom Domain geometry data.
     * @param time Simulation time.
     * @param bcr Boundary condition records.
     * @param bcomp Boundary component index.
     * @param orig_comp Original component index.
     */
    AMREX_GPU_HOST_DEVICE
    void operator()(amrex::IntVect const& iv, amrex::Array4<amrex::Real> const& dest, int dcomp, int numcomp,
                    amrex::GeometryData const& geom, amrex::Real time, const amrex::BCRec* bcr, int bcomp,
                    int orig_comp) const {
        // Does nothing. (Duh.)
    }
};

// Verify that NoOpBC satisfies the boundary condition concept.
static_assert(BCConcept<NoOpBC<1, 1>>);

} // namespace treesap