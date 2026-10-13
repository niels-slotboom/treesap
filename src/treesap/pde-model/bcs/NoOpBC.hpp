#pragma once
#include "../PDEModelConcepts.hpp"

/**
 * @brief Implements an empty boundary condition.
 * @tparam ncomp_ Number of components.
 * @tparam ngrow_ Number of ghost cells.
 */
template <int ncomp_, int ngrow_> struct NoOpBC {
    /// Number of components.
    static constexpr int ncomp = ncomp_;

    /// Number of ghost cells.
    static constexpr int ngrow = ngrow_;

    /**
     * @brief Applies the boundary condition.
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