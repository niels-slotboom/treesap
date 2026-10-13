#pragma once
#include "../PDEModelConcepts.hpp"

/**
 * @brief Implements (constant) Dirichlet boundary conditions.
 * @tparam ncomp_ Number of components.
 * @tparam ngrow_ Number of ghost cells.
 */
template <int ncomp_, int ngrow_> struct DirichletBC {
    /// Number of components.
    static constexpr int ncomp = ncomp_;

    /// Number of ghost cells.
    static constexpr int ngrow = ngrow_;

    /// Fixed boundary values for each component.
    amrex::GpuArray<amrex::Real, ncomp> boundary_values{};

    /**
     * @brief Apply Dirichlet boundary conditions to ghost cells.
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
        const amrex::Box& domain = geom.Domain();

        // Check if cell is outside domain.
        bool is_outside = false;
        for (int dir = 0; dir < 3; ++dir) {
            if (iv[dir] < domain.smallEnd(dir) || iv[dir] > domain.bigEnd(dir)) {
                is_outside = true;
                break;
            }
        }

        if (!is_outside)
            return;

        // Set boundary values.
        for (int comp = 0; comp < ncomp; ++comp) {
            dest(iv, comp) = boundary_values[comp];
        }
    }
};

// Verify that DirichletBC satisfies the boundary condition concept.
static_assert(BCConcept<DirichletBC<1, 1>>);