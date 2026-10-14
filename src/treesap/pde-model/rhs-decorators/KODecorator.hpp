#pragma once

#include "../../util/stencil.hpp"
#include "../PDEModelConcepts.hpp"
#include "NoOpRHS.hpp"

/**
 * @brief Adds Kreiss-Oliger dissipation to a right-hand side model.
 * @tparam RHSType Underlying right-hand side model type.
 */
template <typename RHSType>
    requires RHSConcept<RHSType>
struct KODecorator {
    /// Number of components.
    static constexpr int ncomp = RHSType::ncomp;

    /// Number of ghost cells.
    static constexpr int ngrow = RHSType::ngrow;

    /**
     * @brief Evaluates the decorated right-hand side with added dissipation.
     * @param i Grid cell index in X direction.
     * @param j Grid cell index in Y direction.
     * @param k Grid cell index in Z direction.
     * @param comp Component index.
     * @param time Current simulation time.
     * @param arr Data array view buffer.
     * @return Combined right-hand side value with dissipation.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                                                                    const amrex::Array4<const amrex::Real>& arr) const {
        using namespace stencil;

        amrex::Real ko_term = 0.0;
        ko_term += kreiss_oliger<ngrow, dir::x>(i, j, k, comp, arr);
        ko_term += kreiss_oliger<ngrow, dir::y>(i, j, k, comp, arr);
        ko_term += kreiss_oliger<ngrow, dir::z>(i, j, k, comp, arr);

        return rhs(i, j, k, comp, time, arr) + dx_inv * strength * ko_term;
    }

    RHSType rhs;          ///< Original right-hand side model to be decorated.
    amrex::Real strength; ///< Dissipation strength factor.
    amrex::Real dx_inv;   ///< Inverse grid spacing (1 / dx).
};

// Verify that KODecorator satisfies the right-hand side concept.
static_assert(RHSConcept<KODecorator<NoOpRHS<1, 1>>>);