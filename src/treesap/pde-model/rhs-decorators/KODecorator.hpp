// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Implements the Kreiss-Oliger decorator `treesap::KODecorator`.
 */

#pragma once

#include "../../util/stencil.hpp"
#include "../PDEModelConcepts.hpp"
#include "NoOpRHS.hpp"

namespace treesap {

/**
 * @brief Adds Kreiss-Oliger dissipation to a right-hand side model through a decorator pattern.
 * @tparam RHSType Underlying right-hand side model type.
 * @tparam order_ The order of the Kreiss-Oliger terms; Defaults to `RHSType::ngrow`.
 * @note Satisfies `treesap::RHSConcept`.
 *
 * Concretely, given a dissipation strength \f$\kappa>0\f$, the decoration wraps the original RHS \f$F = \verb|rhs|\f$
 * as
 * \f[
 *      F(\Phi,t) \rightarrow F(\Phi,t) + \kappa (\mathcal{D}_\text{KO}^x \Phi + \mathcal{D}_\text{KO}^y \Phi +
 * \mathcal{D}_\text{KO}^z \Phi)
 * \f]
 * where the Kreiss-Oliger operators \f$\mathcal{D}_\text{KO}\f$ are implemented by `treesap::stencil::kreiss_oliger`.
 */
template <typename RHSType, int order_ = RHSType::ngrow>
    requires RHSConcept<RHSType>
struct KODecorator {
    /// Number of components.
    static constexpr int ncomp = RHSType::ncomp;

    /// Number of ghost cells.
    static constexpr int ngrow = RHSType::ngrow;

    /// Order of the Kreiss-Oliger operator
    static constexpr int order = order_;
    static_assert(order <= ngrow);

    /**
     * @brief Evaluate the decorated right-hand side with added dissipation.
     * @param i Grid cell index in \f$x\f$-direction.
     * @param j Grid cell index in \f$y\f$-direction.
     * @param k Grid cell index in \f$z\f$-direction.
     * @param comp Component index in the MultiFab component range.
     * @param time Current simulation time.
     * @param arr Data array view buffer.
     * @return Combined right-hand side value with dissipation.
     */
    template <int comp>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, amrex::Real time,
                                                                    const amrex::Array4<const amrex::Real>& arr) const {
        using namespace stencil;

        amrex::Real ko_term = 0.0;
        ko_term += kreiss_oliger<order, dir::x>(i, j, k, comp, arr);
        ko_term += kreiss_oliger<order, dir::y>(i, j, k, comp, arr);
        ko_term += kreiss_oliger<order, dir::z>(i, j, k, comp, arr);

        return rhs.template operator()<comp>(i, j, k, time, arr) + dx_inv * strength * ko_term;
    }

    RHSType rhs;          ///< Undecorated right-hand side functor.
    amrex::Real strength; ///< Dissipation strength factor \f$\kappa\f$.
    amrex::Real dx_inv;   ///< Inverse grid spacing \f$1 / \Delta x\f$.
};

// Verify that KODecorator satisfies the right-hand side concept.
static_assert(RHSConcept<KODecorator<NoOpRHS<1, 1>>>);

} // namespace treesap