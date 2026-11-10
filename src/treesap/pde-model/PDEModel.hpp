// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides the `treesap::ABLPDEModel` class template.
 */

#pragma once

#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"

#include "PDEModelConcepts.hpp"
#include "bcs/NoOpBC.hpp"
#include "init/ConstInit.hpp"
#include "rhs-decorators/NoOpRHS.hpp"

namespace treesap {

/**
 * @brief Hold the components that define a partial differential equation model.
 * @tparam RHSType Right-hand side functor type.
 * @tparam InitType Initial condition functor type.
 * @tparam BCType Boundary condition functor type.
 *
 * The right-hand side (RHS) functor is expected to implement the functional \f$F\f$ of the PDE recast into the
 * first-order-in-time form
 * \f[
 *      \partial_t \Phi = F(\Phi,t),
 * \f]
 * where \f$t\f$ is the time variable and \f$\Phi\f$ the state vector of the evolved fields.
 */
template <typename RHSType, typename InitType, typename BCType = NoOpBC<RHSType::ncomp, RHSType::ngrow>>
    requires RHSConcept<RHSType> && BCConcept<BCType> && InitConcept<InitType>
class PDEModel {
  public: // forward constants and types
    /// Number of field components in \f$\Phi\f$
    static constexpr int ncomp = RHSType::ncomp;

    /// Number of ghost cells.
    static constexpr int ngrow = RHSType::ngrow;

    /// Right-hand side functor type.
    using rhs_t = RHSType;

    /// Initial condition functor type.
    using init_t = InitType;

    /// Boundary condition functor type.
    using bc_t = BCType;

    // check if the template parameters are compatible
    static_assert(InitType::ncomp == ncomp);
    static_assert(BCType::ncomp == ncomp);
    static_assert(BCType::ngrow == ngrow);

  public:
    // generic interface to initialise and retrieve the members of the collection of things constituting a PDEModel

    /**
     * @brief Initialize the model container.
     * @param comp_names Names of the components.
     * @param geom Domain geometry.
     * @param dt Time step width.
     * @param rhs Right-hand side functor object.
     * @param init Initial condition functor object.
     * @param bc Boundary condition functor object.
     */
    PDEModel(amrex::Vector<std::string> comp_names, amrex::Geometry geom, amrex::Real dt, RHSType rhs, InitType init,
             BCType bc = BCType{},
             amrex::Vector<amrex::BCRec> bc_recs = amrex::Vector<amrex::BCRec>(
                 ncomp,
                 amrex::BCRec(AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir),
                              AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir))))
        : comp_names(std::move(comp_names)), geom(std::move(geom)), dt(dt), rhs(std::move(rhs)), init(std::move(init)),
          bc(std::move(bc)), bc_recs(std::move(bc_recs)) {}

    /**
     * @brief Get the right-hand side functor object.
     * @return Reference to the right-hand side functor object.
     */
    const RHSType& get_rhs() const { return rhs; }

    /**
     * @brief Get the initial condition functor object.
     * @return Reference to the initial condition functor object.
     */
    const InitType& get_init() const { return init; }

    /**
     * @brief Get the boundary condition functor object.
     * @return Reference to the boundary condition functor object.
     */
    const BCType& get_bc() const { return bc; }

    /**
     * @brief Get the boundary condition records.
     * @return Reference to the boundary condition records.
     */
    const amrex::Vector<amrex::BCRec>& get_bc_recs() const { return bc_recs; }

    /**
     * @brief Get the domain geometry.
     * @return Reference to the domain geometry.
     */
    const auto& get_geometry() const { return geom; }

    /**
     * @brief Get the component names.
     * @return Reference to the vector of component names.
     */
    const auto& get_comp_names() const { return comp_names; }

    /**
     * @brief Get a hint for the timestep dt.
     * @return The suggested timestep size.
     * @note The reason this is referred to as a hint is that certain integration schemes like Dormand-Prince scale the
     * timestep internally. For the TreESaP implementations of Euler and RK4, the dt_hint is used as the timestep width
     * directly.
     */
    amrex::Real dt_hint() const { return dt; }

  protected:
    RHSType rhs;                         ///< Right-hand side object.
    InitType init;                       ///< Initial condition object.
    BCType bc;                           ///< Boundary condition object.
    amrex::Vector<amrex::BCRec> bc_recs; ///< Boundary condition records.

    amrex::Geometry geom;                  ///< Domain geometry.
    amrex::Vector<std::string> comp_names; ///< Names of the components.

    amrex::Real dt;
};

static_assert(PDEModelConcept<PDEModel<NoOpRHS<1, 1>, ConstInit<1>>>);

} // namespace treesap