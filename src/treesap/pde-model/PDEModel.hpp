#pragma once

#include "AMReX_BCRec.H"
#include "AMReX_BC_TYPES.H"

#include "PDEModelConcepts.hpp"
#include "bcs/NoOpBC.hpp"
#include "init/ConstInit.hpp"
#include "rhs-decorators/NoOpRHS.hpp"

namespace tsap {

/**
 * @brief Hold the components that define a partial differential equation model.
 * @tparam RHSType Right-hand side model type.
 * @tparam InitType Initial condition model type.
 * @tparam BCType Boundary condition model type.
 */
template <typename RHSType, typename InitType, typename BCType = NoOpBC<RHSType::ncomp, RHSType::ngrow>>
    requires RHSConcept<RHSType> && BCConcept<BCType> && InitConcept<InitType>
class PDEModel {
  public: // forward constants and types
    /// Number of components.
    static constexpr int ncomp = RHSType::ncomp;

    /// Number of ghost cells.
    static constexpr int ngrow = RHSType::ngrow;

    /// Right-hand side model type.
    using rhs_t = RHSType;

    /// Initial condition model type.
    using init_t = InitType;

    /// Boundary condition model type.
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
     * @param rhs Right-hand side object.
     * @param init Initial condition object.
     * @param bc Boundary condition object.
     */
    PDEModel(amrex::Vector<std::string> comp_names, amrex::Geometry geom, RHSType rhs, InitType init,
             BCType bc = BCType{},
             amrex::Vector<amrex::BCRec> bc_recs = amrex::Vector<amrex::BCRec>(
                 ncomp,
                 amrex::BCRec(AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir),
                              AMREX_D_DECL(amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir))))
        : comp_names(std::move(comp_names)), geom(std::move(geom)), rhs(std::move(rhs)), init(std::move(init)),
          bc(std::move(bc)), bc_recs(std::move(bc_recs)) {}

    /**
     * @brief Get the right-hand side object.
     * @return Reference to the right-hand side object.
     */
    const RHSType& get_rhs() const { return rhs; }

    /**
     * @brief Get the initial condition object.
     * @return Reference to the initial condition object.
     */
    const InitType& get_init() const { return init; }

    /**
     * @brief Get the boundary condition object.
     * @return Reference to the boundary condition object.
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

  protected:
    RHSType rhs;                         ///< Right-hand side object.
    InitType init;                       ///< Initial condition object.
    BCType bc;                           ///< Boundary condition object.
    amrex::Vector<amrex::BCRec> bc_recs; ///< Boundary condition records.

    amrex::Geometry geom;                  ///< Domain geometry.
    amrex::Vector<std::string> comp_names; ///< Names of the components.
};

static_assert(PDEModelConcept<PDEModel<NoOpRHS<1, 1>, ConstInit<1>>>);

} // namespace tsap