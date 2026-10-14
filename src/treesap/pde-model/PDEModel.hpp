#pragma once

#include "PDEModelConcepts.hpp"
#include "bcs/NoOpBC.hpp"
#include "init/ConstInit.hpp"
#include "rhs-decorators/NoOpRHS.hpp"

template <typename RHSType, typename InitType, typename BCType = NoOpBC<RHSType::ncomp, RHSType::ngrow>>
    requires RHSConcept<RHSType> && BCConcept<BCType> && InitConcept<InitType>
class PDEModel {
  public: // forward constants and types
    static constexpr int ncomp = RHSType::ncomp;
    static constexpr int ngrow = RHSType::ngrow;

    using rhs_t = RHSType;
    using init_t = InitType;
    using bc_t = BCType;

    // check if the template parameters are compatible
    static_assert(InitType::ncomp == ncomp);
    static_assert(BCType::ncomp == ncomp);
    static_assert(BCType::ngrow == ngrow);

  public:
    // generic interface to initialise and retrieve the members of the collection of things constituting a PDEModel

    PDEModel(amrex::Geometry geom, RHSType rhs, InitType init, BCType bc = BCType{})
        : geom_(std::move(geom)), rhs(std::move(rhs)), init(std::move(init)), bc(std::move(bc)) {}

    const RHSType& get_rhs() const { return rhs; }
    const InitType& get_init() const { return init; }
    const BCType& get_bc() const { return bc; }

    const amrex::Geometry& geom() const { return geom_; }

  protected:
    RHSType rhs;
    InitType init;
    BCType bc;

    amrex::Geometry geom_;
};

static_assert(PDEModelConcept<PDEModel<NoOpRHS<1, 1>, ConstInit<1>>>);