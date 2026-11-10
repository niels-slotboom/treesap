// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides the `treesap::ABLPDEModel` class template.
 */

#pragma once

#include "PDEModel.hpp"
#include "bcs/DirichletBC.hpp"
#include "init/ConstInit.hpp"
#include "pde-model/rhs-decorators/NoOpRHS.hpp"
#include "rhs-decorators/ABLDecorator.hpp"

namespace treesap {

/**
 * @brief Defines a PDE model with an absorbing boundary layer (ABL) decorator.
 * @tparam RHSType Undecorated right-hand side functor type.
 * @tparam InitType Initial condition functor type.
 * @tparam ABLProfileType Absorbing boundary layer profile type.
 * @note The `RHSType` template parameter should be *undecorated*; this class template handles the ABL decoration
 * internally.
 *
 * For ABL implementation details, see `treesap::ABLDecorator`.
 */
template <typename RHSType, typename InitType, typename ABLProfileType>
class ABLPDEModel
    : public PDEModel<ABLDecorator<RHSType, ABLProfileType>, InitType, DirichletBC<RHSType::ncomp, RHSType::ngrow>> {

    using Base = PDEModel<ABLDecorator<RHSType, ABLProfileType>, InitType, DirichletBC<RHSType::ncomp, RHSType::ngrow>>;
    using DecoratedRHSType = ABLDecorator<RHSType, ABLProfileType>;
    using BCType = DirichletBC<RHSType::ncomp, RHSType::ngrow>;

  public:
    /**
     * @brief Initialize the absorbing boundary layer model.
     * @param comp_names Names of the components.
     * @param geom Domain geometry.
     * @param dt Time step width.
     * @param rhs Right-hand side model object.
     * @param abl_data Absorbing boundary layer configuration data.
     * @param init Initial condition object.
     */
    ABLPDEModel(amrex::Vector<std::string> comp_names, amrex::Geometry geom, amrex::Real dt, RHSType rhs,
                typename DecoratedRHSType::ABLData abl_data, InitType init)
        : Base(std::move(comp_names), geom, dt, DecoratedRHSType{std::move(rhs), geom, abl_data}, std::move(init),
               BCType{abl_data.target_values}) {}
};

static_assert(
    PDEModelConcept<ABLPDEModel<NoOpRHS<1, 1>, ConstInit<1>, decltype([](amrex::Real x) constexpr { return x; })>>);

} // namespace treesap