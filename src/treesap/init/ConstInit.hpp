// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Implements the `treesap::ConstInit` class for initialising to a constant value.
 */

#pragma once
#include "../pde-model/PDEModelConcepts.hpp"

namespace treesap {

/**
 * @brief Initialisation functor which sets all components to a constant value.
 * @tparam ncomp_ Number of components to initialise.
 * @note Satisfies `treesap::InitConcept`.
 */
template <int ncomp_> struct ConstInit {
    static_assert(ncomp_ > 0, "ConstInit must initialise a positive number of components.");

    /// Number of components to initialise.
    static constexpr int ncomp = ncomp_;

    /// Constant value for initialization.
    amrex::Real value;

    /**
     * @brief Returns the constant `value`.
     * @param x \f$x\f$-coordinate.
     * @param y \f$y\f$-coordinate.
     * @param z \f$z\f$-coordinate.
     * @param comp Component index in the MultiFab component range.
     * @return Constant `value`.
     */
    template <int comp>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(amrex::Real x, amrex::Real y, amrex::Real z) const {
        return value;
    }
};

// Verify that ConstInit satisfies the initialization concept for a test case.
static_assert(InitConcept<ConstInit<1>>);

} // namespace treesap