#pragma once
#include "../pde-model/PDEModelConcepts.hpp"

namespace treesap {

/**
 * @brief Sets all components to a constant value.
 * @tparam ncomp_ Number of components to initialize.
 */
template <int ncomp_> struct ConstInit {
    static_assert(ncomp_ > 0, "ConstInit must initialise a positive number of components.");

    /// Number of components.
    static constexpr int ncomp = ncomp_;

    /// Constant value for initialization.
    amrex::Real value;

    /**
     * @brief Returns the constant value.
     * @param x X coordinate.
     * @param y Y coordinate.
     * @param z Z coordinate.
     * @param comp Component index.
     * @return Constant value.
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(amrex::Real x, amrex::Real y, amrex::Real z, int comp) const { return value; }
};

// Verify that ConstInit satisfies the initialization concept for a test case.
static_assert(InitConcept<ConstInit<1>>);

} // namespace treesap