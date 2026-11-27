// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom
/**
 * @file
 * @brief Provides the C++ concepts which define the PDE model interface used by the time integration abstraction.
 */

#pragma once

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>

#include <concepts>
#include <type_traits>

namespace treesap {

/**
 * @brief Verifies that a type represents a spatial right-hand side operator.
 *
 * An RHS operator defines the time evolution of the simulation variables by
 * providing one component-wise evaluation function for each evolved field
 * component. Each evaluation computes the corresponding time derivative at
 * a given grid point, using the current simulation state and the associated
 * grid geometry.
 *
 * The concept additionally requires the RHS to expose compile-time metadata
 * specifying the number of evolved components and the number of ghost cells
 * required by its spatial operators. It must also provide a default-
 * constructible `Workspace` type, which the integrator instantiates once per
 * grid point and passes by reference to every component evaluation at that
 * point. This workspace provides storage for intermediate quantities that
 * need to persist across component evaluations without requiring additional
 * global memory.
 *
 * RHS implementations must be trivially copyable so that they can be passed
 * into GPU kernels, and their component-wise evaluation functions must be
 * callable on a const instance.
 *
 * @tparam F Type to check against the concept.
 */
template <typename F>
concept RHSConcept = std::is_trivially_copyable_v<F> && requires {
    typename std::integral_constant<int, F::ncomp>;
    typename std::integral_constant<int, F::ngrow>;
    requires F::ncomp > 0;
    requires F::ngrow >= 0;
    typename F::Workspace;
} && std::is_default_constructible_v<typename F::Workspace> && []<std::size_t... Is>(std::index_sequence<Is...>) {
    // Check that every component-specific RHS instantiation is valid.
    return (requires(const F & f,
                     const amrex::IntVect idx,                     ///< Grid index to evaluate RHS at
                     amrex::Real time,                             ///< Current simulation time
                     const amrex::Array4<const amrex::Real> & arr, ///< View into source FAB data buffer
                     const amrex::GeometryData & geom,             ///< GPU-safe domain geometry data
                     typename F::Workspace & workspace             ///< Workspace to place temporaries in
            ) {
        {
            f.template operator()<static_cast<int>(Is)>(idx, time, arr, geom, workspace)
        } -> std::convertible_to<amrex::Real>;
    } && ...);
}(std::make_index_sequence<F::ncomp>{});

/**
 * @brief Verifies that a type implements boundary condition filling logic.
 * @tparam F Type to check against the concept.
 */
template <typename F>
concept BCConcept = std::is_trivially_copyable_v<F> &&
                    requires {
                        typename std::integral_constant<int, F::ncomp>;
                        typename std::integral_constant<int, F::ngrow>;
                        requires F::ncomp > 0;
                        requires F::ngrow >= 0;
                    } &&
                    requires(const F& t,
                             const amrex::IntVect& iv,               ///< Grid cell index (i, j, k) being filled
                             const amrex::Array4<amrex::Real>& dest, ///<  View into destination FAB data buffer
                             int dcomp,                       ///< Starting component index in destination to fill
                             int numcomp,                     ///< Number of components to process in this pass
                             const amrex::GeometryData& geom, ///< GPU-safe domain geometry data
                             amrex::Real time,                ///< Current simulation time
                             const amrex::BCRec* bcr,         ///< Array of boundary condition specifiers per component
                             int bcomp,                       ///< Component index offset into specifier array
                             int orig_comp                    ///< Original component index in source data
                    ) {
                        { t(iv, dest, dcomp, numcomp, geom, time, bcr, bcomp, orig_comp) } -> std::same_as<void>;
                    };

/**
 * @brief Verifies that a type provides valid initial condition evaluation.
 * @tparam F Type to check against the concept.
 */
template <typename F>
concept InitConcept = std::is_trivially_copyable_v<F> &&
                      requires {
                          typename std::integral_constant<int, F::ncomp>;
                          requires F::ncomp > 0;
                      } &&
                      requires(const F& f,
                               amrex::Real x, ///< Physical coordinate in X direction
                               amrex::Real y, ///< Physical coordinate in Y direction
                               amrex::Real z  ///< Physical coordinate in Z direction
                      ) {
                          { f.template operator()<0>(x, y, z) } -> std::convertible_to<amrex::Real>;
                      };

/**
 * @brief Verifies that a type encapsulates a complete partial differential equation model.
 * @tparam M Type to check against the concept.
 */
template <typename M>
concept PDEModelConcept = requires {
    typename std::integral_constant<int, M::ncomp>;
    typename std::integral_constant<int, M::ngrow>;
    requires M::ncomp > 0;
    requires M::ngrow >= 0;
    typename M::rhs_t;
    typename M::init_t;
    typename M::bc_t;
} && requires(M m) {
    requires RHSConcept<typename M::rhs_t>;
    { m.get_rhs() } -> std::same_as<const typename M::rhs_t&>;

    requires InitConcept<typename M::init_t>;
    { m.get_init() } -> std::same_as<const typename M::init_t&>;

    requires BCConcept<typename M::bc_t>;
    { m.get_bc() } -> std::same_as<const typename M::bc_t&>;
    { m.get_bc_recs() } -> std::convertible_to<const amrex::Vector<amrex::BCRec>&>;

    { m.get_geometry() } -> std::same_as<const amrex::Geometry&>;

    { m.get_comp_names() } -> std::convertible_to<amrex::Vector<std::string>>;

    { m.dt_hint() } -> std::convertible_to<amrex::Real>;
};

} // namespace treesap