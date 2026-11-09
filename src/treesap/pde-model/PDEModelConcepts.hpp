// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>

#include <concepts>
#include <type_traits>

namespace treesap {

/**
 * @brief Verifies that a type evaluates the spatial right-hand side operator.
 * @tparam F Type to check against the concept.
 */
template <typename F>
concept RHSConcept = std::is_trivially_copyable_v<F> && requires {
    typename std::integral_constant<int, F::ncomp>;
    typename std::integral_constant<int, F::ngrow>;
    requires F::ncomp > 0;
    requires F::ngrow >= 0;
} && requires(const F& f, int i, int j, int k, amrex::Real time, const amrex::Array4<const amrex::Real>& arr) {
    // Use a compile-time constant (e.g., 0) as the template argument,
    // assuming components are indexed from 0 up to F::ncomp - 1.
    {
        f.template operator()<0>(i, j, k, time, arr)
    } -> std::convertible_to<amrex::Real>; // if this fails, check if operator() is marked const
};

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
                             const amrex::IntVect& iv,               ///< 1. Grid cell index (i, j, k) being filled
                             const amrex::Array4<amrex::Real>& dest, ///< 2. View into destination FAB data buffer
                             int dcomp,                       ///< 3. Starting component index in destination to fill
                             int numcomp,                     ///< 4. Number of components to process in this pass
                             const amrex::GeometryData& geom, ///< 5. GPU-safe domain geometry data
                             amrex::Real time,                ///< 6. Current simulation time
                             const amrex::BCRec* bcr, ///< 7. Array of boundary condition specifiers per component
                             int bcomp,               ///< 8. Component index offset into specifier array
                             int orig_comp            ///< 9. Original component index in source data
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
                               amrex::Real x, ///< 1. Physical coordinate in X direction
                               amrex::Real y, ///< 2. Physical coordinate in Y direction
                               amrex::Real z  ///< 3. Physical coordinate in Z direction
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