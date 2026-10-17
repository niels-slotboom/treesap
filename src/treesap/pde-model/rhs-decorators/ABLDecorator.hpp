#pragma once

#include "../../util/stencil.hpp"
#include "../PDEModelConcepts.hpp"
#include "NoOpRHS.hpp"

/**
 * @brief Verifies that a profile type is trivially copyable, accepts a real value,
 *        and evaluates to zero at 0.0 (bulk interface) and one at 1.0 (domain boundary).
 * @tparam F Profile type to check.
 */
// needs the implementer to be constexpr
template <typename F>
concept ABLProfileConcept = std::is_trivially_copyable_v<F> && requires(F f, amrex::Real s) {
    { f(s) } -> std::convertible_to<amrex::Real>;
    requires f(0.0) == 0.0;
    requires f(1.0) == 1.0;
};

/**
 * @brief Decorates a right-hand side model to add absorbing boundary layer (ABL) damping.
 * @tparam RHSType Underlying right-hand side model type.
 * @tparam ABLProfileType Profile function type.
 */
template <typename RHSType, typename ABLProfileType>
    requires RHSConcept<RHSType> && ABLProfileConcept<ABLProfileType>
struct ABLDecorator {
    /// Number of components.
    static constexpr int ncomp = RHSType::ncomp;

    /// Number of ghost cells.
    static constexpr int ngrow = RHSType::ngrow;

    struct ABLData {
        ABLData(amrex::GpuArray<amrex::Real, ncomp> target_values_, int thickness_, amrex::Real max_strength_,
                ABLProfileType profile_)
            : target_values(std::move(target_values_)), thickness(thickness_),
              inv_thickness(1.0 / static_cast<amrex::Real>(thickness_)), max_strength(max_strength_),
              profile(std::move(profile_)) {}

        ABLProfileType profile; ///< Profile of damping strength (s = 0 at interface, s = 1 at boundary).
        int thickness;
        amrex::Real inv_thickness;                         ///< Inverse layer thickness (1 / thickness).
        amrex::Real max_strength;                          ///< Maximum damping strength.
        amrex::GpuArray<amrex::Real, ncomp> target_values; ///< Target values for field damping.
    };

    /**
     * @brief Initializes the absorbing boundary layer decorator.
     * @param rhs Underlying right-hand side model.
     * @param profile Profile function.
     * @param geom Grid geometry.
     * @param thickness Layer thickness in cells.
     * @param max_strength Maximum damping strength.
     * @param target_values Target field values for damping.
     */
    ABLDecorator(RHSType rhs, const amrex::Geometry& geom, ABLData data_)
        : rhs(std::move(rhs)), data(std::move(data_)), bulk([&]() {
              auto b = geom.Domain();
              b.grow(-data.thickness);
              return b;
          }()) {}

    /**
     * @brief Evaluates the decorated right-hand side with added boundary damping.
     * @param i Grid cell index in X direction.
     * @param j Grid cell index in Y direction.
     * @param k Grid cell index in Z direction.
     * @param comp Component index.
     * @param time Current simulation time.
     * @param arr Data array view buffer.
     * @return Adjusted right-hand side value.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                                                                    const amrex::Array4<const amrex::Real>& arr) const {
        return rhs(i, j, k, comp, time, arr) - sigma(i, j, k) * (arr(i, j, k, comp) - data.target_values[comp]);
    }

    /**
     * @brief Computes the damping coefficient at a grid cell.
     * @param i Grid cell index in X direction.
     * @param j Grid cell index in Y direction.
     * @param k Grid cell index in Z direction.
     * @return Damping coefficient value.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real sigma(int i, int j, int k) const {
        using namespace stencil;

        amrex::IntVect idx{i, j, k};

        if (bulk.contains(idx)) // return 0 if we are inside the bulk
            return 0.0;

        // if not in the bulk, compute distance to bulk
        amrex::IntVect nearest_in_bulk = {amrex::Clamp(idx[dir::x], bulk.smallEnd(dir::x), bulk.bigEnd(dir::x)),
                                          amrex::Clamp(idx[dir::y], bulk.smallEnd(dir::y), bulk.bigEnd(dir::y)),
                                          amrex::Clamp(idx[dir::z], bulk.smallEnd(dir::z), bulk.bigEnd(dir::z))};
        amrex::IntVect connecting_vector = idx - nearest_in_bulk;
        amrex::Real norm = std::sqrt(connecting_vector[dir::x] * connecting_vector[dir::x] +
                                     connecting_vector[dir::y] * connecting_vector[dir::y] +
                                     connecting_vector[dir::z] * connecting_vector[dir::z]);

        amrex::Real scaled_dist = amrex::Clamp(data.inv_thickness * norm, 0.0, 1.0);
        return data.max_strength * data.profile(scaled_dist);
    }

    RHSType rhs; ///< Underlying right-hand side model.
    ABLData data;
    amrex::Box bulk; ///< Inner simulation domain box without damping layer.
};

// Verify that ABLDecorator satisfies the right-hand side concept.
static_assert(RHSConcept<ABLDecorator<NoOpRHS<1, 1>, decltype([](amrex::Real x) constexpr { return x; })>>);