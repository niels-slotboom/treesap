#pragma once

#include "../PDEModelConcepts.hpp"
#include "NoOpRHS.hpp"

// needs the implementer to be constexpr
template <typename F>
concept ABLProfileConcept = std::is_trivially_copyable_v<F> && requires(F f, amrex::Real s) {
    { f(s) } -> std::convertible_to<amrex::Real>;
    requires f(0.0) == 0.0;
    requires f(1.0) == 1.0;
};

template <typename RHSType, typename ABLProfileType>
    requires RHSConcept<RHSType> && ABLProfileConcept<ABLProfileType>
struct ABLDecorator {
    static constexpr int ncomp = RHSType::ncomp;
    static constexpr int ngrow = RHSType::ngrow;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, int comp, amrex::Real time,
                                                                    const amrex::Array4<const amrex::Real>& arr) const {
        return rhs(i, j, k, comp, time, arr) -
               max_strength * (arr(i, j, k, comp) - target_values[comp]); // TODO: Add extra factor involving profile
    }

    RHSType rhs;
    ABLProfileType profile;                            ///< Profile of the damping strength, s = 0 is the bulk
                                                       ///< interface and s = 1 the domain boundary
    int thickness;                                     ///< Layer thickness in cells
    amrex::GpuArray<amrex::Real, ncomp> target_values; ///< values that the damping should bring the field towards
    amrex::Real max_strength;                          ///< maximum damping strength (applies where profile(x) = 1)
};

static_assert(RHSConcept<ABLDecorator<NoOpRHS<1, 1>, decltype([](amrex::Real x) { return x; })>>);