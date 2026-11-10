// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides the concrete `treesap::EulerIntegrator` implementation of `treesap::Integrator`.
 */

#pragma once
#include "AMReX_Array4.H"
#include "AMReX_MFIter.H"
#include "Integrator.hpp"

namespace treesap {

/**
 * @brief Concrete `Integrator` implementation for the forward Euler scheme.
 * @tparam PDEModelType Type of the PDE model to be forwarded to the base class `Integrator`.
 *
 * For a PDE of the form
 * \f[
 *      \partial_t \Phi = F(\Phi,t),
 * \f]
 * with $F = \verb|model.get_rhs()|$, the forward Euler scheme prescribes the update equation
 * \f[
 *      \verb|state_new| = \verb|state_old| + \verb|dt_hint| \cdot F(\verb|state_old|,t).
 * \f]
 */
template <typename PDEModelType>
    requires PDEModelConcept<PDEModelType>
class EulerIntegrator : public Integrator<PDEModelType> {
    using Base = Integrator<PDEModelType>;

    // bring member constants into class scope
    using Base::ncomp;
    using Base::ngrow;

    // bring protected members into scope
    using Base::model;
    using Base::state_new;
    using Base::state_old;
    using Base::time;

    using typename Base::RHSType;

  public:
    // Inherit constructor
    using Integrator<PDEModelType>::Integrator;

  protected:
    /**
     * @brief Compute the new state with the Euler method.
     * @param delta_time Size of the time step.
     *
     * The update equation is
     * \f[
     *      \verb|state_new| = \verb|state_old| + \verb|dt_hint| \cdot F(\verb|state_old|,t),
     * \f]
     * where $F = \verb|model.get_rhs()|$.
     */
    amrex::Real computeNewState(amrex::Real delta_time) override {
        // make local for lambda capture
        RHSType rhs = model.get_rhs();
        amrex::Real time_ = time;

        for (amrex::MFIter mfi(state_old); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            amrex::Array4<const amrex::Real> arr_old = state_old.const_array(mfi);
            amrex::Array4<amrex::Real> arr_new = state_new.array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                static_for<ncomp>([=] AMREX_GPU_DEVICE<int comp>() {
                    arr_new(i, j, k, comp) =
                        arr_old(i, j, k, comp) + delta_time * rhs.template operator()<comp>(i, j, k, time_, arr_old);
                });
            });
        }

        return delta_time;
    }
};

}; // namespace treesap