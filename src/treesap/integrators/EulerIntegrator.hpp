#pragma once
#include "AMReX_Array4.H"
#include "AMReX_MFIter.H"
#include "Integrator.hpp"

/**
 * @brief Implement the Euler time integration scheme.
 * @tparam PDEModelType Type of the PDE model.
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
     */
    void computeNewState(amrex::Real delta_time) override {
        // make local for lambda capture
        RHSType rhs = model.get_rhs();
        amrex::Real time_ = time;

        for (amrex::MFIter mfi(state_old); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            amrex::Array4<const amrex::Real> arr_old = state_old.const_array(mfi);
            amrex::Array4<amrex::Real> arr_new = state_new.array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int comp = 0; comp < ncomp; ++comp) {
                    arr_new(i, j, k, comp) = arr_old(i, j, k, comp) + delta_time * rhs(i, j, k, comp, time_, arr_old);
                }
            });
        }
    }
};