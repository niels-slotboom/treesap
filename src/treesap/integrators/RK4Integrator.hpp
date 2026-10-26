#pragma once
#include "AMReX_MFIter.H"
#include "Integrator.hpp"

namespace tsap {

/**
 * @brief Implements the Runge-Kutta 4 time integration scheme.
 * @tparam PDEModelType Type of the PDE model.
 */
template <typename PDEModelType>
    requires PDEModelConcept<PDEModelType>
class RK4Integrator : public Integrator<PDEModelType> {
    using Base = Integrator<PDEModelType>;

  private: // import base class members to scope
    using Base::ncomp;
    using Base::ngrow;

    using Base::ba;
    using Base::dm;
    using Base::FillPatch;

    using Base::model;

    using Base::state_new;
    using Base::state_old;
    using Base::steps;
    using Base::time;

    using typename Base::RHSType;

  public:
    /**
     * @brief Initialize the Runge-Kutta 4 integrator.
     * @param model_ PDE model instance.
     * @param block_size Block size for box array partition.
     */
    RK4Integrator(PDEModelType model_, const amrex::IntVect& block_size)
        : Base(std::move(model_), block_size), init(state_old), acc(state_new), arg(ba, dm, ncomp, ngrow),
          arg_next(ba, dm, ncomp, ngrow) {}

  private:                                // coefficients
    static constexpr int stage_count = 4; ///< Number of integration stages.
    static constexpr amrex::Array<amrex::Real, stage_count> a_fraction_arr{0.0, 0.5, 0.5,
                                                                           1.0}; ///< Stage time offset coefficients.
    static constexpr amrex::Array<amrex::Real, stage_count> b_fraction_arr{
        1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0}; ///< Stage accumulation coefficients.

  public: // public interface
    /**
     * @brief Compute the new state using four Runge-Kutta stages.
     * @param delta_time Size of the time step.
     */
    void computeNewState(amrex::Real delta_time) override {
        runStage<0>(delta_time);
        runStage<1>(delta_time);
        runStage<2>(delta_time);
        runStage<3>(delta_time);
    }

  private: // fused stage kernel
    /**
     * @brief Execute a single integration stage.
     * @tparam stage_number Index of the current stage.
     * @param dt Size of the time step.
     */
    template <int stage_number> void runStage(amrex::Real dt) {
        constexpr bool is_first = (stage_number == 0);
        constexpr bool is_last = (stage_number == (stage_count - 1));

        amrex::Real a_next = dt * a_fraction_arr[(stage_number + 1) % stage_count];
        amrex::Real time_arg = time + dt * a_fraction_arr[stage_number];
        amrex::Real b = dt * b_fraction_arr[stage_number];

        if constexpr (!is_first) { // finalise arg for rhs evaluation
            std::swap(arg_next, arg);
            FillPatch(arg, time_arg);
        }

        RHSType rhs_ = model.get_rhs(); // local copy for lambda capture

        for (amrex::MFIter mfi(init); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.validbox();
            amrex::Array4<const amrex::Real> init_arr = init.const_array(mfi);
            amrex::Array4<amrex::Real> acc_arr = acc.array(mfi);
            amrex::Array4<const amrex::Real> arg_arr = arg.const_array(mfi);
            amrex::Array4<amrex::Real> arg_next_arr = arg_next.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int comp = 0; comp < ncomp; ++comp) {
                    // compute stage k_i for current voxel and accumulate
                    amrex::Real stage;
                    if constexpr (is_first) {
                        stage = rhs_(i, j, k, comp, time_arg, init_arr);
                        acc_arr(i, j, k, comp) = init_arr(i, j, k, comp) + b * stage;
                    } else {
                        stage = rhs_(i, j, k, comp, time_arg, arg_arr);
                        acc_arr(i, j, k, comp) += b * stage;
                    }

                    // prepare arg_next if there is a next stage
                    if constexpr (!is_last) {
                        arg_next_arr(i, j, k, comp) = init_arr(i, j, k, comp) + a_next * stage;
                    }
                }
            });
        }
    }

  private: // class members
    // aliases for existing base class registers
    const amrex::MultiFab& init; ///< Reference to the initial state container.
    amrex::MultiFab& acc;        ///< Reference to the accumulated state container.
    // temporary registers for more efficient evaluation
    amrex::MultiFab arg;      ///< Temporary container for current stage argument.
    amrex::MultiFab arg_next; ///< Temporary container for next stage argument.
};

} // namespace tsap