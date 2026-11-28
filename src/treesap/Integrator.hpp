// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Provides the `treesap::Integrator` base class template.
 */

#pragma once

#include "pde-model/PDEModelConcepts.hpp"
#include "util/PlotExporter.hpp"
#include "util/ScopedTimer.hpp"
#include "util/static_for.hpp"

#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>

/**
 * @namespace treesap
 * @brief Contains all of the infrastructre provided by the TreESaP library.
 */
namespace treesap {

/**
 * @brief Abstract base class template managing time integration for a PDE model.
 * @tparam PDEModelType Type of the PDE model.
 *
 * This class defines the common interface and lifecycle of a time integrator, independently of the particular
 * integration scheme or the number of grid levels being managed. Derived implementations are responsible for
 * initialising the simulation state and advancing it in time.
 *
 * The public `initialise()` and `step()` methods manage the overall execution protocol, including profiling,
 * timestep counting, and simulation time. The actual initialisation and time integration procedures are delegated
 * to the derived implementation through `initialise_impl()` and `step_impl()`, respectively.
 *
 * The optional `pre_step()` and `post_step()` hooks allow derived implementations to perform additional operations
 * immediately before and after each timestep. These may, for example, be used for diagnostics, data export, or
 * other operations associated with the timestep lifecycle.
 */
template <typename PDEModelType>
    requires PDEModelConcept<PDEModelType>
class Integrator {
  protected:
    static constexpr int ncomp = PDEModelType::ncomp;
    static constexpr int ngrow = PDEModelType::ngrow;

    using RHSType = typename PDEModelType::rhs_t;
    using InitType = typename PDEModelType::init_t;
    using BCType = typename PDEModelType::bc_t;

  public:
    /**
     * @brief Construct a new Integrator object.
     * @param model_ PDE model instance.
     */
    Integrator(PDEModelType model_) : model(std::move(model_)) {}

    /**
     * @brief Destroy the Integrator object.
     */
    virtual ~Integrator() = default;

    /**
     * @brief Initialise the simulation state.
     *
     * Delegates the initialisation procedure to `initialise_impl()` and measures its execution time.
     */
    void initialise() {
        ScopedTimer timer("Integrator::initialise()");
        initialise_impl();
    }

    /**
     * @brief Execute one timestep.
     *
     * Calls `pre_step()`, advances the simulation through `step_impl()`, updates the timestep counter and simulation
     * time, and finally calls `post_step()`. The timestep width returned by `step_impl()` determines the amount by
     * which the simulation time is advanced.
     *
     * @note The actual timestep width is determined by the derived implementation. For fixed-step schemes such as
     * Euler and RK4, it will typically equal `model.dt_hint()`, whereas adaptive schemes may choose a different
     * timestep width.
     */
    void step() {
        {
            ScopedTimer timer("Integrator::step() [PRE]");
            pre_step();
        }
        {
            ScopedTimer timer("Integrator::step() [STEP]");
            amrex::Real delta_time = step_impl();
            steps++;
            time += delta_time;
        }
        {
            ScopedTimer timer("Integrator::step() [POST]");
            post_step();
        }
    }

  protected:
    /**
     * @brief Initialise the simulation state.
     *
     * Derived implementations must provide the initialisation procedure appropriate to their state representation
     * and level structure. This typically involves evaluating the initial condition supplied by `model` and
     * populating the corresponding state containers.
     */
    virtual void initialise_impl() = 0;

    /**
     * @brief Advance the simulation by one timestep.
     * @return Actual timestep width used by the integration procedure.
     *
     * Derived implementations must provide the time integration procedure, advancing the simulation state according
     * to their integration scheme. The returned timestep width is used by `step()` to update the simulation time.
     */
    virtual amrex::Real step_impl() = 0;

    /**
     * @brief Perform operations immediately before each timestep.
     *
     * Optional hook for operations that should precede the time integration procedure, such as diagnostics or
     * preparation of data required during the timestep. Derived implementations may override this method if needed.
     */
    virtual void pre_step() {}

    /**
     * @brief Perform operations immediately after each timestep.
     *
     * Optional hook for operations that should follow the time integration procedure, such as diagnostics or data
     * export. At this point, the timestep counter and simulation time have already been updated. Derived
     * implementations may override this method if needed.
     */
    virtual void post_step() {}

  protected:
    PDEModelType model; ///< PDE model instance.

    size_t steps = 0;       ///< Number of completed time steps.
    amrex::Real time = 0.0; ///< Current simulation time.
};

} // namespace treesap