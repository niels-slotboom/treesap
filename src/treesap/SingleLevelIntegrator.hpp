#pragma once

#include "Integrator.hpp"
#include "pde-model/PDEModelConcepts.hpp"

namespace treesap {

/**
 * @brief Abstract base class for single-level time integration.
 *
 * Provides the state storage and grid infrastructure required to evolve a PDE model on a single grid level.
 * In particular, it constructs the box array and distribution mapping, manages the old and new state MultiFabs,
 * and provides routines for initialisation and boundary filling.
 *
 * The general integration lifecycle is defined by the parent `Integrator` class. This class implements that
 * lifecycle for a single-level grid, while leaving the actual time-integration scheme to derived classes.
 * These must implement `computeNewState()`, which advances the solution and returns the time-step width used.
 *
 * An optional plot exporter can be configured to write the solution at specified intervals.
 */
template <typename PDEModelType>
    requires PDEModelConcept<PDEModelType>
class SingleLevelIntegrator : public Integrator<PDEModelType> {
  protected:
    using Base = Integrator<PDEModelType>;

    using Base::ncomp;
    using Base::ngrow;

    using typename Base::BCType;
    using typename Base::InitType;
    using typename Base::RHSType;

    using Base::model;
    using Base::steps;
    using Base::time;

  public:
    /**
     * @brief Construct a single-level integrator for a PDE model.
     * @param model_ PDE model instance.
     * @param block_size Maximum box dimensions used to partition the computational domain.
     *
     * Constructs the box array from the model geometry, creates its distribution mapping, and allocates
     * the old and new state MultiFabs with the component count and ghost-cell width specified by the model.
     */
    SingleLevelIntegrator(PDEModelType model_, const amrex::IntVect& block_size)
        : Base(std::move(model_)), ba(makeBoxArray(model.get_geometry(), block_size)), dm(ba),
          state_old(ba, dm, ncomp, ngrow), state_new(ba, dm, ncomp, ngrow) {}

    /**
     * @brief Destroy the single-level integrator.
     */
    virtual ~SingleLevelIntegrator() = default;

    /**
     * @brief Configure the plot file exporter.
     * @param path Output directory path.
     * @param export_interval Number of steps between exports.
     *
     * Creates an exporter associated with the current state, model geometry, and component names.
     * Exporting is performed automatically by `pre_step()` at the configured interval.
     */
    void configureExporter(const fs::path& path, int export_interval) {
        exporter.emplace(state_old, model.get_geometry(), model.get_comp_names(), path, export_interval);
    }

    /**
     * @brief Initialise the single-level solution.
     *
     * Evaluates the model's initial-condition functor at the centre of each valid grid cell and stores
     * the resulting component values in `state_old`. The initial-condition functor is copied from the
     * PDE model and evaluated independently for each component.
     */
    void initialise_impl() override {
        InitType init = model.get_init();
        const amrex::Geometry& geom = model.get_geometry();
        amrex::GpuArray<amrex::Real, 3> dx = geom.CellSizeArray();
        amrex::GpuArray<amrex::Real, 3> lo = geom.ProbLoArray();

        for (amrex::MFIter mfi(state_old); mfi.isValid(); ++mfi) {
            amrex::Box box = mfi.validbox();
            amrex::Array4<amrex::Real> arr = state_old.array(mfi);
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real x = lo[0] + (i + 0.5) * dx[0];
                amrex::Real y = lo[1] + (j + 0.5) * dx[1];
                amrex::Real z = lo[2] + (k + 0.5) * dx[2];

                static_for<ncomp>(
                    [=] AMREX_GPU_DEVICE<int comp>() { arr(i, j, k, comp) = init.template operator()<comp>(x, y, z); });
            });
        }
    }

    /**
     * @brief Perform operations immediately before advancing the solution.
     *
     * Exports the current solution if an exporter has been configured and an export is due.
     * Then fills the ghost cells of `state_old`, exchanging data between neighbouring boxes and
     * applying the model's physical boundary conditions.
     */
    void pre_step() final override {
        if (exporter) {
            exporter->exportIfNecessary(steps, time);
        }

        FillPatch(state_old, time);
    }

    /**
     * @brief Advance the solution by one time step.
     * @return Width of the time step actually taken.
     *
     * Delegates the update to `computeNewState()`, passing the time-step hint supplied by the PDE model.
     * The returned time-step width is used by the parent `Integrator` to update the simulation time.
     */
    amrex::Real step_impl() final override { return computeNewState(model.dt_hint()); }

    /**
     * @brief Perform operations immediately after advancing the solution.
     *
     * Swaps the old and new state MultiFabs, making the newly computed solution the current state
     * for the next time step.
     */
    void post_step() final override { std::swap(state_old, state_new); }

  protected:
    /**
     * @brief Compute the new solution from the current state.
     * @param dt_hint Suggested time-step width supplied by the PDE model.
     * @return Width of the time step actually taken.
     *
     * Derived classes implement this method to advance the solution according to their
     * time-integration scheme, writing the result into `state_new`. The returned value specifies
     * the time interval covered by that update and need not equal `dt_hint` for schemes that
     * dynamically adjust the time-step width.
     *
     * For fixed-step schemes such as `SingleLevelEulerIntegrator` and `SingleLevelRK4Integrator`, the returned width
     * will typically equal `dt_hint`. Adaptive schemes, such as Dormand-Prince, may return a
     * different value.
     */
    virtual amrex::Real computeNewState(amrex::Real dt_hint) = 0;

    /**
     * @brief Fill the ghost cells of a MultiFab.
     * @param mf MultiFab whose ghost cells are to be filled.
     * @param time_ Simulation time at which the boundary conditions are evaluated.
     *
     * Uses `amrex::FillPatchSingleLevel` to exchange data between neighbouring boxes and fill
     * physical boundaries using the boundary-condition functor provided by the PDE model,
     * `model.get_bc()`.
     */
    void FillPatch(amrex::MultiFab& mf, amrex::Real time_) {
        ScopedTimer timer("Integrator::FillPatch()");
        amrex::Vector<amrex::MultiFab*> src{&mf};
        amrex::Vector<amrex::Real> times{time_};

        amrex::GpuBndryFuncFab<BCType> gpu_bndry_func(model.get_bc());
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCType>> phys_bc(model.get_geometry(), model.get_bc_recs(),
                                                                   gpu_bndry_func);

        amrex::FillPatchSingleLevel(mf, time_, src, times, 0, 0, ncomp, model.get_geometry(), phys_bc, 0);
    }

  private:
    /**
     * @brief Create a box array covering the computational domain.
     * @param geom Domain geometry.
     * @param block_size Maximum box dimensions used for partitioning.
     * @return Box array covering the domain, partitioned according to `block_size`.
     */
    static amrex::BoxArray makeBoxArray(const amrex::Geometry& geom, const amrex::IntVect& block_size) {
        amrex::BoxArray result(geom.Domain());
        result.maxSize(block_size);
        return result;
    }

  protected:
    amrex::BoxArray ba;            ///< Box array defining the spatial partition of the domain.
    amrex::DistributionMapping dm; ///< Distribution of boxes across processes.

    amrex::MultiFab state_old; ///< Solution state at the beginning of the current time step.
    amrex::MultiFab state_new; ///< Solution state being computed for the next time step.

  private:
    std::optional<PlotExporter> exporter = std::nullopt; ///< Optional plot file exporter.
};

} // namespace treesap