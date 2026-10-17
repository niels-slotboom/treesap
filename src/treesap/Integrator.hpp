#pragma once

#include "pde-model/PDEModelConcepts.hpp"
#include "util/PlotExporter.hpp"

#include <AMReX.H>
#include <AMReX_BCRec.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>

/**
 * @brief Manage time integration for a PDE model.
 * @tparam PDEModelType Type of the PDE model.
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
     * @param block_size Block size for box array partition.
     */
    Integrator(PDEModelType model_, const amrex::IntVect& block_size)
        : model(std::move(model_)), ba(makeBoxArray(model.get_geometry(), block_size)), dm(ba),
          state_old(ba, dm, ncomp, ngrow), state_new(ba, dm, ncomp, ngrow) {}

    /**
     * @brief Destroy the Integrator object.
     */
    virtual ~Integrator() = default;

    /**
     * @brief Configure the plot file exporter.
     * @param path Output directory path.
     * @param export_interval Number of steps between exports.
     */
    void configureExporter(const fs::path& path, int export_interval) {
        exporter.emplace(state_old, model.get_geometry(), model.get_comp_names(), path, export_interval);
    }

    /**
     * @brief Initialize the state container using the initial condition functor.
     */
    void initialise() {
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

                for (int comp = 0; comp < ncomp; comp++) {
                    arr(i, j, k, comp) = init(x, y, z, comp);
                }
            });
        }
    }

    /**
     * @brief Execute one time step.
     * @param delta_time Size of the time step.
     */
    void step(amrex::Real delta_time) {
        FillPatch(state_old, time);

        if (exporter)
            exporter->exportIfNecessary(steps, time);

        computeNewState(delta_time);
        std::swap(state_old, state_new);

        steps++;
        time += delta_time;
    }

  protected:
    /**
     * @brief Compute the new state from the old state.
     * @param delta_time Size of the time step.
     */
    virtual void computeNewState(amrex::Real delta_time) = 0;

    /**
     * @brief Fill internal and external boundaries.
     * @param mf MultiFab to fill.
     * @param time_ Current simulation time.
     */
    void FillPatch(amrex::MultiFab& mf, amrex::Real time_) {
        amrex::Vector<amrex::MultiFab*> src{&mf};
        amrex::Vector<amrex::Real> times{time_};

        amrex::GpuBndryFuncFab<BCType> gpu_bndry_func(model.get_bc());
        amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCType>> phys_bc(model.get_geometry(), model.get_bc_recs(),
                                                                   gpu_bndry_func);

        amrex::FillPatchSingleLevel(mf, time_, src, times, 0, 0, ncomp, model.get_geometry(), phys_bc, 0);
    }

  private:
    /**
     * @brief Create a box array from the geometry and block size.
     * @param geom Domain geometry.
     * @param block_size Block size for partitioning.
     * @return Generated box array.
     */
    static amrex::BoxArray makeBoxArray(const amrex::Geometry& geom, const amrex::IntVect& block_size) {
        amrex::BoxArray result(geom.Domain());
        result.maxSize(block_size);
        return result;
    }

  protected:
    PDEModelType model; ///< PDE model instance.

    amrex::BoxArray ba;            ///< Box array for domain partition.
    amrex::DistributionMapping dm; ///< Distribution mapping for processes.

    amrex::MultiFab state_old; ///< State at the current time level.
    amrex::MultiFab state_new; ///< State at the new time level.

    size_t steps = 0;       ///< Number of completed time steps.
    amrex::Real time = 0.0; ///< Current simulation time.
  private:
    std::optional<PlotExporter> exporter = std::nullopt; ///< Exporter instance.
};