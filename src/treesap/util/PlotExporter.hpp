// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once

#include "AMReX.H"
#include "AMReX_Geometry.H"
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "util/ScopedTimer.hpp"

#include <filesystem>

namespace treesap {

namespace fs = std::filesystem;

/**
 * @brief Manages the export of AMReX plot files at specified step intervals.
 * Used by `treesap::Integrator` as a `std::optional` member to periodically export the simulation state during
 * timestepping.
 */
class PlotExporter {
  public:
    /**
     * @brief Initializes the plot exporter.
     * @param data_ Reference to the `amrex::MultiFab` holding the export data.
     * @param geom_ Reference to the grid geometry of `data_`.
     * @param comp_names_ Vector of component name strings.
     * @param path_ Base path for file export.
     * @param export_interval Number of steps between exports.
     *
     * The step number is appended to `path_` upon export to make multiple exports distinguishable and prevent
     * overwrites. The export is triggered at step `0`, step `export_interval`, step `2 * export_interval` etc.
     */
    PlotExporter(const amrex::MultiFab& data_, const amrex::Geometry& geom_,
                 const amrex::Vector<std::string>& comp_names_, fs::path path_, int export_interval)
        : data(data_), geom(geom_), comp_names(comp_names_), path(std::move(path_)), export_interval(export_interval) {}

    /**
     * @brief Writes a plot file if the current step matches the export interval.
     * @param step Current simulation step.
     * @param time Current simulation time.
     *
     * Export interval is matched if `step % export_interval == 0`. Thus, the export is triggered at step `0`, step
     * `export_interval`, step `2 * export_interval` etc.
     */
    void exportIfNecessary(int step, amrex::Real time) const {
        if (step % export_interval != 0) // only export when step % export_interval == 0
            return;
        ScopedTimer timer("PlotExporter::exportIfNecessary()");
        amrex::WriteSingleLevelPlotfile(amrex::Concatenate(path.string(), step, 5), data, comp_names, geom, time, step);
    }

  private:
    const amrex::MultiFab& data;                  ///< Reference to the export data.
    const amrex::Geometry& geom;                  ///< Reference to the grid geometry of data.
    const amrex::Vector<std::string>& comp_names; ///< Reference to the vector of component names.
    fs::path path;                                ///< Base file path for export.
    int export_interval;                          ///< Step interval for export execution.
};

} // namespace treesap