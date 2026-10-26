#pragma once

#include "AMReX.H"
#include "AMReX_Geometry.H"
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include <filesystem>

namespace tsap {

namespace fs = std::filesystem;

/**
 * @brief Manages the export of AMReX plot files at specified step intervals.
 */
class PlotExporter {
  public:
    /**
     * @brief Initializes the plot exporter.
     * @param data_ Reference to the simulation data.
     * @param geom_ Reference to the grid geometry.
     * @param comp_names_ Vector of component names.
     * @param path_ Base path for exported files.
     * @param export_interval Number of steps between exports.
     */
    PlotExporter(const amrex::MultiFab& data_, const amrex::Geometry& geom_,
                 const amrex::Vector<std::string>& comp_names_, fs::path path_, int export_interval)
        : data(data_), geom(geom_), comp_names(comp_names_), path(std::move(path_)), export_interval(export_interval) {}

    /**
     * @brief Writes a plot file if the current step matches the export interval.
     * @param step Current simulation step.
     * @param time Current simulation time.
     */
    void exportIfNecessary(int step, amrex::Real time) const {
        if (step % export_interval != 0) // only export when step % export_interval == 0
            return;
        amrex::WriteSingleLevelPlotfile(amrex::Concatenate(path.string(), step, 5), data, comp_names, geom, time, step);
    }

  private:
    const amrex::MultiFab& data;                  ///< Reference to the simulation data.
    const amrex::Geometry& geom;                  ///< Reference to the grid geometry.
    const amrex::Vector<std::string>& comp_names; ///< Reference to the vector of component names.
    fs::path path;                                ///< Base file path for export.
    int export_interval;                          ///< Step interval for export execution.
};

} // namespace tsap