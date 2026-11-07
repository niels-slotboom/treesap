// AMReX includes
#include "AMReX_Box.H"
#include "AMReX_CoordSys.H"
#include "AMReX_Geometry.H"
#include "AMReX_RealBox.H"
#include <AMReX.H>
#include <AMReX_Print.H>

// TreESaP includes
#include "init/FunctionInit.hpp"
#include "integrators/RK4Integrator.hpp"
#include "pde-model/ABLPDEModel.hpp"

// our files
#include "WaveEqnRHS.hpp"
#include "pde-model/bcs/DirichletBC.hpp"
#include "pde-model/rhs-decorators/ABLDecorator.hpp"
#include "util/ScopedTimer.hpp"

int main(int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        // --- set up the domain of the problem ---
        amrex::Box index_range{{0, 0, 0}, {256, 256, 256}};              // 256³ resolution
        amrex::RealBox coord_range{{-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0}}; // coordinate domain [-2,2]³
        int is_per[] = {0, 0, 0};                                        // no periodic boundaries
        amrex::Geometry geom(index_range, &coord_range, amrex::CoordSys::cartesian, is_per);

        // --- simulation parameters ---
        constexpr int ngrow = 1;

        // --- define types for the PDE model ---
        using RHSType = WaveEqnRHS<ngrow>;                      // our custom wave equation RHS implementation
        using InitType = treesap::FunctionInit<RHSType::ncomp>; // initialisation with function expressions in strings;
                                                                // RHS dictates ncomp
        using BCType = treesap::DirichletBC<RHSType::ncomp,
                                            RHSType::ngrow>; // Dirichlet boundaries with RHS-dictated ncomp and ngrow
        using ModelType = treesap::PDEModel<RHSType, InitType, BCType>;

        // --- instantiate PDE model ---
        amrex::Real dx = geom.CellSize(0);
        amrex::Real dt = std::sqrt(8.0) * dx; // we will use RK4, which at ngrow = 1 has this as a CFL bound
        RHSType rhs(geom.data());             // RHS of the bulk PDE, without ABL terms
        InitType init({"exp(-(x^2 + y^2 + z^2)/0.1^2)", "0.0", "0.0", "0.0"}); // functions used to initialise fields
        BCType bc{0.0, 0.0, 0.0, 0.0};                                         // boundary values set to 0
        ModelType model({"phi", "u_0", "u_1", "u_2"}, std::move(geom), dt, std::move(rhs), std::move(init),
                        std::move(bc));

        // --- set up integrator ---
        treesap::RK4Integrator integrator(std::move(model), {128, 128, 128});
        integrator.initialise(); // populate initial field values
        integrator.configureExporter("raw/wave_eqn",
                                     20);     // configure exports to happen every 20 steps,
                                              // with plotfiles being saved to ./raw/wave_eqn*
        treesap::ScopedTimer::enabled = true; // enable simple profiling

        // --- run integration loop ---
        for (size_t step = 0; step <= 2500; ++step) {
            integrator.step();
        }

        // --- print profiling overview ---
        if (treesap::ScopedTimer::enabled) {
            treesap::ScopedTimer::printOverview();
        }
    }
    amrex::Finalize();
}