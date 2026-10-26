#include <AMReX.H>
#include <AMReX_Print.H>

#include "AMReX_CoordSys.H"
#include "WaveEqnRHS.hpp"
#include "init/ConstInit.hpp"
#include "init/FunctionInit.hpp"
#include "integrators/RK4Integrator.hpp"
#include "pde-model/ABLPDEModel.hpp"
#include "pde-model/rhs-decorators/NoOpRHS.hpp"
#include <AMReX.H>

/**
 * @brief Construct the absorbing boundary layer test model.
 * @return Configured ABLPDEModel instance.
 */
inline auto create_abl_test_model(const amrex::Geometry& geom) {
    using RHSType = tsap::NoOpRHS<1, 1>;
    using InitType = tsap::ConstInit<1>;

    RHSType rhs{};
    InitType init{0.0}; // Initialize field to zero everywhere

    auto profile = [](amrex::Real s) constexpr {
        return s * s * (3.0 - 2.0 * s); // Cubic smoothstep profile
    };

    using DecoratedRHS = tsap::ABLDecorator<RHSType, decltype(profile)>;
    typename DecoratedRHS::ABLData abl_data{amrex::GpuArray<amrex::Real, 1>{1.0}, // Target value set to 1.0
                                            16,                                   // Thickness in cells
                                            10.0,                                 // Maximum damping strength
                                            profile};

    return tsap::ABLPDEModel<RHSType, InitType, decltype(profile)>({"phi"}, geom, rhs, abl_data, init);
}

/**
 * @brief Construct the absorbing boundary layer test model with wave equation RHS.
 * @return Configured ABLPDEModel instance.
 */
template <int ngrow> inline auto create_wave_eqn_abl_test_model(const amrex::Geometry& geom) {
    using RHSType = WaveEqnRHS<ngrow>;
    constexpr int ncomp = RHSType::ncomp;
    using InitType = tsap::FunctionInit<ncomp>;

    // phi expression (initial position)
    std::string phi_expr = "exp(-((x - (-0.4))^2 + y^2 + z^2) / 0.045) * cos(80.42478 * (x - (-0.4)))";

    // dt_phi expression (initial velocity for a purely right-moving wave with c = 1)
    std::string dphi_expr = "exp(-((x - (-0.4))^2 + y^2 + z^2) / 0.045) * (((x - (-0.4)) / 0.0225) * cos(80.42478 * (x "
                            "- (-0.4))) + 80.42478 * sin(80.42478 * (x - (-0.4))))";

    double dx = geom.CellSize(0);

    RHSType rhs(dx);
    InitType init({phi_expr, dphi_expr}); // Initialize field to zero everywhere

    auto profile = [](amrex::Real s) constexpr {
        return s * s * s * (10.0 - 15 * s + 6 * s * s); // Cubic smoothstep profile
    };

    using DecoratedRHS = tsap::ABLDecorator<RHSType, decltype(profile)>;
    typename DecoratedRHS::ABLData abl_data{amrex::GpuArray<amrex::Real, ncomp>{0.0, 0.0},
                                            64,       // Thickness in cells
                                            0.3 / dx, // Maximum damping strength
                                            profile};

    return tsap::ABLPDEModel<RHSType, InitType, decltype(profile)>({"phi", "pi"}, geom, rhs, abl_data, init);
}

int main(int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::Box dom{{0, 0, 0}, {512, 512, 512}};
        amrex::RealBox coord_dom{{-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0}};
        amrex::Geometry geom(dom, coord_dom, amrex::CoordSys::cartesian, {0, 0, 0});

        auto model = create_wave_eqn_abl_test_model<2>(geom);

        tsap::RK4Integrator<decltype(model)> integrator(std::move(model), {128, 128, 128});

        integrator.initialise();

        integrator.configureExporter("raw/test", 20);

        size_t steps_total = 5000;
        for (size_t step = 0; step <= steps_total; step++) {
            integrator.step(sqrt(0.5) * geom.CellSize(0));
            amrex::AllPrint() << "step " << step << " done" << std::endl;
        }
    }
    amrex::Finalize();
}