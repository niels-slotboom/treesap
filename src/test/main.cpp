#include <AMReX.H>
#include <AMReX_Print.H>

#include "init/ConstInit.hpp"
#include "pde-model/ABLPDEModel.hpp"
#include "pde-model/rhs-decorators/NoOpRHS.hpp"
#include <AMReX.H>

/**
 * @brief Construct the absorbing boundary layer test model.
 * @return Configured ABLPDEModel instance.
 */
inline auto create_abl_test_model(const amrex::Geometry& geom) {
    using RHSType = NoOpRHS<1, 1>;
    using InitType = ConstInit<1>;

    RHSType rhs{};
    InitType init{0.0}; // Initialize field to zero everywhere

    auto profile = [](amrex::Real s) constexpr {
        return s * s * (3.0 - 2.0 * s); // Cubic smoothstep profile
    };

    using DecoratedRHS = ABLDecorator<RHSType, decltype(profile)>;
    typename DecoratedRHS::ABLData abl_data{amrex::GpuArray<amrex::Real, 1>{1.0}, // Target value set to 1.0
                                            8,                                    // Thickness in cells
                                            10.0,                                 // Maximum damping strength
                                            profile};

    return ABLPDEModel<RHSType, InitType, decltype(profile)>({"name"}, geom, rhs, abl_data, init);
}

int main(int argc, char** argv) {
    amrex::Initialize(argc, argv);
    {
        amrex::AllPrint() << "Hello, World!" << std::endl;
    }
    amrex::Finalize();
}