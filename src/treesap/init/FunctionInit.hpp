#pragma once
#include "../pde-model/PDEModelConcepts.hpp"

#include <AMReX_Array.H>
#include <AMReX_Parser.H>

#include <string>
#include <utility>

namespace treesap {

/**
 * @brief Initializes components using string expressions parsed at run time.
 * @tparam ncomp_ Number of components to initialize.
 */
template <int ncomp_> struct FunctionInit {
    static_assert(ncomp_ > 0, "FunctionInit must initialise a positive number of components.");

    /// Number of components.
    static constexpr int ncomp = ncomp_;

    FunctionInit() = delete;

    /**
     * @brief Constructor creates host parsers and extracts the compiled executors.
     * @param exprs_ Array of string expressions for each component.
     */
    explicit FunctionInit(amrex::Array<std::string, ncomp> exprs_)
        : exprs(std::move(exprs_)), parsers(std::make_shared<amrex::Array<amrex::Parser, ncomp>>()) {
        for (int i = 0; i < ncomp; ++i) {
            amrex::Parser& parser = (*parsers)[i];
            parser.define(exprs[i]);
            parser.registerVariables({"x", "y", "z"});
            executors[i] = parser.compile<3>();
        }
    }

    /**
     * @brief Device-friendly call operator (evaluates the compiled parser executor).
     * @param x X coordinate.
     * @param y Y coordinate.
     * @param z Z coordinate.
     * @param comp Component index.
     * @return Evaluated real value.
     */
    AMREX_GPU_HOST_DEVICE
    amrex::Real operator()(amrex::Real x, amrex::Real y, amrex::Real z, int comp) const {
        return executors[comp](x, y, z);
    }

  private:
    amrex::Array<std::string, ncomp> exprs; ///< String expressions for each component.
    std::shared_ptr<amrex::Array<amrex::Parser, ncomp>>
        parsers; ///< Shared pointer because parsers need to be heap-allocated to allow for copy.
    amrex::GpuArray<amrex::ParserExecutor<3>, ncomp> executors; ///< Compiled executors for execution.
};

// Verify that FunctionInit satisfies the initialization concept.
static_assert(InitConcept<FunctionInit<1>>);

} // namespace treesap