// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

#pragma once
#include "../pde-model/PDEModelConcepts.hpp"

#include <AMReX_Array.H>
#include <AMReX_Parser.H>

#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace treesap {

/**
 * @brief InitConcept-satisfying functor which initialises components using string expressions parsed at run time.
 * @tparam ncomp_ Number of components to initialize.
 *
 * @details This class is designed to be trivially copyable so it can be passed directly
 *          into GPU kernels (e.g., via `amrex::ParallelFor`). Because an
 *          `amrex::ParserExecutor` requires its parent `amrex::Parser` to remain alive
 *          and at a stable memory location, the parent parsers are stored in a static
 *          host-side registry. This decouples the parser lifetime from the functor copy
 *          that gets shipped to the device.
 */
template <int ncomp_> struct FunctionInit {
    static_assert(ncomp_ > 0, "FunctionInit must initialise a positive number of components.");

    /// Number of components.
    static constexpr int ncomp = ncomp_;

    FunctionInit() = delete;

    /**
     * @brief Constructor creates host parsers and extracts the compiled executors.
     * @param exprs Array of string expressions for each component.
     *
     * @details Compiles each string expression into an `amrex::Parser` instance, which
     *          is permanently stored in a static host registry to guarantee its lifetime
     *          for the duration of the program. The resulting lightweight `ParserExecutor`
     *          is then stored locally in the class instance.
     */
    explicit FunctionInit(const amrex::Array<std::string, ncomp>& exprs) {
        // Emplace exactly ONE parser array for this FunctionInit instance into the stable registry.
        auto& parsers = get_parser_registry().emplace_back();

        for (int i = 0; i < ncomp; ++i) {
            amrex::Parser& parser = parsers[i];
            parser.define(exprs[i]);
            parser.registerVariables({"x", "y", "z"});
            executors[i] = parser.compile<3>();
        }
    }

    ~FunctionInit() = default;

    /**
     * @brief Device-friendly call operator (evaluates the compiled parser executor).
     * @param x X coordinate.
     * @param y Y coordinate.
     * @param z Z coordinate.
     * @param comp Component index.
     * @return Evaluated real value.
     */
    template <int comp>
    AMREX_GPU_HOST_DEVICE amrex::Real operator()(amrex::Real x, amrex::Real y, amrex::Real z) const {
        return executors[comp](x, y, z);
    }

  private:
    /**
     * @brief Static registry maintaining the lifetime of parent `amrex::Parser` objects.
     *
     * @return std::deque<amrex::Array<amrex::Parser, ncomp>>& Reference to the global registry.
     *
     * @details
     * **Why a static deque?**
     * 1. **Trivial Copiability:** `amrex::Parser` instances contain internal state and pointers
     *    that make them non-trivially-copyable. To pass `FunctionInit` into GPU kernels via
     *    `amrex::ParallelFor`, the struct must be trivially copyable, meaning the bulky
     *    `amrex::Parser` objects cannot live inside the struct itself.
     * 2. **Memory Stability:** An `amrex::ParserExecutor` holds a dependency on the memory address
     *    of the `amrex::Parser` it was compiled from. We use a `std::deque` rather than a
     *    `std::vector` because `std::deque` guarantees that insertions do *not* cause reallocations
     *    that move existing elements, preventing dangling pointers or memory invalidation.
     */
    static auto& get_parser_registry() {
        static std::deque<amrex::Array<amrex::Parser, ncomp>> registry;
        return registry;
    }

    amrex::GpuArray<amrex::ParserExecutor<3>, ncomp> executors; ///< Compiled executors safe for GPU execution.
};

// Verify that FunctionInit satisfies the initialization concept.
static_assert(InitConcept<FunctionInit<1>>);

} // namespace treesap