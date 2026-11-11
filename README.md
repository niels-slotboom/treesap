@mainpage TreESaP: Tremendously Efficient Sandbox for PDEs


# TreESaP Library
_Tremendously Efficient Sandbox for PDEs_

<hr>

## Quick Links
- [Documentation](https://niels-slotboom.github.io/treesap/)
- [GitHub Repository](https://github.com/niels-slotboom/treesap)
- [Code Examples](https://niels-slotboom.github.io/treesap/examples_index.html)

<hr>

## Overview
The **TreESaP** library is a **header-only time integration abstraction for tensorial partial differential equations** (PDEs) built on top of the **AMReX** framework. Concretely, it is built to enable integration of PDEs of the form
\f[
    \partial_t \Phi = F(\Phi,t),
\f]
where \f$\Phi\f$ is the state vector and \f$t\f$ the time variable. Here, \f$F\f$ is a functional which the user implements in the form of a right-hand side functor. The primary goal of this library is to provide **zero-cost, compile-time abstractions** that allow its user to translate almost isomorphically between a PDE's mathematical formulation and its numerical implementation.

The library's core features comprise:
- **PDE Model**: `C++20` concepts and classes which are used to represent (discretisations of) PDEs, such as `treesap::RHSConcept`, `treesap::BCConcept`, `treesap::InitConcept` and `treesap::PDEModel`;

- **Time integrators**: An abstract base class/interface defined by `treesap::Integrator`, as well as concrete implementations `treesap::RK4Integrator` and `treesap::EulerIntegrator`;

- **Statically Unrolled Derivative Stencils**: The `treesap::stencil` namespace implements centered first-, second- and third-order derivative stencils for arbitrary stencil widths, fully unrolled at compile time to remove call overhead while retatining readable code;

- **Statically Resolved Tensor Aliasing**: The `treesap::tensor` namespace provides an aliasing system for tensors, which greatly improves code readability by handling indexing arithmetic of `amrex::MultiFab` components with no incurred runtime overhead;

- **Tensor Operations**: Utilities in `treesap::tensor` implement tensor operations such as symmetrisation and index contraction, resolved at compile time.

<hr>

## Documentation, Source Code and Examples

The Doxygen-based documentation is available [here](https://niels-slotboom.github.io/treesap/) or compiled locally by running `doxygen` in the repository's root directory, which can be obtained [here](https://github.com/niels-slotboom/treesap). 

Alongside the library source code, the [TreESaP GitHub repository](https://github.com/niels-slotboom/treesap) contains code examples which illustrate its intended usage. The code example source files are found under `examples/` relative to the repository's root, with accompanying prose overviews provided [here](https://niels-slotboom.github.io/treesap/examples_index.html).

<hr>

## Usage
Although it is header-only, TreESaP exposes a CMake target, and can be imported into any project using the CMake snippet
```cmake
include(FetchContent)
FetchContent_Declare(
    treesap
    GIT_REPOSITORY https://github.com/niels-slotboom/treesap.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(treesap)
```
Downstream targets must then link against the `treesap` `INTERFACE` target.

Alternatively, the library can be installed as a CMake package and then added to any other project.