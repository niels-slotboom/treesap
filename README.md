# TrEESaP

TrEESaP is a C++20 header-only library for defining and evolving explicit partial differential equation (PDE) systems on structured grids using AMReX. It separates a PDE model's right-hand side, initial conditions, boundary conditions and metadata from the time-integration scheme.

The documented model form is

$$
\partial_t u = F(u,t),
$$

where $u$ is the collection of evolved fields and $F$ is the spatial right-hand-side operator. The repository provides forward Euler and classical fourth-order Runge–Kutta (RK4) integrators, reusable finite-difference stencils, boundary-condition and RHS decorators, and an example wave-equation test program. For a more detailed description of the implementation and its components, see the [project documentation](doc.md).
 
## Main ideas

- **Composable PDE models.** A model combines an RHS functor, an initial-condition evaluator, boundary-condition logic, AMReX geometry, component names and boundary-condition records.
- **Compile-time interface checking.** C++20 concepts specify the interfaces expected of RHS, initial-condition and boundary-condition functors.
- **AMReX-backed grid execution.** The integrator manages `MultiFab` state, grid decomposition, ghost cells and patch filling, while kernels operate over AMReX grid boxes.
- **Selectable time integration.** The base `Integrator` defines the shared evolution workflow; concrete schemes implement the state update.
- **Reusable RHS modifiers.** Decorators can add absorbing-boundary-layer damping or Kreiss–Oliger numerical dissipation around an existing RHS.
- **Finite-difference utilities.** Compile-time stencils provide derivative and differential-operator evaluations on AMReX data arrays.

## Repository layout

```text
src/
├── test
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── WaveEqnRHS.hpp
└── treesap
    ├── CMakeLists.txt
    ├── init
    │   ├── ConstInit.hpp
    │   └── FunctionInit.hpp
    ├── Integrator.hpp
    ├── integrators
    │   ├── EulerIntegrator.hpp
    │   └── RK4Integrator.hpp
    ├── pde-model
    │   ├── ABLPDEModel.hpp
    │   ├── bcs
    │   │   ├── DirichletBC.hpp
    │   │   └── NoOpBC.hpp
    │   ├── PDEModelConcepts.hpp
    │   ├── PDEModel.hpp
    │   └── rhs-decorators
    │       ├── ABLDecorator.hpp
    │       ├── KODecorator.hpp
    │       └── NoOpRHS.hpp
    └── util
        ├── PlotExporter.hpp
        └── stencil.hpp
```

### `src/test/`

Contains the test/example executable and its build configuration.

- `CMakeLists.txt` — configures the test target.
- `main.cpp` — program entry point for the test.
- `WaveEqnRHS.hpp` — RHS implementation for the wave-equation example.

### `src/treesap/`

The library implementation.

- `CMakeLists.txt` — configures the TrEESaP library target.

#### `init/` — initial conditions

- `ConstInit.hpp` — initial-condition utility for constant field values.
- `FunctionInit.hpp` — evaluates user-supplied mathematical expressions in the spatial coordinates using AMReX's parser. The expressions are compiled into parser executors suitable for host/device evaluation.

#### `Integrator.hpp` — shared time-integration infrastructure

The base integrator owns the old and new solution states in AMReX `MultiFab` containers, sets up the box decomposition and distribution mapping, evaluates initial conditions, fills ghost regions, optionally exports simulation output, and advances the simulation clock and step counter.

Derived integrators implement `computeNewState(delta_time)`.

#### `integrators/` — time-stepping schemes

- `EulerIntegrator.hpp` — explicit forward Euler:
  $$
  u^{n+1} = u^n + \Delta t\,F(u^n,t^n).
  $$
- `RK4Integrator.hpp` — classical explicit fourth-order Runge–Kutta. Its stages are compile-time specialised, and the implementation fuses RHS evaluation and intermediate-state updates into AMReX device kernels. Temporary `MultiFab` registers are reused between stages.

#### `pde-model/` — PDE definitions and composition

- `PDEModelConcepts.hpp` — defines the C++20 concepts used to check RHS operators, initial-condition evaluators, boundary-condition handlers and complete PDE models.
- `PDEModel.hpp` — general explicit PDE model. It combines the RHS, initial conditions, boundary conditions, geometry, component names and AMReX boundary-condition records. The default boundary-condition type is `NoOpBC` when none is specified.
- `ABLPDEModel.hpp` — specialised model that wraps the RHS in an absorbing-boundary-layer decorator and uses Dirichlet boundary conditions targeting configured field values.

##### `pde-model/bcs/` — boundary conditions

- `DirichletBC.hpp` — boundary-condition handler for prescribed target values, used by the absorbing-boundary-layer model.
- `NoOpBC.hpp` — no-op boundary-condition handler, useful when no custom physical boundary filling is required.

##### `pde-model/rhs-decorators/` — RHS modifiers

- `ABLDecorator.hpp` — wraps an RHS with a spatially varying relaxation term near the domain edges:
  $$
  \partial_t u = F(u,t) - \sigma(\mathbf{x})(u-u_{\mathrm{target}}).
  $$
  The damping coefficient is zero in the interior and increases through the absorbing layer according to a configurable profile.
- `KODecorator.hpp` — wraps an RHS with Kreiss–Oliger dissipation to damp high-frequency numerical modes.
- `NoOpRHS.hpp` — no-op RHS utility.

#### `util/` — utilities

- `PlotExporter.hpp` — simulation-output utility configured through the integrator's `configureExporter(path, export_interval)` interface.
- `stencil.hpp` — compile-time-resolved, unrolled finite-difference operators for AMReX data arrays. The documented operations include arbitrary partial derivatives up to third order, gradient, squared gradient magnitude, Laplacian, Hessian and directional Kreiss–Oliger dissipation.

  **Grid-spacing note:** the stencil routines return unscaled finite differences. Apply the appropriate factor of $\Delta x^{-k}$ for a derivative of order $k$. The implementation assumes uniform spacing, $\Delta x=\Delta y=\Delta z$.

## Typical workflow

1. Define an RHS functor that satisfies `RHSConcept`.
2. Choose an initial-condition evaluator, such as `ConstInit` or `FunctionInit`.
3. Choose a boundary-condition handler and provide the required AMReX geometry and boundary-condition records.
4. Assemble these components into a `PDEModel` (or use `ABLPDEModel` when an absorbing boundary layer is needed).
5. Instantiate an integrator such as `EulerIntegrator` or `RK4Integrator`.
6. Initialise the state, then repeatedly call `step(delta_time)`. Configure `PlotExporter` through the integrator if simulation output is required.

The exact constructor signatures and build options should be taken from the headers and CMake files in the repository.

## Design constraints and assumptions

- The model architecture targets explicit evolution systems of the form $\partial_t u=F(u,t)$.
- RHS and boundary-condition functors are required by the corresponding concepts to be trivially copyable; initial-condition evaluators must be copy-constructible.
- Component counts and ghost-cell widths are checked for consistency at compile time where specified by `PDEModel`.
- The documented stencil routines assume a uniform grid and do not apply grid-spacing scaling themselves.
- Ghost-cell filling and grid-distributed state storage are handled through AMReX infrastructure.

## Build and tests

The repository includes CMake configuration files under `src/treesap/` and `src/test/`. Consult those files and the project's top-level build configuration for the precise configure, build and run commands; those commands are not specified in the supplied documentation.
