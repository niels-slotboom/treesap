### `PDEModelConcepts.hpp`

`pde-model/PDEModelConcepts.hpp` defines the compile-time C++20 concepts that govern the architecture of the TrEESaP library, ensuring that user-provided models and operators conform to required type and interface specifications.

* **`RHSConcept`**: Verifies that a spatial right-hand side operator type is trivially copyable, exposes valid compile-time constants `ncomp > 0` and `ngrow >= 0`, and implements a call operator taking grid indices `(i, j, k)`, a component index, simulation time, and an AMReX data array view (`amrex::Array4<const amrex::Real>`), returning a type convertible to `amrex::Real`.


* **`BCConcept`**: Verifies that a boundary condition filling type is trivially copyable, exposes `ncomp` and `ngrow`, and implements a call operator taking an index vector `iv`, a destination data array view, component ranges, GPU-safe geometry data, time, and AMReX boundary condition records (`amrex::BCRec`), returning `void`.


* **`InitConcept`**: Verifies that an initial condition evaluator is copy-constructible, exposes `ncomp > 0`, and implements a call operator taking physical spatial coordinates `(x, y, z)` and a component index, returning a value convertible to `amrex::Real`.


* **`PDEModelConcept`**: Verifies that a complete PDE model type defines the required component and ghost cell properties alongside nested types `rhs_t`, `init_t`, and `bc_t`. It mandates that these nested types satisfy `RHSConcept`, `InitConcept`, and `BCConcept` respectively, and requires accessor methods for the RHS, initial conditions, boundary condition logic, boundary condition records (`amrex::BCRec`), geometry (`amrex::Geometry`), and component names.

---

### `PDEModel.hpp`

`pde-model/PDEModel.hpp` provides the explicit general implementation of a partial differential equation model conforming to `PDEModelConcept`.

* **Assumed PDE Form**: The library assumes systems of PDEs take the semi-discrete explicit evolution form $\partial_t u = F(u, t)$, where $u$ represents the state variables and $F$ denotes the right-hand side operator.
* **Template Parameters & Defaults**: The class is templated over `RHSType`, `InitType`, and `BCType`. `BCType` defaults to `NoOpBC<RHSType::ncomp, RHSType::ngrow>` if omitted. Static assertions ensure that component counts and ghost cell dimensions match across components (`InitType::ncomp == ncomp`, `BCType::ncomp == ncomp`, and `BCType::ngrow == ngrow`).


* **User-Provided Components**: Users supply everything required to define a complete PDE problem via the constructor:
    * Component names (`amrex::Vector<std::string>`)


    * Domain geometry (`amrex::Geometry`)


    * Bulk PDE RHS operator (`RHSType`)


    * Initial condition evaluator (`InitType`)


    * Boundary condition handler (`BCType`, with a default constructor option)


    * Boundary condition records (`amrex::Vector<amrex::BCRec>`, which default internally to periodic / internal directions `amrex::BCType::int_dir`)




* **Accessors**: Provides uniform accessor methods (`get_rhs()`, `get_init()`, `get_bc()`, `get_bc_recs()`, `get_geometry()`, and `get_comp_names()`) to retrieve the underlying model components.

---

### `ABLDecorator.hpp`

`pde-model/rhs-decorators/ABLDecorator.hpp` implements an absorbing boundary layer (ABL) decorator that wraps any valid RHS operator and transforms it by adding a relaxation/damping term near the domain boundaries. Mathematically, it modifies the evolution equation from $\partial_t u = F(u, t)$ to:

$$\partial_t u = F(u, t) - \sigma(\mathbf{x})(u - u_{\text{target}})$$

where:

* $F(u, t)$ is the underlying bulk right-hand side operator provided by `RHSType`.

* $\sigma(\mathbf{x})$ is the spatial damping coefficient, which evaluates to $0$ inside the interior "bulk" region and increases smoothly toward the domain boundaries based on a profile function (`ABLProfileConcept`) scaled by the layer thickness and maximum strength.

* $u_{\text{target}}$ represents the target field values (`target_values`) towards which the solution is nudged inside the boundary layer.

---

### `KODecorator.hpp`

`pde-model/rhs-decorators/KODecorator.hpp` implements a Kreiss-Oliger dissipation decorator that wraps any valid RHS operator and augments it with high-frequency numerical dissipation to stabilize hyperbolic PDE simulations. Mathematically, it transforms the evolution equation from $\partial_t u = F(u, t)$ to:

$$\partial_t u = F(u, t) + \sigma_{\text{KO}} \Delta x^{-1} \sum_{d \in \{x,y,z\}} D_{\text{KO}}^{(d)} u$$

where:

* $F(u, t)$ is the underlying bulk right-hand side operator provided by `RHSType`.

* $\sigma_{\text{KO}}$ is the dissipation `strength` factor.

* $\Delta x^{-1}$ is the inverse grid spacing (`dx_inv`).

* $D_{\text{KO}}^{(d)}$ represents the Kreiss-Oliger dissipation difference operator applied along spatial dimension $d$, utilizing the configured ghost cell width (`ngrow`) for stencil evaluation.

---
### `ABLPDEModel.hpp`

`pde-model/ABLPDEModel.hpp` provides a specialized implementation of `PDEModel` designed for simulations that require an absorbing boundary layer (ABL) at the domain edges.

* **Model Specialization**: It inherits from and configures a base `PDEModel` where:
    * The underlying right-hand side (`RHSType`) is automatically wrapped in an `ABLDecorator<RHSType, ABLProfileType>`.

    * The boundary condition handler (`BCType`) is set to `DirichletBC`, configured directly with the target damping values.


* **`ABLData` Struct**: Encapsulates all necessary configuration parameters for the absorbing boundary layer:
    * `target_values`: An `amrex::GpuArray` specifying the target field values for each component toward which the solution is relaxed.

    * `thickness`: The integer width of the boundary layer measured in grid cells.

    * `inv_thickness`: A precomputed inverse layer thickness ($1 / \text{thickness}$) used for efficient scaling.
    
    * `max_strength`: The maximum amplitude factor for the damping operator.

    * `profile`: A constexpr profile function adhering to `ABLProfileConcept` that transitions smoothly from $0.0$ at the inner bulk interface to $1.0$ at the outer domain boundary.

---

### `Integrator.hpp`

`Integrator.hpp` defines the abstract base class for time integration of partial differential equations in the TrEESaP library, managing grid data structures, domain partitioning, ghost cell communication, and the execution loop.

* **Template Constraint**: Templated over `PDEModelType`, requiring it to satisfy `PDEModelConcept`.

* **State Management**: Maintains AMReX `MultiFab` containers (`state_old` and `state_new`) configured with the model's component count (`ncomp`) and ghost cell width (`ngrow`), distributed across a partitioned `BoxArray` (`ba`) and `DistributionMapping` (`dm`).

* **Public Interface**:
    * **Constructor**: Initializes the PDE model instance, creates a partitioned box array based on a user-defined block size, sets up distribution mapping, and allocates the state multi-fabs.

    * `configureExporter(path, export_interval)`: Configures the optional `PlotExporter` to output simulation data at specified step intervals.

    * `initialise()`: Evaluates the model's initial condition functor across the physical coordinates of all valid grid cells to populate `state_old`.

    * `step(delta_time)`: Advances the simulation by a single time step (`delta_time`) by invoking boundary patch fills, triggering data exports, executing the concrete state update, swapping old and new states, and updating the step counter and simulation time.


* **Protected & Virtual Interface**:
    * `computeNewState(delta_time)`: A pure virtual function (`= 0`) that derived integration schemes (e.g., explicit Runge-Kutta or Euler) must implement to compute the new state from the old state.

    * `FillPatch(mf, time_)`: Updates internal and external ghost regions using AMReX's single-level fill patch utilities and physical boundary condition functions (`PhysBCFunct`), driven by the model's boundary condition object and records.

---

### `RK4Integrator.hpp`

`integrators/RK4Integrator.hpp` implements the explicit classical 4th-order Runge-Kutta (RK4) time-integration scheme, inheriting the base infrastructure and interface from `Integrator`.

* **Mathematical Formulation**: Employs standard RK4 Butcher tableau coefficients for its four stages:
    * Stage time fraction offsets (`a_fraction_arr`): $0.0, 0.5, 0.5, 1.0$.

    * Stage accumulation weights (`b_fraction_arr`): $1/6, 1/3, 1/3, 1/6$.


* **Performance-Critical Details**:
    * **Compile-Time Specialization**: Uses template metaprogramming (`template <int stage_number>`) combined with `constexpr` conditionals (`is_first`, `is_last`) to eliminate runtime branching overhead across the four distinct integration stages.

    * **Kernel Fusion**: Fuses spatial RHS evaluation, accumulation into the new state container (`acc_arr`), and preparation of the argument for subsequent stages (`arg_next_arr`) into a single unified `amrex::ParallelFor` device kernel per `MFIter` block, maximizing arithmetic intensity and minimizing global memory roundtrips.

    * **Zero-Allocation Register Management**: Utilizes dedicated internal `amrex::MultiFab` temporary registers (`arg`, `arg_next`) with pointer/reference swaps to manage intermediate stage states without dynamic memory reallocations during time-stepping loops.

--- 
### `EulerIntegrator.hpp`

`integrators/EulerIntegrator.hpp` implements a first-order explicit forward Euler time-integration scheme. It computes the new state via the standard update formula $u^{n+1} = u^n + \Delta t F(u^n, t^n)$ using a single `amrex::ParallelFor` kernel across the valid grid boxes.

Ready for `stencil.hpp` when you are.

---

### 9. `stencil.hpp`

`utils/stencil.hpp` provides compile-time resolved, fully unrolled finite-difference derivative stencils and differential operators for AMReX data arrays, operating without built-in grid scaling.

* **Grid Scaling & Assumptions**:
    * The stencils strictly assume a uniform grid spacing in all directions ($\Delta x = \Delta y = \Delta z$).

    * The implementation provides raw compile-time unrolled stencils **without grid scaling**. The user is entirely responsible for applying the appropriate scaling factor by dividing the result by the correct power of the grid spacing ($\Delta x^{-k}$ for a derivative of order $k$).


* **Relevant User-Facing Calls**:
    * `stencil::derivative<ngrow, Dirs...>(i, j, k, comp, arr)` (and overload taking `amrex::IntVect`): Computes arbitrary-order partial derivatives (up to third derivatives) given compile-time coordinate direction templates (`dir::x`, `dir::y`, `dir::z`) and ghost cell width `ngrow`.

    * `stencil::gradient<ngrow>(i, j, k, comp, arr)`: Computes the gradient vector of a field component, returning an `amrex::GpuArray<amrex::Real, 3>`.

    * `stencil::gradient_squared<ngrow>(i, j, k, comp, arr)`: Computes the squared magnitude of the gradient vector.

    * `stencil::laplacian<ngrow>(i, j, k, comp, arr)`: Computes the Laplacian operator via the sum of unmixed second-order partial derivatives.

    * `stencil::hessian<ngrow>(i, j, k, comp, arr)`: Computes the full second-order Hessian matrix, returning an `amrex::Array2D`.

    * `stencil::kreiss_oliger<ngrow, dir>(i, j, k, comp, arr)`: Computes the Kreiss-Oliger dissipation term along a specified coordinate direction using the configured ghost width.

---

### `FunctionInit.hpp`

`init/FunctionInit.hpp` provides a flexible initial condition functor that evaluates user-supplied mathematical string expressions at runtime using AMReX's parser infrastructure.

* **Purpose**: Enables configuring initial field states using readable mathematical expression strings (e.g., `"sin(pi*x)*exp(-y*y)"`) instead of hardcoded C++ lambdas, compiling them into efficient device-compatible executors.

* **Template Parameter**: `ncomp_` specifies the positive number of components to initialize.

* **Implementation Details**:
    * **Parser Allocation**: Uses a `std::shared_ptr` wrapping an array of `amrex::Parser` objects to handle heap allocation and ensure copyability across tasks and execution scopes.
    
    * **Variable Registration**: Automatically registers spatial coordinate variables `{"x", "y", "z"}` for each component expression string.

    * **GPU Execution**: Compiles each parser into an `amrex::ParserExecutor<3>` stored within a GPU-safe `amrex::GpuArray`, enabling high-performance evaluation inside device kernels via the `AMREX_GPU_HOST_DEVICE` call operator.
