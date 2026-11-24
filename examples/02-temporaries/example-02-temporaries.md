# Example 02 - Temporaries {#example-02-temporaries}

In this example, we consider tensor temporaries (`treesap::tensor::TensorAliasBase::Temporary`), explaining what they are, how they relate to tensor aliases, and when they should be used.

## What Are Temporaries?

To understand temporaries, we should first recall what a tensor alias — that is, a `Derived` implementation of `treesap::tensor::TensorAliasBase` — represents. A tensor alias does not own any memory; rather, it provides a *view* into an existing `amrex::Array4` reference. It abstracts away the conversion between tensor indices and the corresponding `Array4` component indices, with these conversions being evaluable at compile time.

Consequently, a tensor alias can only refer to data that is already stored somewhere in memory. In practice, however, we often want to store intermediate tensorial quantities without writing them back to the underlying grid data. Consider, for example, the Christoffel symbols of a metric. Typically, the metric itself is an evolution variable, whereas its Christoffel symbols are derived quantities. We can compute the latter on the fly during an RHS evaluation, but may still need to retain the resulting values for subsequent operations or RHS component evaluations. This calls for a separate storage mechanism.

This is precisely what a *temporary* provides. Conceptually, we want an object that:

- owns its own memory, sufficient to store exactly one instance of the tensor;
- supports the same tensor-indexing convention as its corresponding alias;
- can be used independently of the underlying `amrex::Array4` data.

The `Temporary` member struct of `treesap::tensor::TensorAliasBase` provides this functionality. Every implementation of `TensorAliasBase` inherits its corresponding `::Temporary` type, which provides storage and indexing consistent with the alias itself. Since `::Temporary` is a member type inherited by any `Derived` implementation, a custom alias implementation automatically provides an associated `::Temporary` type.

In summary, **an alias provides a view into existing tensor data, whereas its associated temporary provides owned storage for an independent tensorial value.** The two share the same relative indexing convention, but differ in where their data comes from and who owns it.

## The Code Example

The file `examples/02-temporaries/NonLinearRHS.hpp` implements a right-hand side functor for the quasi-linear wave equation

$$
    \partial_t^2 \phi= \Delta \phi - \gamma |\nabla \phi|^2 \partial_t \phi.
$$

To see why this example is useful for illustrating temporaries, we first consider its first-order-in-time formulation,

$$
    \begin{aligned}
        \partial_t \phi &= \pi, \\
        \partial_t \pi &= \Delta \phi - \gamma |\nabla\phi|^2\pi,
    \end{aligned}
$$

obtained by introducing the auxiliary variable \f$\pi = \partial_t \phi\f$.

Two quantities in this formulation are particularly interesting from the perspective of intermediate storage. Firstly, the local field value of \f$\pi\f$ appears in both RHS components. We would therefore like to load it from global memory only once and retain it in a temporary for subsequent reuse. Secondly, the gradient \f$\nabla\phi\f$ enters through its squared norm, \f$|\nabla\phi|^2\f$. We can evaluate the gradient once into a `treesap::tensor::VectorAlias::Temporary` and subsequently compute its squared norm, rather than relying on compiler optimisations to eliminate duplicate stencil evaluations in an expression such as

```cpp
amrex::Real grad = contract([&]<int a>(){
    return derivative<ngrow,a>(i,j,k,phi,{}) * derivative<ngrow,a>(i,j,k,phi,{});
});
```

Let us now examine the relevant aspects of the implementation.

### Setting Up the RHS

For the most part, setting up the right-hand side follows the same strategy as the one shown in the first code example (cf. @subpage example-01-wave-equation). Since, however, we use the alias types both to define the `phi` and `pi` instances and to introduce their associated temporaries, we declare them as type aliases in the struct body:

```cpp
template <int ngrow_> struct NonLinearRHS {
    static constexpr int ncomp = 2;
    static constexpr int ngrow = ngrow_;

    // alias types must be defined outside of the operator() scope to be available for temporary definitions
    using PhiAliasType = treesap::tensor::ScalarAlias<0>;
    using PiAliasType = treesap::tensor::ScalarAlias<PhiAliasType::next>;

    // ensure that we have the correct number of components
    static_assert(PiAliasType::next == ncomp); 
};
```

In addition, we add the two relevant simulation parameters to the struct body, together with a constructor to initialise them:

```cpp
amrex::Real dx_inv_sq;
amrex::Real gamma;

AMREX_GPU_HOST_DEVICE explicit NonLinearRHS(const amrex::GeometryData& geom, amrex::Real gamma)
    : dx_inv_sq(1.0 / (geom.CellSize(0) * geom.CellSize(0))), gamma(gamma) {}
```

In further analogy to @subpage example-01-wave-equation, we add the standard tensor alias setup to the `operator()` implementation:

```cpp
template <int comp>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(int i, int j, int k, amrex::Real t,
                                                                const amrex::Array4<const amrex::Real>& arr) const {
    // import namespaces for more concise code
    using namespace treesap;
    using namespace tensor;
    using namespace stencil;

    // set up the aliases for field variables:
    PhiAliasType phi{{arr}}; // component range {0}
    PiAliasType pi{{arr}};   // component range {1}

    // constexpr if to decide which field comp belongs to
    if constexpr (phi.contains(comp)) { 
        // phi_rhs implementation...
    } else if constexpr (pi.contains(comp)) { 
        // pi_rhs implementation...
    } else {
        return NAN; // this should never be reached
    }
}
```

### The \f$\pi\f$ Temporary

Since \f$\pi\f$ appears in both the RHS for \f$\partial_t \phi\f$ and that for \f$\partial_t \pi\f$, we would like to load it from global memory only once and store it in a temporary for subsequent reuse. There is, however, a complication: `operator()` is invoked separately for each RHS component (once for `comp = 0`, and again for `comp = 1`). Consequently, the temporary must survive the scope boundary between these invocations.

A natural first instinct might be to declare a `static` variable within the `operator()` scope. This would be unsuitable, however, since multiple GPU threads could end up accessing the same memory location. Instead, we store the temporary as a member of the `NonLinearRHS` struct. Under the execution model employed by the TreESaP integrators, each thread operates on its own copy of the RHS functor, allowing the member to serve as per-thread scratch storage.

There is one additional consideration: the `treesap::RHSConcept` contract requires the `operator()` implementation to be `const`. This ensures that its invocation cannot modify the simulation parameters stored in the functor, such as \f$\gamma\f$. Since our scratch temporary must nevertheless be modified, we declare it `mutable`.

Concretely, we add the following member to the struct:

```cpp
 mutable PiAliasType::Temporary pi_temp;
``` 

Our `operator()` implementation must initialise it during the first invocation, so that its value can be reused in subsequent ones. All TreESaP integrators evaluate RHS components starting from `comp = 0`, allowing us to do this by adding the following block to the `operator()` body, immediately after initialising the aliases `phi` and `pi`:

```cpp
// prepare scratch state temporaries if comp == 0, i.e. the first operator() evaluation
if constexpr (comp == 0) {
    pi_temp = pi.load(i, j, k); // load pi value from memory
                                // just once, so it can be reused for
                                // the comp = 0 and comp = 1 evaluation.
}
```

For a tensor alias, the `.load(i,j,k)` method loads the entire tensor at the given grid point \f$(i,j,k)\f$ into its associated temporary. In principle, for a scalar quantity, one could simply use an `amrex::Real` as a temporary variable. For higher-rank tensors, however, this quickly becomes cumbersome, and it is preferable to use `TensorAliasBase::Temporary` uniformly for all tensorial quantities.

Having loaded the local value of \f$\pi\f$ into `pi_temp`, we can now use it to evaluate the RHS of \f$\partial_t \phi\f$ by amending its `if constexpr` branch as follows:

```cpp
if constexpr (phi.contains(comp)) { // phi_rhs implementation
    return pi_temp({});             // = pi(i,j,k) as stored in temporary
} 
```

The `{}` argument is provided because a `Temporary` expects tensor indices to be passed, whereas a scalar has none — hence the empty `treesap::tensor::IndexArray`. We will access `pi_temp` in the same way when implementing the RHS of \f$\partial_t \pi\f$ below.

### The \f$\nabla\phi\f$ Temporary

The evaluation of the right-hand side of
$$
    \partial_t \pi = \Delta \phi - \gamma |\nabla\phi|^2\pi
$$
requires us to compute the squared norm of the gradient, \f$|\nabla\phi|^2 = \nabla\phi \cdot \nabla\phi\f$. To avoid evaluating the same derivative stencils twice, once for each factor in the inner product, we can store the gradient in a `treesap::tensor::VectorAlias::Temporary` and subsequently compute its squared norm.

More concretely, this is carried out by adding the following to the `pi_rhs` branch body:

```cpp
VectorAlias<>::Temporary phi_grad{};  // temporary to hold grad(phi)
static_for<3>([&]<int a>() {
     phi_grad({a}) = derivative<ngrow, a>(i, j, k, phi, {});
});
amrex::Real grad_sq = contract([&]<int a>() { 
    return phi_grad({a}) * phi_grad({a}); 
});
```

In the first line, we instantiate the vector temporary. The subsequent call to `static_for` populates each of its components with the corresponding derivative stencil value. The reason for using `static_for` rather than an ordinary `for` loop is that the directions `a` must be known at compile time in order to be used as template parameters. The call to `static_for<3>` therefore instantiates the supplied lambda for \f$a = 0, 1, 2\f$, effectively unrolling the loop at compile time.

Notice that, unlike `pi_temp`, the gradient temporary is declared locally within `operator()`. This is sufficient because its contents are only needed during the current invocation: there is no need to preserve them across RHS component evaluations. Both temporaries use the same underlying `Temporary` mechanism, but their different lifetimes reflect their different purposes.

Once we have computed the squared norm of the gradient, all that remains is to construct the Laplacian of \f$\phi\f$ by contracting its second derivatives and assemble the final result:

```cpp
amrex::Real laplacian = contract([&]<int a>() { 
    return derivative<ngrow, a, a>(i, j, k, phi, {});
});
return dx_inv_sq * (laplacian - gamma * pi_temp({}) * grad_sq);
```

### Closing Remarks

This example illustrates two complementary uses of tensor temporaries: storing intermediate quantities locally within an RHS evaluation, as demonstrated by \f$\nabla\phi\f$, and retaining scratch state across multiple RHS component evaluations, as demonstrated by \f$\pi\f$. In both cases, the associated `Temporary` provides an indexing interface consistent with its tensor alias while owning its own storage. This makes temporaries particularly useful for intermediate tensorial computations, allowing derived quantities to be stored and reused without introducing additional evolution variables or writing intermediate results to global memory.