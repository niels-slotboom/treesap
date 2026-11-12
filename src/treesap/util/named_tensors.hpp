// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Niels Slotboom

/**
 * @file
 * @brief Defines the standard named tensors `treesap::tensor::kronecker` (\f$\delta_{ab}\f$) and
 * `treesap::tensor::levi_civita` (\f$\epsilon_{a_1...a_d}\f$) as compile-time evaluated templates.
 */

#pragma once

namespace treesap {
namespace tensor {

/**
 * @brief Kronecker Symbol \f$\delta_{ab}\f$.
 * @tparam a First tensor index.
 * @tparam b Second tensor index.
 *
 * The value of this template follows the standard Kronecker symbol definition reading
 * \f[
 *      \delta_{a b} = \begin{cases}1,&a=b,\\ 0,&a\neq b.\end{cases}
 * \f]
 */
template <int a, int b> constexpr int kronecker = ((a == b) ? 1 : 0);

/**
 * @brief Totally antisymmetric Levi-Civita tensor \f$\varepsilon_{a_1...a_d}\f$ in \f$d=\f$`sizeof...(Indices)`
 * dimensions.
 * @tparam Indices... Index pack for which the Levi-Civita is evaluated.
 *
 * For \f$(a_1,...,a_d)=\f$`Indices...`, the template evaluates to the standard definition
 * \f[
 *      \varepsilon_{a_1...a_d} = \begin{cases}0,&a_i = a_j \text{for any} i \neq j,\\
 *                              1, &\text{if $(a_1,...,a_d)$ is an even permutation of $(1,...,d)$,}\\
 *                              -1, &\text{if $(a_1,...,a_d)$ is an odd permutation of $(1,...,d)$.}
 *                              \end{cases}
 * \f]
 */
template <int... Indices>
constexpr int levi_civita = []() consteval {
    constexpr int d = sizeof...(Indices);
    int indices[] = {Indices...};
    int swaps = 0;

    for (int i = 0; i < d; i++) {
        int a = indices[i];
        for (int j = i + 1; j < d; j++) {
            if (a < 0 || a >= d)
                return 0;
            int b = indices[j];
            if (a == b || b < 0 || b >= d)
                return 0;
            if (a > b) {
                indices[i] = b;
                indices[j] = a;
                swaps++;
            }
        }
    }
    return swaps % 2 ? -1 : 1;
}();

} // namespace tensor
} // namespace treesap