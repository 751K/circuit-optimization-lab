//! Dense MNA primitives shared by the Rust solver paths.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    pub kind: i64,
    pub reference: usize,
    pub value: f64,
}

impl Term {
    pub fn resolve(self, state: &[f64], inputs: &[f64]) -> Option<f64> {
        match self.kind {
            0 => state.get(self.reference).copied(),
            1 => inputs.get(self.reference).copied(),
            _ => Some(self.value),
        }
    }

    /// Validate a compiled terminal before it reaches an indexing hot path.
    pub fn is_valid(self, state_size: usize, allow_input: bool) -> bool {
        match self.kind {
            0 => self.reference < state_size,
            1 => allow_input,
            2 => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct DenseSystem {
    size: usize,
    pub residual: Vec<f64>,
    pub jacobian: Vec<f64>,
}

impl DenseSystem {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            residual: vec![0.0; size],
            jacobian: vec![0.0; size * size],
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn add_jacobian(&mut self, row: usize, col: usize, value: f64) {
        self.jacobian[row * self.size + col] += value;
    }

    pub fn stamp_gmin(&mut self, state: &[f64], node_count: usize, gmin: f64) {
        for (row, voltage) in state.iter().copied().take(node_count).enumerate() {
            self.residual[row] -= voltage * gmin;
            self.add_jacobian(row, row, -gmin);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stamp_resistor(
        &mut self,
        state: &[f64],
        inputs: &[f64],
        a: Term,
        b: Term,
        ai: Option<usize>,
        bi: Option<usize>,
        conductance: f64,
    ) -> bool {
        let Some(va) = a.resolve(state, inputs) else {
            return false;
        };
        let Some(vb) = b.resolve(state, inputs) else {
            return false;
        };
        let current = (va - vb) * conductance;
        if let Some(row) = ai {
            self.residual[row] -= current;
            self.add_jacobian(row, row, -conductance);
            if let Some(col) = bi {
                self.add_jacobian(row, col, conductance);
            }
        }
        if let Some(row) = bi {
            self.residual[row] += current;
            self.add_jacobian(row, row, -conductance);
            if let Some(col) = ai {
                self.add_jacobian(row, col, conductance);
            }
        }
        true
    }

    pub fn stamp_current_source(&mut self, pi: Option<usize>, qi: Option<usize>, value: f64) {
        if let Some(row) = pi {
            self.residual[row] -= value;
        }
        if let Some(row) = qi {
            self.residual[row] += value;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stamp_capacitor(
        &mut self,
        state: &[f64],
        inputs: &[f64],
        a: Term,
        b: Term,
        ai: Option<usize>,
        bi: Option<usize>,
        capacitance: f64,
        inv_h: f64,
        bdf: [f64; 3],
        previous_dv: f64,
        previous2_dv: f64,
    ) -> bool {
        let Some(va) = a.resolve(state, inputs) else {
            return false;
        };
        let Some(vb) = b.resolve(state, inputs) else {
            return false;
        };
        let current = capacitance
            * inv_h
            * (bdf[0] * (va - vb) + bdf[1] * previous_dv + bdf[2] * previous2_dv);
        let gc = bdf[0] * capacitance * inv_h;
        if let Some(row) = ai {
            self.residual[row] -= current;
            self.add_jacobian(row, row, -gc);
            if let Some(col) = bi {
                self.add_jacobian(row, col, gc);
            }
        }
        if let Some(row) = bi {
            self.residual[row] += current;
            self.add_jacobian(row, row, -gc);
            if let Some(col) = ai {
                self.add_jacobian(row, col, gc);
            }
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stamp_voltage_source(
        &mut self,
        state: &[f64],
        inputs: &[f64],
        a: Term,
        b: Term,
        pi: Option<usize>,
        qi: Option<usize>,
        branch: usize,
        emf: f64,
    ) -> bool {
        let Some(va) = a.resolve(state, inputs) else {
            return false;
        };
        let Some(vb) = b.resolve(state, inputs) else {
            return false;
        };
        let branch_current = state[branch];
        if let Some(row) = pi {
            self.residual[row] -= branch_current;
            self.add_jacobian(row, branch, -1.0);
            self.add_jacobian(branch, row, 1.0);
        }
        if let Some(row) = qi {
            self.residual[row] += branch_current;
            self.add_jacobian(row, branch, 1.0);
            self.add_jacobian(branch, row, -1.0);
        }
        self.residual[branch] = va - vb - emf;
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stamp_vcvs(
        &mut self,
        state: &[f64],
        inputs: &[f64],
        a: Term,
        b: Term,
        cp: Term,
        cn: Term,
        pi: Option<usize>,
        qi: Option<usize>,
        cpi: Option<usize>,
        cni: Option<usize>,
        branch: usize,
        mu: f64,
    ) -> bool {
        if !self.stamp_voltage_source(state, inputs, a, b, pi, qi, branch, 0.0) {
            return false;
        }
        let Some(vcp) = cp.resolve(state, inputs) else {
            return false;
        };
        let Some(vcn) = cn.resolve(state, inputs) else {
            return false;
        };
        self.residual[branch] -= mu * (vcp - vcn);
        if let Some(col) = cpi {
            self.add_jacobian(branch, col, -mu);
        }
        if let Some(col) = cni {
            self.add_jacobian(branch, col, mu);
        }
        true
    }

    pub fn stamp_cccs(
        &mut self,
        state: &[f64],
        pi: Option<usize>,
        qi: Option<usize>,
        control_branch: usize,
        beta: f64,
    ) {
        let current = beta * state[control_branch];
        if let Some(row) = pi {
            self.residual[row] += current;
            self.add_jacobian(row, control_branch, beta);
        }
        if let Some(row) = qi {
            self.residual[row] -= current;
            self.add_jacobian(row, control_branch, -beta);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stamp_ccvs(
        &mut self,
        state: &[f64],
        inputs: &[f64],
        a: Term,
        b: Term,
        pi: Option<usize>,
        qi: Option<usize>,
        branch: usize,
        control_branch: usize,
        gamma: f64,
    ) -> bool {
        if !self.stamp_voltage_source(state, inputs, a, b, pi, qi, branch, 0.0) {
            return false;
        }
        self.residual[branch] -= gamma * state[control_branch];
        self.add_jacobian(branch, control_branch, -gamma);
        true
    }
}

/// Solve `matrix * x = -rhs` using the historical in-place GEPP algorithm.
pub fn solve_dense_neg_rhs_in_place(matrix: &mut [f64], rhs: &mut [f64]) -> bool {
    let n = rhs.len();
    if matrix.len() != n * n {
        return false;
    }
    for value in rhs.iter_mut() {
        *value = -*value;
    }
    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = matrix[k * n + k].abs();
        for row in (k + 1)..n {
            let value = matrix[row * n + k].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if pivot_abs == 0.0 || !pivot_abs.is_finite() {
            return false;
        }
        if pivot != k {
            for col in k..n {
                matrix.swap(k * n + col, pivot * n + col);
            }
            rhs.swap(k, pivot);
        }
        let diagonal = matrix[k * n + k];
        for row in (k + 1)..n {
            let factor = matrix[row * n + k] / diagonal;
            if factor != 0.0 {
                matrix[row * n + k] = 0.0;
                for col in (k + 1)..n {
                    matrix[row * n + col] -= factor * matrix[k * n + col];
                }
                rhs[row] -= factor * rhs[k];
            }
        }
    }
    for row in (0..n).rev() {
        let mut accumulator = rhs[row];
        for col in (row + 1)..n {
            accumulator -= matrix[row * n + col] * rhs[col];
        }
        let diagonal = matrix[row * n + row];
        if diagonal == 0.0 || !diagonal.is_finite() {
            return false;
        }
        rhs[row] = accumulator / diagonal;
        if !rhs[row].is_finite() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solved(reference: usize) -> Term {
        Term {
            kind: 0,
            reference,
            value: 0.0,
        }
    }

    fn rail(value: f64) -> Term {
        Term {
            kind: 2,
            reference: 0,
            value,
        }
    }

    #[test]
    fn terminal_resolution_covers_all_kinds() {
        let state = [3.0, 4.0];
        let inputs = [5.0];
        assert_eq!(solved(1).resolve(&state, &inputs), Some(4.0));
        assert_eq!(
            Term {
                kind: 1,
                reference: 0,
                value: 0.0
            }
            .resolve(&state, &inputs),
            Some(5.0)
        );
        assert_eq!(rail(6.0).resolve(&state, &inputs), Some(6.0));
    }

    #[test]
    fn gepp_matches_known_solution_and_pivots() {
        let mut matrix = vec![0.0, 2.0, 1.0, 1.0];
        let mut rhs = vec![-4.0, -3.0];
        assert!(solve_dense_neg_rhs_in_place(&mut matrix, &mut rhs));
        assert_eq!(rhs, vec![1.0, 2.0]);
    }

    #[test]
    fn passive_stamp_uses_historical_residual_signs() {
        let state = [2.0];
        let mut system = DenseSystem::new(1);
        assert!(system.stamp_resistor(&state, &[], solved(0), rail(0.0), Some(0), None, 0.5));
        system.stamp_current_source(None, Some(0), 0.25);
        assert_eq!(system.residual, vec![-0.75]);
        assert_eq!(system.jacobian, vec![-0.5]);
    }

    #[test]
    fn voltage_source_stamp_builds_augmented_rows() {
        let state = [1.5, -0.2];
        let mut system = DenseSystem::new(2);
        assert!(system.stamp_voltage_source(
            &state,
            &[],
            solved(0),
            rail(0.0),
            Some(0),
            None,
            1,
            1.0,
        ));
        assert_eq!(system.residual, vec![0.2, 0.5]);
        assert_eq!(system.jacobian, vec![0.0, -1.0, 1.0, 0.0]);
    }
}
