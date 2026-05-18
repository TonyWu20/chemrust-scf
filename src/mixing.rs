use crate::types::Density;

/// Stores past densities for charge mixing (Pulay/DIIS in later phases).
#[derive(Debug, Clone)]
pub struct DensityHistory {
    densities: Vec<Density>,
    max_history: usize,
}

impl DensityHistory {
    pub fn new(max_history: usize) -> Self {
        Self {
            densities: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Mix a new density with stored history.
    ///
    /// Returns `(mixed_density, input_snapshot)`:
    /// - `mixed_density`: output density for the next iteration.
    /// - `input_snapshot`: clone of the input, stored as
    ///   `previous_density` for convergence delta: `||ρ_new - ρ_old|| < tol`.
    pub fn mix(&mut self, density: Density) -> (Density, Density) {
        let _ = (density, self.max_history);
        todo!()
    }

    /// Number of mixing iterations performed so far.
    pub fn iterations(&self) -> usize {
        self.densities.len()
    }
}
