use nalgebra::{Point3, Vector3};
use pyo3::prelude::*;

#[pyclass]
pub struct MolecularGraph {
    #[pyo3(get)]
    pub atomic_numbers: Vec<i64>,
    pub positions: Vec<Point3<f64>>,
}

#[pymethods]
impl MolecularGraph {
    #[new]
    fn new(atomic_numbers: Vec<i64>, coords: Vec<(f64, f64, f64)>) -> Self {
        let positions = coords
            .into_iter()
            .map(|(x, y, z)| Point3::new(x, y, z))
            .collect();

        MolecularGraph {
            atomic_numbers,
            positions,
        }
    }

    /// Finds all neighbors within a cutoff distance (e.g., 1.6 Angstroms)
    /// Returns a list of (atom_i, atom_j, distance)
    fn find_bonds(&self, cutoff: f64) -> Vec<(usize, usize, f64)> {
        let mut bonds = Vec::new();
        let n = self.positions.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let d = nalgebra::distance(&self.positions[i], &self.positions[j]);
                if d <= cutoff {
                    bonds.push((i, j, d));
                }
            }
        }
        bonds
    }
}

#[pymodule]
fn valence(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MolecularGraph>()?;
    Ok(())
}
