use criterion::{black_box, criterion_group, criterion_main, Criterion};
// Import your logic from the lib
use valence::MolecularGraph;

fn bench_bond_finding(c: &mut Criterion) {
    // Setup a fake molecule with 1000 atoms
    let atoms = vec![6; 1000];
    let coords = vec![vec![1.0, 1.0, 1.0]; 1000];
    let graph = MolecularGraph::new(atoms, coords);

    c.bench_function("find_bonds_1000_atoms", |b| {
        // black_box prevents the compiler from optimizing the call away
        b.iter(|| graph.find_bonds(black_box(4.5)))
    });
}

criterion_group!(benches, bench_bond_finding);
criterion_main!(benches);
