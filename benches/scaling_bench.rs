#[cfg(feature = "codspeed")]
use codspeed_criterion_compat as criterion;
#[cfg(not(feature = "codspeed"))]
use criterion::{black_box, BenchmarkId, Criterion};

fn generate_data(size: usize) -> Vec<f32> {
    vec![1.0; size]
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inference Scaling");
    for &size in &[128, 256, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &s| {
            b.iter(|| black_box(generate_data(s)));
        });
    }
    group.finish();
}

criterion::criterion_group!(benches, bench_scaling);
criterion::criterion_main!(benches);
