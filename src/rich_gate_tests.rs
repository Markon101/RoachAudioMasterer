use super::*;
use crate::{dsp::SpectralTransform, native_audio::Audio};
fn fixture() -> (Gate, Prepared, Damage, Components, Cache, Audio) {
    let input = Audio {
        rate: RATE,
        channels: vec![
            (0..2048)
                .map(|i| {
                    (std::f64::consts::TAU * 900.0 * i as f64 / RATE as f64).sin() as f32 * 0.2
                })
                .collect(),
            (0..2048)
                .map(|i| {
                    (std::f64::consts::TAU * 930.0 * i as f64 / RATE as f64).sin() as f32 * 0.17
                })
                .collect(),
        ],
    };
    let mut target = input.clone();
    for c in 0..2 {
        for (i, x) in target.channels[c].iter_mut().enumerate() {
            *x += 0.025
                * (std::f64::consts::TAU * (7500.0 + c as f64 * 2500.0) * i as f64 / RATE as f64)
                    .sin() as f32;
        }
    }
    let d = Damage {
        cutoff: 6000.0,
        transition: 500.0,
        power: 2.0,
    };
    let p = scene_features::prepare(&input, d, 11);
    let s = Stft::default();
    let ms = target.mid_side();
    let desired: [Vec<C>; 2] = std::array::from_fn(|c| {
        let z = s.analyze(&ms[c]);
        z.data
            .iter()
            .zip(&p.base[c].data)
            .enumerate()
            .map(|(i, (a, b))| {
                if i % BINS > d.cutoff_bin() {
                    (a - b)
                        * (if (i % BINS).is_multiple_of(2) {
                            1.0
                        } else {
                            -1.0
                        })
                        / p.scale[c]
                } else {
                    C::default()
                }
            })
            .collect()
    });
    let mut random = Rng(718);
    let parts: [[Vec<C>; 2]; 4] = std::array::from_fn(|j| {
        std::array::from_fn(|c| {
            desired[c]
                .iter()
                .map(|z| match j {
                    0 => *z * 1.3,
                    1 => C::new(-z.im, z.re) * 0.2,
                    2 => C::new(random.signed(), random.signed()) * 0.002,
                    _ => *z * 0.1,
                })
                .collect()
        })
    });
    let base = std::array::from_fn(|c| {
        parts[0][c]
            .iter()
            .zip(&parts[1][c])
            .zip(&parts[2][c])
            .map(|((a, b), z)| *a + *b + *z)
            .collect::<Vec<_>>()
    });
    let flow = std::array::from_fn(|c| {
        base[c]
            .iter()
            .zip(&parts[3][c])
            .map(|(a, b)| a + b)
            .collect()
    });
    let parts = Components { parts, base, flow };
    let cache = Cache::new(&p, d, &parts);
    let m = Model::new(71);
    let mut f = Model::new(73);
    f.stage = "flow".into();
    f.prior_fingerprint = Some(m.fingerprint());
    (Gate::new(&m, &f, false, 400008), p, d, parts, cache, target)
}
#[test]
fn complete_gate_waveform_gradient_matches_perturbation() {
    let (mut g, p, d, parts, cache, target) = fixture();
    let index = g.head.b2();
    let old = g.head.weights[index];
    let eps = 0.002;
    for penalty in [false, true] {
        let (_, grad) = gradient(&g, &p, d, &parts, &cache, &target, penalty);
        g.head.weights[index] = old + eps;
        let plus = gradient(&g, &p, d, &parts, &cache, &target, penalty).0;
        g.head.weights[index] = old - eps;
        let minus = gradient(&g, &p, d, &parts, &cache, &target, penalty).0;
        g.head.weights[index] = old;
        let fd = (plus - minus) / (2.0 * eps as f64);
        println!(
            "gate gradient penalty={penalty} finite={fd} analytic={}",
            grad[index]
        );
        assert!((fd - grad[index] as f64).abs() < 1e-4 + 0.03 * fd.abs());
        assert!(fd.abs() > 1e-5);
    }
}
#[test]
fn allocation_warm_start_exact_and_full_gradient() {
    let (g, p, d, parts, cache, target) = fixture();
    let mut a = g.expanded(420000);
    assert_eq!(a.head.weights.len(), 1710);
    assert_eq!(
        forward(&g, &p, d, &parts, &cache, 0.5).field,
        forward(&a, &p, d, &parts, &cache, 0.5).field
    );
    for j in [a.head.b2(), a.head.b2() + 4, a.head.b2() + 5] {
        let old = a.head.weights[j];
        a.head.weights[j] = old + 0.13;
        let (_, gr) = gradient(&a, &p, d, &parts, &cache, &target, false);
        let eps = 0.002;
        a.head.weights[j] = old + 0.13 + eps;
        let plus = gradient(&a, &p, d, &parts, &cache, &target, false).0;
        a.head.weights[j] = old + 0.13 - eps;
        let minus = gradient(&a, &p, d, &parts, &cache, &target, false).0;
        let fd = (plus - minus) / (2.0 * eps as f64);
        assert!(
            (fd - gr[j] as f64).abs() < 1e-4 + 0.03 * fd.abs(),
            "allocator j{j} finite{fd} analytic{}",
            gr[j]
        );
        a.head.weights[j] = old;
    }
}
#[test]
fn allocator_actuator_lesion_preserves_disabled_path() -> Result<()> {
    let (g, p, d, parts, cache, _) = fixture();
    let mut g = g.expanded(420000);
    let b = g.head.b2();
    g.head.weights[b + 4] = 0.6;
    g.head.weights[b + 5] = 0.4;
    let f = forward(&g, &p, d, &parts, &cache, 0.5);
    assert_eq!(
        f.field,
        cap_from_forward(&g, &f, &p, d, &parts, &cache).field
    );
    g.cap_boost = [true, false];
    let capped = cap_from_forward(&g, &f, &p, d, &parts, &cache);
    assert_eq!(capped.field, forward(&g, &p, d, &parts, &cache, 0.5).field);
    assert_eq!(capped.gates[1], f.gates[1]);
    assert!(capped.gates[0]
        .iter()
        .flatten()
        .zip(f.gates[0].iter().flatten())
        .all(|(a, b)| a <= b));
    assert_eq!(
        capped.field,
        inference_tiled(&g, &p, d, &parts, 0.5, "cpu")?.field
    );
    Ok(())
}
#[test]
fn component_decomposition_and_tiled_cpu_parity() -> Result<()> {
    let (g, p, d, parts, cache, _) = fixture();
    let a = forward(&g, &p, d, &parts, &cache, 0.5);
    let b = inference_tiled(&g, &p, d, &parts, 0.5, "cpu")?;
    assert_eq!(a.field, b.field);
    let mut m = Model::new(71);
    let mut r = Rng(911);
    let start = m.head.w2();
    for w in &mut m.head.weights[start..] {
        *w = r.signed() * 0.05;
    }
    let mut e = Engine::new(&m, "cpu")?;
    let z = e.deterministic(&p, d, None)?;
    let c = e.components(&p, d)?;
    let sum: [Vec<C>; 2] = std::array::from_fn(|j| {
        c[0][j]
            .iter()
            .zip(&c[1][j])
            .zip(&c[2][j])
            .map(|((a, b), c)| *a + *b + *c)
            .collect()
    });
    assert_eq!(sum, z);
    Ok(())
}
#[cfg(feature = "opencl")]
#[test]
#[ignore = "requires real OpenCL GPU"]
fn gate_nonzero_gpu_parity() -> Result<()> {
    let (mut g, p, d, parts, _, _) = fixture();
    let mut r = Rng(121);
    for w in &mut g.head.weights {
        *w = r.signed() * 0.1;
    }
    for allocator in [false, true] {
        let mut g = if allocator {
            g.expanded(420000)
        } else {
            g.clone()
        };
        let mut r = Rng(122);
        for w in &mut g.head.weights {
            *w = r.signed() * 0.1;
        }
        let a = inference_tiled(&g, &p, d, &parts, 0.7, "cpu")?;
        let b = inference_tiled(&g, &p, d, &parts, 0.7, "opencl")?;
        let max = a
            .field
            .iter()
            .flatten()
            .zip(b.field.iter().flatten())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f32, f32::max);
        println!("gate GPU max={max}");
        assert!(max < 2e-6);
    }
    Ok(())
}
