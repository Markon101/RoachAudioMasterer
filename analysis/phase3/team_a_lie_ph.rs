//! Team A isolated mathematical prototype; no production module imports.
//! Compile tests with rustc --edition=2021 --test this_file.rs -o /tmp/team_a_lie_ph.
//! Fixed H = |z|^2/2; the midpoint storage identity is derived in the linked memo.

pub type Matrix = [[f64; 4]; 4];
/// theta order: 01, 02, 03, 12, 13, 23. Positive theta rotates i toward j.
pub const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Each entry is assigned with its negative counterpart, never independently.
/// Mathematical guarantee holds for every real theta. IEEE NaN/Inf are rejected.
pub fn skew_generator(theta: [f64; 6]) -> Option<Matrix> {
    if !theta.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut j = [[0.0; 4]; 4];
    for (k, &(i, l)) in PAIRS.iter().enumerate() {
        j[i][l] = -theta[k];
        j[l][i] = theta[k];
    }
    Some(j)
}

pub fn legacy_theta(kappa: f64) -> [f64; 6] {
    [16.0, 0.0, kappa, -kappa, 0.0, 32.0]
}

/// Rolling input measurements; caller must supply stereo sub-band measurements.
/// SceneStats supplies the first four, once computed on a bounded rolling view.
#[derive(Clone, Copy)]
pub struct HopFeatures {
    pub high_slope_db_oct: f64,
    pub high_flatness: f64,
    pub sample_headroom_db: f64,
    pub high_jitter_rad: f64,
    pub onset: f64,
    pub active: bool,
    pub side_sub_fraction: f64,
    pub sub_lr_correlation: f64,
    pub air_authority: f64,
    pub rumble_authority: f64,
}

/// Illustrative controls, not fitted thresholds or a validated damage estimator.
/// Arrays describe separate mid and side 4D banks, with fixed storage functions.
pub fn hop_targets(f: HopFeatures) -> Option<([f64; 6], [f64; 6], [f64; 4])> {
    let values = [
        f.high_slope_db_oct,
        f.high_flatness,
        f.sample_headroom_db,
        f.high_jitter_rad,
        f.onset,
        f.side_sub_fraction,
        f.sub_lr_correlation,
        f.air_authority,
        f.rumble_authority,
    ];
    if !values.iter().all(|v| v.is_finite()) {
        return None;
    }
    let c = |v: f64| v.clamp(0.0, 1.0);
    let dull = c((-f.high_slope_db_oct - 6.0) / 6.0);
    let tonal = 1.0 - c(f.high_flatness);
    let headroom = c((f.sample_headroom_db - 1.0) / 5.0);
    let reliable_phase = (-f.high_jitter_rad.max(0.0).powi(2)).exp();
    let air = if f.active {
        c(f.air_authority) * c(f.onset) * dull * tonal * headroom * reliable_phase
    } else {
        0.0
    };
    let rumble = if f.active {
        c(f.rumble_authority)
            * c((f.side_sub_fraction - 0.10) / 0.40)
            * c((0.90 - f.sub_lr_correlation) / 0.90)
    } else {
        0.0
    };
    // Untie legacy 03 and 12: transient->air in mid; sub->sink in side.
    let mid = [16.0, 0.0, 12.0 * air, 0.0, 0.0, 32.0];
    let side = [16.0, 0.0, 0.0, 0.0, 0.0, 32.0 + 12.0 * rumble];
    let side_r = [0.5, 2.0, 1.0 + 8.0 * rumble, 4.0 + 16.0 * rumble];
    Some((mid, side, side_r))
}

pub fn energy(z: [f64; 4]) -> f64 {
    0.5 * z.iter().map(|v| v * v).sum::<f64>()
}

/// Fixed-quartic-H continuous certificate as well as the quadratic special case.
pub fn unforced_storage_rate(z: [f64; 4], theta: [f64; 6], r: [f64; 4], beta: f64) -> Option<f64> {
    if !z.iter().chain(r.iter()).all(|v| v.is_finite())
        || r.iter().any(|v| *v < 0.0)
        || !beta.is_finite()
        || beta < 0.0
    {
        return None;
    }
    let j = skew_generator(theta)?;
    let g = z.map(|v| v + beta * v.powi(3));
    let mut rate = 0.0;
    for i in 0..4 {
        let dz = (0..4).map(|l| j[i][l] * g[l]).sum::<f64>() - r[i] * g[i];
        rate += g[i] * dz;
    }
    rate.is_finite().then_some(rate)
}

fn solve(mut a: Matrix, mut b: [f64; 4]) -> Option<[f64; 4]> {
    for k in 0..4 {
        let pivot = (k..4).max_by(|&i, &l| a[i][k].abs().total_cmp(&a[l][k].abs()))?;
        if a[pivot][k].abs() < 1e-14 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..4 {
            let scale = a[i][k] / a[k][k];
            for l in k..4 {
                a[i][l] -= scale * a[k][l];
            }
            b[i] -= scale * b[k];
        }
    }
    let mut y = [0.0; 4];
    for i in (0..4).rev() {
        y[i] = (b[i] - (i + 1..4).map(|l| a[i][l] * y[l]).sum::<f64>()) / a[i][i];
    }
    y.iter().all(|v| v.is_finite()).then_some(y)
}

/// Unforced quadratic midpoint step, theta and r frozen for one hop.
/// Rejection leaves the caller's old state untouched. No energy projection.
pub fn midpoint_unforced(z: [f64; 4], h: f64, theta: [f64; 6], r: [f64; 4]) -> Option<[f64; 4]> {
    if !h.is_finite()
        || h <= 0.0
        || !z.iter().chain(r.iter()).all(|v| v.is_finite())
        || r.iter().any(|v| *v < 0.0)
    {
        return None;
    }
    let mut a = skew_generator(theta)?;
    for i in 0..4 {
        a[i][i] -= r[i];
    }
    let mut lhs = [[0.0; 4]; 4];
    let mut rhs = z;
    for i in 0..4 {
        for l in 0..4 {
            lhs[i][l] = if i == l { 1.0 } else { 0.0 };
            lhs[i][l] -= 0.5 * h * a[i][l];
            rhs[i] += 0.5 * h * a[i][l] * z[l];
        }
    }
    if !lhs
        .iter()
        .flatten()
        .chain(rhs.iter())
        .all(|v| v.is_finite())
    {
        return None;
    }
    solve(lhs, rhs)
}

/// Optional quadratic split-flow route: finite donor->receiver transfer without
/// reversal in this isolated conservative substep (not in the full midpoint map).
pub fn directed_pair_exchange(z: &mut [f64; 4], donor: usize, receiver: usize, angle_budget: f64) {
    assert!(donor < 4 && receiver < 4 && donor != receiver);
    assert!(angle_budget.is_finite() && angle_budget >= 0.0);
    assert!(z.iter().all(|v| v.is_finite()));
    let x = z[donor];
    let y = z[receiver];
    let sign = if x * y < 0.0 { -1.0 } else { 1.0 };
    let angle = sign * angle_budget.min(x.abs().atan2(y.abs()));
    let (s, c) = angle.sin_cos();
    z[donor] = c * x - s * y;
    z[receiver] = s * x + c * y;
}

#[cfg(test)]
mod tests {
    use super::*;
    fn random(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }
    #[test]
    fn independent_basis_and_legacy_mapping() {
        for k in 0..6 {
            let mut theta = [0.0; 6];
            theta[k] = 1.0;
            let j = skew_generator(theta).unwrap();
            assert_eq!(j.iter().flatten().filter(|&&v| v != 0.0).count(), 2);
            for i in 0..4 {
                for l in 0..4 {
                    assert_eq!(j[i][l], -j[l][i]);
                }
            }
        }
        assert_eq!(
            skew_generator(legacy_theta(6.0)).unwrap(),
            [
                [0.0, -16.0, 0.0, -6.0],
                [16.0, 0.0, 6.0, 0.0],
                [0.0, -6.0, 0.0, -32.0],
                [6.0, 0.0, 32.0, 0.0]
            ]
        );
    }
    #[test]
    fn switched_controls_decay_and_satisfy_midpoint_balance() {
        let mut seed = 400042;
        let mut z = [1.5, -1.0, 0.8, -0.6];
        for hop in 0..10_000 {
            let theta = std::array::from_fn(|_| 100.0 * random(&mut seed));
            let r = std::array::from_fn(|_| 0.5 + 4.0 * random(&mut seed).abs());
            let h = [256.0 / 48_000.0, 0.001, 0.05, 0.2][hop % 4];
            let y = midpoint_unforced(z, h, theta, r).unwrap();
            let mid: [f64; 4] = std::array::from_fn(|i| 0.5 * (z[i] + y[i]));
            let loss = h * (0..4).map(|i| r[i] * mid[i].powi(2)).sum::<f64>();
            let scale = energy(z).max(1.0);
            assert!(energy(y) <= energy(z) + 1e-12 * scale);
            assert!((energy(y) - energy(z) + loss).abs() < 1e-12 * scale);
            z = y;
        }
        assert!(energy(z) < 1e-20);
    }
    #[test]
    fn continuous_quadratic_and_quartic_certificate() {
        let mut seed = 7234001;
        for _ in 0..256 {
            let z = std::array::from_fn(|_| random(&mut seed));
            let theta = std::array::from_fn(|_| 100.0 * random(&mut seed));
            let r = std::array::from_fn(|_| random(&mut seed).abs());
            for beta in [0.0, 0.5, 250.0] {
                let expected = -(0..4)
                    .map(|i| r[i] * (z[i] + beta * z[i].powi(3)).powi(2))
                    .sum::<f64>();
                let actual = unforced_storage_rate(z, theta, r, beta).unwrap();
                assert!(actual <= 1e-9);
                assert!((actual - expected).abs() < 1e-9 * expected.abs().max(1.0));
            }
        }
    }
    #[test]
    fn conservative_null_and_directed_empty_receiver() {
        let z = [0.9, -0.7, 0.2, -0.1];
        let y = midpoint_unforced(z, 0.1, [19.0; 6], [0.0; 4]).unwrap();
        assert!((energy(y) - energy(z)).abs() < 1e-12);
        let mut seed = 33422;
        for case in 0..128 {
            let mut z = [
                random(&mut seed),
                0.0,
                0.0,
                if case % 2 == 0 {
                    0.0
                } else {
                    random(&mut seed)
                },
            ];
            let old = z;
            directed_pair_exchange(&mut z, 0, 3, 0.2);
            assert!(z[3].powi(2) >= old[3].powi(2) - 1e-12);
            assert!((energy(z) - energy(old)).abs() < 1e-12);
        }
    }
    #[test]
    fn schedule_nulls_and_invalid_controls() {
        let f = HopFeatures {
            high_slope_db_oct: -12.0,
            high_flatness: 0.1,
            sample_headroom_db: 6.0,
            high_jitter_rad: 0.1,
            onset: 1.0,
            active: true,
            side_sub_fraction: 0.6,
            sub_lr_correlation: -1.0,
            air_authority: 1.0,
            rumble_authority: 1.0,
        };
        let (mid, side, r) = hop_targets(f).unwrap();
        assert!(mid[2] > 0.0 && side[5] > 32.0 && r[3] > 4.0);
        assert_eq!(side[2], 0.0);
        let (mid, side, r) = hop_targets(HopFeatures { active: false, ..f }).unwrap();
        assert_eq!(mid[2], 0.0);
        assert_eq!(side[5], 32.0);
        assert_eq!(r[3], 4.0);
        let (mid, side, r) = hop_targets(HopFeatures {
            air_authority: 0.0,
            rumble_authority: 0.0,
            ..f
        })
        .unwrap();
        assert_eq!(mid[2], 0.0);
        assert_eq!(side[5], 32.0);
        assert_eq!(r[3], 4.0);
        let (_, side, _) = hop_targets(HopFeatures {
            side_sub_fraction: 0.0,
            sub_lr_correlation: 1.0,
            ..f
        })
        .unwrap();
        assert_eq!(side[5], 32.0);
        assert!(skew_generator([f64::NAN; 6]).is_none());
        assert!(midpoint_unforced([1.0; 4], 0.01, [0.0; 6], [-1.0; 4]).is_none());
    }
}
