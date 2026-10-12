//! Input-history associative geometry. Runtime state, not fitting model weights.
//! A bounded Gaussian-mixture energy field; graph edges record soft temporal associations.
use crate::signal_time::{SignalClock, SignalInstant};
use serde::Serialize;
const N: usize = 8;
const D: usize = 4;

#[derive(Clone, Serialize)]
pub struct Landscape {
    centers: [[f64; D]; N],
    mass: [f64; N],
    variance: [f64; D],
    transitions: [[f64; N]; N],
    previous: [f64; N],
    pub observations: usize,
}
#[derive(Clone, Serialize)]
pub struct Query {
    pub energy: f64,
    pub gradient: [f64; D],
    pub hessian: [[f64; D]; D],
    pub recall: [f64; D],
    pub association: f64,
    pub fixed_query_field_rate: f64,
    responsibilities: [f64; N],
}
impl Default for Landscape {
    fn default() -> Self {
        Self {
            centers: std::array::from_fn(|i: usize| {
                std::array::from_fn(|j| {
                    if (if j == 3 {
                        i.count_ones() as usize % 2
                    } else {
                        (i >> j) & 1
                    }) == 0
                    {
                        -0.7
                    } else {
                        0.7
                    }
                })
            }),
            mass: [1.0 / N as f64; N],
            variance: [1.0; D],
            transitions: [[1.0 / N as f64; N]; N],
            previous: [1.0 / N as f64; N],
            observations: 0,
        }
    }
}
impl Landscape {
    fn precision(&self) -> [f64; D] {
        self.variance.map(|v| 1.0 / (v + 0.05))
    }
    pub fn query(&self, z: [f64; D]) -> Query {
        let metric = self.precision();
        let logits: [f64; N] = std::array::from_fn(|i| {
            self.mass[i].max(1e-12).ln()
                - 0.5
                    * (0..D)
                        .map(|j| metric[j] * (z[j] - self.centers[i][j]).powi(2))
                        .sum::<f64>()
        });
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum = logits.iter().map(|l| (l - max).exp()).sum::<f64>();
        let responsibilities = logits.map(|l| (l - max).exp() / sum);
        let gradient = std::array::from_fn(|j| {
            (0..N)
                .map(|i| responsibilities[i] * metric[j] * (z[j] - self.centers[i][j]))
                .sum()
        });
        let recall = std::array::from_fn(|j| {
            (0..N)
                .map(|i| responsibilities[i] * self.centers[i][j])
                .sum()
        });
        let hessian = std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                let second = (0..N)
                    .map(|i| {
                        responsibilities[i]
                            * metric[j]
                            * metric[k]
                            * (z[j] - self.centers[i][j])
                            * (z[k] - self.centers[i][k])
                    })
                    .sum::<f64>();
                (if j == k { metric[j] } else { 0.0 }) - second + gradient[j] * gradient[k]
            })
        });
        let association = (0..N)
            .map(|i| {
                (0..N)
                    .map(|j| self.previous[i] * self.transitions[i][j] * responsibilities[j])
                    .sum::<f64>()
            })
            .sum();
        Query {
            energy: -(max + sum.ln()),
            gradient,
            hessian,
            recall,
            association,
            fixed_query_field_rate: 0.0,
            responsibilities,
        }
    }
    /// Exponential relaxation coefficients in signal time; center updates are convex.
    pub fn observe(&mut self, z: [f64; D], sample_delta: u64, clock: SignalClock) -> Query {
        self.observe_elapsed(z, clock.at(sample_delta))
    }
    pub fn observe_elapsed(&mut self, z: [f64; D], elapsed: SignalInstant) -> Query {
        let z = z.map(|x| x.clamp(-4.0, 4.0));
        let mut q = self.query(z);
        let dt = elapsed.seconds();
        if dt == 0.0 {
            return q;
        }
        let fast = 1.0 - (-dt / 0.150).exp();
        let slow = 1.0 - (-dt / 2.5).exp();
        let residual = std::array::from_fn::<_, D, _>(|j| (z[j] - q.recall[j]).powi(2));
        for j in 0..D {
            self.variance[j] =
                (self.variance[j] + slow * (residual[j] - self.variance[j])).clamp(0.01, 16.0);
        }
        for i in 0..N {
            let step = fast * q.responsibilities[i];
            for j in 0..D {
                self.centers[i][j] += step * (z[j] - self.centers[i][j]);
            }
            self.mass[i] = (self.mass[i] + slow * (q.responsibilities[i] - self.mass[i])).max(1e-6);
            for j in 0..N {
                self.transitions[i][j] +=
                    slow * self.previous[i] * (q.responsibilities[j] - self.transitions[i][j]);
            }
            let sum = self.transitions[i].iter().sum::<f64>();
            for t in &mut self.transitions[i] {
                *t /= sum;
            }
        }
        let sum = self.mass.iter().sum::<f64>();
        for m in &mut self.mass {
            *m /= sum;
        }
        self.previous = q.responsibilities;
        self.observations += 1;
        q.fixed_query_field_rate = (self.query(z).energy - q.energy) / dt;
        q
    }
    pub fn components(&self, threshold: f64) -> usize {
        let mut seen = [false; N];
        let mut count = 0;
        for root in 0..N {
            if seen[root] {
                continue;
            }
            count += 1;
            let mut queue = vec![root];
            seen[root] = true;
            while let Some(i) = queue.pop() {
                for j in 0..N {
                    if !seen[j]
                        && (self.transitions[i][j] + self.transitions[j][i]) * 0.5 >= threshold
                    {
                        seen[j] = true;
                        queue.push(j);
                    }
                }
            }
        }
        count
    }

    /// One-step associative forecast. Read-only; no future observation is visible.
    pub fn forecast_from(&self, z: [f64; D]) -> [f64; D] {
        let current = self.query(z).responsibilities;
        let next: [f64; N] =
            std::array::from_fn(|j| (0..N).map(|i| current[i] * self.transitions[i][j]).sum());
        std::array::from_fn(|k| (0..N).map(|j| next[j] * self.centers[j][k]).sum())
    }

    pub fn associative_check() -> serde_json::Value {
        let patterns = [
            [1.5, -1.5, 0.0, 0.0],
            [0.0, 1.5, -1.5, 0.0],
            [-1.5, 0.0, 1.5, 0.0],
        ];
        let clock = SignalClock { sample_rate: 48000 };
        let mut rows = Vec::new();
        for seed in [73u64, 101, 131] {
            for reverse in [false, true] {
                let order = if reverse { [0usize, 2, 1] } else { [0, 1, 2] };
                let mut cyclic = Self::default();
                let mut shuffled = Self::default();
                let mut rng = crate::synth::Rng(seed);
                for i in 0..6000 {
                    let c = order[i % 3];
                    let random = (rng.next_u64() % 3) as usize;
                    let a = patterns[c].map(|x| x + rng.signed() as f64 * 0.02);
                    let b = patterns[random].map(|x| x + rng.signed() as f64 * 0.02);
                    cyclic.observe(a, 240, clock);
                    shuffled.observe(b, 240, clock);
                }
                let mut lesion = cyclic.clone();
                lesion.transitions = [[1.0 / N as f64; N]; N];
                for (name, map) in [
                    ("cyclic", &cyclic),
                    ("shuffled_history", &shuffled),
                    ("uniform_transition_lesion", &lesion),
                ] {
                    let mut error = 0.0;
                    let mut correct = 0;
                    let mut test_rng = crate::synth::Rng(seed ^ 0x191882);
                    for i in 0..300 {
                        let current = order[i % 3];
                        let next = order[(i + 1) % 3];
                        let z = patterns[current].map(|x| x + test_rng.signed() as f64 * 0.02);
                        let prediction = map.forecast_from(z);
                        let distance = |p: &[f64; D]| {
                            (0..D).map(|j| (prediction[j] - p[j]).powi(2)).sum::<f64>()
                        };
                        error += distance(&patterns[next]);
                        let nearest = (0..3)
                            .min_by(|&a, &b| {
                                distance(&patterns[a]).total_cmp(&distance(&patterns[b]))
                            })
                            .unwrap();
                        correct += usize::from(nearest == next);
                    }
                    rows.push(serde_json::json!({"seed":seed,"reverse":reverse,"control":name,
                        "one_step_accuracy":correct as f64/300.0,"mean_squared_coordinate_error":error/300.0,
                        "training_signal_duration":clock.at(6000*240),"queries":300}));
                }
            }
        }
        serde_json::json!({"schema":"associative-cycle-check-v1","scope":"three-pattern latent cycle; not natural audio or semantic memory",
            "query_visibility":"current coordinate only; no state update during evaluation",
            "controls":"matched shuffled history and transition lesion retaining centers/mass/metric","rows":rows})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn energy_gradient_and_associative_state_are_finite_and_bounded() {
        let mut map = Landscape::default();
        let clock = SignalClock { sample_rate: 48000 };
        for i in 0..1000 {
            map.observe([(i as f64 * 0.07).sin(), 0.3, -0.7, 0.8], 240, clock);
        }
        let z = [0.2, -0.4, 0.7, 0.1];
        let q = map.query(z);
        for j in 0..D {
            let mut a = z;
            let mut b = z;
            a[j] += 1e-5;
            b[j] -= 1e-5;
            let fd = (map.query(a).energy - map.query(b).energy) / 2e-5;
            assert!(
                (fd - q.gradient[j]).abs() < 1e-6,
                "fd={fd}, gradient={}",
                q.gradient[j]
            );
            for k in 0..D {
                let curvature = (map.query(a).gradient[k] - map.query(b).gradient[k]) / 2e-5;
                assert!((curvature - q.hessian[k][j]).abs() < 1e-5);
            }
        }
        assert!(map
            .centers
            .iter()
            .flatten()
            .all(|x| x.is_finite() && x.abs() <= 4.0));
        assert!(map.precision().iter().all(|x| *x > 0.0 && x.is_finite()));
        assert!(q.association.is_finite() && (0.0..=1.0).contains(&q.association));
        assert!((1..=8).contains(&map.components(0.15)));
    }
    #[test]
    fn frozen_map_and_zero_time_do_not_invent_temporal_learning() {
        let mut map = Landscape::default();
        let old = serde_json::to_vec(&map).unwrap();
        map.observe([2.0, 1.0, -1.0, 0.0], 0, SignalClock { sample_rate: 48000 });
        assert_eq!(serde_json::to_vec(&map).unwrap(), old);
        map.observe_elapsed(
            [2.0, 1.0, -1.0, 0.0],
            SignalInstant {
                nanoseconds: 1,
                fractional_numerator: 0,
                fractional_denominator: 1,
            },
        );
        assert_ne!(serde_json::to_vec(&map).unwrap(), old);
    }
}
