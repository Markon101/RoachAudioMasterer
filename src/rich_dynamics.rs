//! Small bounded spectral field; project operator, no literal fluid solver.
use crate::{
    native_dsp::BINS,
    scene_features::{Prepared, Row},
};
use rustfft::num_complex::Complex32 as C;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Kind {
    Diagonal,
    Rotation,
    Transport,
    Basis,
}
impl Kind {
    pub fn mode(self) -> u8 {
        match self {
            Self::Diagonal => 3,
            Self::Rotation => 4,
            Self::Transport => 5,
            Self::Basis => 6,
        }
    }
    pub fn from_mode(m: u8) -> Self {
        match m {
            3 => Self::Diagonal,
            4 => Self::Rotation,
            5 => Self::Transport,
            _ => Self::Basis,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Diagonal => "diagonal",
            Self::Rotation => "rotation",
            Self::Transport => "transport",
            Self::Basis => "basis",
        }
    }
}
#[derive(Clone, Copy)]
pub struct Terms {
    pub z: C,
    pub lf: C,
    pub lt: C,
    pub h: C,
    pub n: C,
}
pub fn terms(p: &Prepared, state: &[Vec<C>; 2], r: Row) -> Terms {
    let (c, t, k) = (r.channel, r.time, r.bin);
    let i = t * BINS + k;
    let z = state[c][i];
    let left = if k > 0 { state[c][i - 1] } else { z };
    let right = if k + 1 < BINS { state[c][i + 1] } else { z };
    let phase = match k % 4 {
        0 => C::new(1.0, 0.0),
        1 => C::new(0.0, 1.0),
        2 => C::new(-1.0, 0.0),
        _ => C::new(0.0, -1.0),
    };
    let before = if t > 0 { state[c][i - BINS] * phase } else { z };
    let after = if t + 1 < p.frames {
        state[c][i + BINS] * phase.conj()
    } else {
        z
    };
    Terms {
        z,
        lf: left + right - z * 2.0,
        lt: before + after - z * 2.0,
        h: p.harmonic[c][i],
        n: p.noise[c][i],
    }
}
fn rotate(z: C) -> C {
    C::new(-z.im, z.re)
}
pub fn velocity(kind: Kind, y: &[f32], t: Terms) -> C {
    let a = 0.75 * y[2].tanh();
    let b = 0.75 * y[3].tanh();
    if kind == Kind::Diagonal {
        return C::new(y[0] + a * t.z.re, y[1] + b * t.z.im);
    }
    let mut v = if kind == Kind::Basis {
        t.h * (0.85 * y[0].tanh()) + t.n * (0.25 * y[1].tanh())
    } else {
        C::new(y[0], y[1])
    };
    v += t.z * a + rotate(t.z) * b;
    if matches!(kind, Kind::Transport | Kind::Basis) {
        v += t.lf * (0.08 * y[4].tanh()) + t.lt * (0.04 * y[5].tanh());
    }
    v
}
pub fn derivative(kind: Kind, y: &[f32], t: Terms, g: C) -> [f32; 6] {
    let dot = |z: C| (g * z.conj()).re;
    let mut d = [0.0; 6];
    if kind == Kind::Diagonal {
        d[0] = g.re;
        d[1] = g.im;
        d[2] = g.re * t.z.re * 0.75 * (1.0 - y[2].tanh().powi(2));
        d[3] = g.im * t.z.im * 0.75 * (1.0 - y[3].tanh().powi(2));
        return d;
    }
    if kind == Kind::Basis {
        d[0] = dot(t.h) * 0.85 * (1.0 - y[0].tanh().powi(2));
        d[1] = dot(t.n) * 0.25 * (1.0 - y[1].tanh().powi(2));
    } else {
        d[0] = g.re;
        d[1] = g.im;
    }
    d[2] = dot(t.z) * 0.75 * (1.0 - y[2].tanh().powi(2));
    d[3] = dot(rotate(t.z)) * 0.75 * (1.0 - y[3].tanh().powi(2));
    if matches!(kind, Kind::Transport | Kind::Basis) {
        d[4] = dot(t.lf) * 0.08 * (1.0 - y[4].tanh().powi(2));
        d[5] = dot(t.lt) * 0.04 * (1.0 - y[5].tanh().powi(2));
    }
    d
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_field_parameter_derivatives() {
        let t = Terms {
            z: C::new(0.2, -0.4),
            lf: C::new(-0.5, 0.1),
            lt: C::new(0.2, 0.8),
            h: C::new(0.5, -0.7),
            n: C::new(0.1, 0.3),
        };
        let y = [0.3, -0.2, 0.1, 0.4, -0.5, 0.6];
        let g = C::new(-0.7, 0.6);
        for kind in [Kind::Diagonal, Kind::Rotation, Kind::Transport, Kind::Basis] {
            let d = derivative(kind, &y, t, g);
            for j in 0..6 {
                let mut a = y;
                let mut b = y;
                let e = 0.001;
                a[j] += e;
                b[j] -= e;
                let fd =
                    ((velocity(kind, &a, t) - velocity(kind, &b, t)) * g.conj()).re / (2.0 * e);
                assert!((fd - d[j]).abs() < 1e-4, "{kind:?} j{j} {fd} {}", d[j]);
            }
        }
    }
    #[test]
    fn rotation_tangent_and_structured_zero() {
        let z = C::new(0.3, 0.8);
        assert!((z * rotate(z).conj()).re.abs() < 1e-7);
        let t = Terms {
            z: C::default(),
            lf: C::default(),
            lt: C::default(),
            h: C::default(),
            n: C::default(),
        };
        assert_eq!(velocity(Kind::Basis, &[1.0; 6], t), C::default());
    }
    #[test]
    fn phase_aligned_temporal_transport_preserves_stationary_partial() {
        let audio = crate::native_audio::Audio {
            rate: 48000,
            channels: vec![vec![0.0; 2048]; 2],
        };
        let d = crate::scene_features::Damage {
            cutoff: 6000.0,
            transition: 500.0,
            power: 2.0,
        };
        let p = crate::scene_features::prepare(&audio, d, 11);
        let mut state = crate::scene_features::zero_state(&p);
        for k in 129..133 {
            let phase = C::from_polar(1.0, std::f32::consts::FRAC_PI_2 * (k % 4) as f32);
            let mut z = C::new(0.4, -0.1);
            for t in 0..p.frames {
                state[0][t * BINS + k] = z;
                z *= phase;
            }
            for t in 0..p.frames {
                let a = terms(
                    &p,
                    &state,
                    Row {
                        channel: 0,
                        time: t,
                        bin: k,
                    },
                );
                assert!(a.lt.norm() < 2e-6, "t{t} k{k} {:?}", a.lt);
            }
        }
    }
}
