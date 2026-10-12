//! Exact signal time, independent of processing wall-clock time.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct SignalInstant {
    pub nanoseconds: u128,
    pub fractional_numerator: u32,
    pub fractional_denominator: u32,
}
impl SignalInstant {
    pub fn seconds(self) -> f64 {
        (self.nanoseconds as f64
            + self.fractional_numerator as f64 / self.fractional_denominator as f64)
            * 1e-9
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SignalClock {
    pub sample_rate: u32,
}
impl SignalClock {
    pub fn at(self, sample: u64) -> SignalInstant {
        assert!(self.sample_rate > 0);
        let ticks = sample as u128 * 1_000_000_000;
        SignalInstant {
            nanoseconds: ticks / self.sample_rate as u128,
            fractional_numerator: (ticks % self.sample_rate as u128) as u32,
            fractional_denominator: self.sample_rate,
        }
    }
    pub fn seconds_for_samples(self, samples: u64) -> f64 {
        samples as f64 / self.sample_rate as f64
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sample_time_is_exact_and_does_not_accumulate_rounding() {
        let c = SignalClock { sample_rate: 48000 };
        let first = c.at(1);
        assert_eq!(first.nanoseconds, 20833);
        assert_eq!(first.fractional_numerator, 16000);
        assert_eq!(c.at(48000).nanoseconds, 1_000_000_000);
        assert_eq!(c.at(48000 * 3600).nanoseconds, 3_600_000_000_000);
        assert_eq!(c.at(48000 * 3600).fractional_numerator, 0);
        let x = SignalClock { sample_rate: 44100 }.at(44100 * 3600);
        assert_eq!(x.nanoseconds, 3_600_000_000_000);
        assert_eq!(x.fractional_numerator, 0);
    }
}
