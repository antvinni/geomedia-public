//! The recording's own noise level over time, for the silence check: an unmatched close pass
//! counts as "should have been heard" only when the recording was quiet when its sound arrived.
//! A loud stretch (traffic, wind, speech) can mask a pass, so silence there is unknown and costs
//! nothing.

use gm_doppler::audio::{ANALYSIS_RATE, Audio, decimate};

/// Level per `step_s` of audio (dB, arbitrary reference) in the band discovery listens to.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    pub step_s: f64,
    pub db: Vec<f32>,
    /// The recording's median level (computed once: the silence check asks per cell).
    pub reference_db: Option<f32>,
}

/// High-pass corner (Hz): below it is rumble and handling noise that no pass tone sits under.
const NOISE_HP_HZ: f64 = 60.0;
/// Median window around an arrival time.
pub const NOISE_HALF_WINDOW_S: f64 = 10.0;
/// Silence is only evidence where the level is at most this far above the recording's median.
/// Provisional: a fixed design constant, not yet the plan's per-curve heard/silent/unknown test
/// calibrated on discover-bench (Phase 4 follow-up).
pub const NOISE_QUIET_MARGIN_DB: f32 = 3.0;

fn median(v: &mut [f32]) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    let m = v.len() / 2;
    Some(*v.select_nth_unstable_by(m, f32::total_cmp).1)
}

impl NoiseProfile {
    /// Per-second level of `a` above NOISE_HP_HZ (decimated to the analysis rate, so ≤ 2 kHz).
    pub fn from_audio(a: &Audio) -> NoiseProfile {
        let d = decimate(a, ANALYSIS_RATE);
        let rate = f64::from(d.sample_rate.max(1));
        let k = (-std::f64::consts::TAU * NOISE_HP_HZ / rate).exp();
        let (mut x1, mut y1) = (0.0f64, 0.0f64);
        let step = (rate as usize).max(1);
        let mut db = Vec::with_capacity(d.samples.len() / step + 1);
        let mut acc = 0.0f64;
        for (i, &s) in d.samples.iter().enumerate() {
            let x = f64::from(s);
            // One-pole high-pass: y[n] = k (y[n−1] + x[n] − x[n−1]).
            let y = k * (y1 + x - x1);
            (x1, y1) = (x, y);
            acc += y * y;
            if (i + 1) % step == 0 {
                db.push((10.0 * (acc / step as f64 + 1e-20).log10()) as f32);
                acc = 0.0;
            }
        }
        NoiseProfile::new(1.0, db)
    }

    pub fn new(step_s: f64, db: Vec<f32>) -> NoiseProfile {
        let reference_db = median(&mut db.clone());
        NoiseProfile {
            step_s,
            db,
            reference_db,
        }
    }

    /// Median level within ±NOISE_HALF_WINDOW_S of `t` (seconds of audio).
    pub fn at(&self, t: f64) -> Option<f32> {
        let lo = ((t - NOISE_HALF_WINDOW_S) / self.step_s).floor().max(0.0) as usize;
        let hi =
            (((t + NOISE_HALF_WINDOW_S) / self.step_s).ceil().max(0.0) as usize).min(self.db.len());
        if lo >= hi {
            return None;
        }
        median(&mut self.db[lo..hi].to_vec())
    }

    /// Was the recording quiet enough at `t` for a missing pass to count as silence? `true`
    /// when there is no profile there (no evidence of masking either way: the old fixed rule).
    pub fn quiet_at(&self, t: f64) -> bool {
        match (self.at(t), self.reference_db) {
            (Some(l), Some(r)) => l <= r + NOISE_QUIET_MARGIN_DB,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_profile_flags_a_loud_stretch() {
        // 120 s of quiet noise with 30 s at +12 dB from 60 s.
        let rate = 8000u32;
        let mut x: u64 = 1;
        let samples: Vec<f32> = (0..rate as usize * 120)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let n = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let t = i as f64 / f64::from(rate);
                if (60.0..90.0).contains(&t) {
                    4.0 * n
                } else {
                    n
                }
            })
            .collect();
        let p = NoiseProfile::from_audio(&Audio::new(rate, samples));
        assert!(p.quiet_at(20.0));
        assert!(!p.quiet_at(75.0));
    }
}
