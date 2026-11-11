//! Zero-phase windowed-sinc decimation to the analysis rate.

use super::decode::Audio;

/// Rate the band up to ~1.6 kHz is analysed at: discovery's first band and the propeller guide.
pub const ANALYSIS_RATE: u32 = 4000;

/// Low-pass and resample `a` to `target_hz` (no-op copy if already at or below it).
pub fn decimate(a: &Audio, target_hz: u32) -> Audio {
    if target_hz == 0 || a.sample_rate <= target_hz {
        return a.clone();
    }
    let src = f64::from(a.sample_rate);
    let dst = f64::from(target_hz);
    let table = Kernel::new(src, dst);
    let x = &a.samples;
    let n_out = (x.len() as f64 * dst / src).floor() as usize;
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        // Exact input position of output sample n, from integer rates.
        let num = n as u64 * u64::from(a.sample_rate);
        let k0 = (num / u64::from(target_hz)) as i64;
        let frac = (num % u64::from(target_hz)) as f64 / dst;
        let phase = ((frac * PHASES as f64).round() as usize).min(PHASES);
        let (w, base) = (&table.weights[phase], k0 - table.half as i64);
        let (mut acc, mut wsum) = (0.0f32, 0.0f32);
        for (j, wj) in w.iter().enumerate() {
            let k = base + j as i64;
            if k >= 0 && (k as usize) < x.len() {
                acc += wj * x[k as usize];
                wsum += wj;
            }
        }
        out.push(if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 });
    }
    Audio {
        sample_rate: target_hz,
        samples: out,
        warnings: a.warnings.clone(),
    }
}

/// Sub-sample phases of the precomputed kernel.
const PHASES: usize = 256;
/// Kernel half-width in output periods.
const HALF_PERIODS: f64 = 28.0;

/// Blackman-windowed sinc (pass band 0.4·target) tabulated at PHASES fractional offsets.
struct Kernel {
    half: usize,
    weights: Vec<Vec<f32>>,
}

impl Kernel {
    fn new(src: f64, dst: f64) -> Self {
        let fc = 0.4 * dst;
        let half_w = HALF_PERIODS / dst;
        let half = (half_w * src).ceil() as usize;
        let weights = (0..=PHASES)
            .map(|p| {
                let frac = p as f64 / PHASES as f64; // output centre is `frac` input samples after tap `half`
                (0..=2 * half)
                    .map(|j| {
                        let dt = (j as f64 - half as f64 - frac) / src;
                        if dt.abs() > half_w {
                            return 0.0;
                        }
                        let arg = 2.0 * fc * dt;
                        let sinc = if arg.abs() < 1e-12 {
                            1.0
                        } else {
                            (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
                        };
                        let u = dt / half_w;
                        let win = 0.42
                            + 0.5 * (std::f64::consts::PI * u).cos()
                            + 0.08 * (2.0 * std::f64::consts::PI * u).cos();
                        (sinc * win) as f32
                    })
                    .collect()
            })
            .collect();
        Self { half, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, f: f64, start_s: f64, secs: f64) -> Audio {
        let n = (secs * f64::from(rate)) as usize;
        let samples = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                if t < start_s {
                    0.0
                } else {
                    (std::f64::consts::TAU * f * t).sin() as f32
                }
            })
            .collect();
        Audio::new(rate, samples)
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn decimate_keeps_tone_time() {
        let d = decimate(&tone(48000, 300.0, 2.0, 4.0), 4000);
        assert_eq!(d.sample_rate, 4000);
        assert_eq!(d.samples.len(), 16000);
        // Onset: first sample above half amplitude, within ±2 ms (one quarter period of 300 Hz is 0.83 ms).
        let onset = d.samples.iter().position(|v| v.abs() > 0.5).unwrap() as f64 / 4000.0;
        assert!((onset - 2.0).abs() < 0.002, "{onset}");
        assert!((rms(&d.samples[12000..]) - 0.5f64.sqrt()).abs() < 0.02);
        let hi = decimate(&tone(48000, 2000.0, 0.0, 2.0), 4000);
        let atten = 20.0 * (rms(&hi.samples[1000..7000]) / 0.5f64.sqrt()).log10();
        assert!(atten < -40.0, "{atten} dB");
        let odd = decimate(&tone(44100, 300.0, 0.0, 2.0), 4000);
        assert_eq!(odd.samples.len(), 8000);
        assert!((rms(&odd.samples[1000..7000]) - 0.5f64.sqrt()).abs() < 0.02);
    }

    #[test]
    fn already_low_rate_unchanged() {
        let a = tone(4000, 100.0, 0.0, 1.0);
        assert_eq!(decimate(&a, 4000).samples, a.samples);
    }

    #[test]
    fn decimation_is_fast() {
        let a = tone(48000, 300.0, 0.0, 60.0);
        let t0 = std::time::Instant::now();
        let d = decimate(&a, 4000);
        let secs = t0.elapsed().as_secs_f64();
        assert_eq!(d.samples.len(), 240_000);
        // Shared CI runners are too noisy for a timing bound; check it on developer machines.
        if std::env::var_os("CI").is_none() {
            assert!(secs < 0.4, "60 s of 48 kHz took {secs:.2} s");
        }
    }
}
