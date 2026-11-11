//! Short-time power spectrum with robust whitening: each bin in dB above its spectral
//! neighbourhood's median in the same frame (removes broadband noise, wind and speech bursts but
//! keeps narrow lines, steady or sweeping), clipped. Steady lines
//! are then masked separately (`lines.rs`), not erased.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use super::decode::Audio;

#[derive(Clone, Debug)]
pub struct SpecParams {
    pub window_s: f64,
    pub hop_s: f64,
    /// FFT length = next power of two ≥ window·zero_pad.
    pub zero_pad: usize,
    pub f_min: f64,
    pub f_max: f64,
}

impl Default for SpecParams {
    fn default() -> Self {
        Self {
            window_s: 1.0,
            hop_s: 0.25,
            zero_pad: 4,
            f_min: 30.0,
            f_max: 1600.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Spectrogram {
    /// Frame centres, seconds from the first sample.
    pub times: Vec<f64>,
    pub freqs: Vec<f64>,
    /// [frame][bin], dB.
    pub power_db: Vec<Vec<f32>>,
    /// [frame][bin], robust whitened level (dB-like units, see `whiten_robust`).
    pub whitened: Vec<Vec<f32>>,
    /// Bins holding a persistent narrowband line (hum, an engine): skipped by ridge finding.
    pub mask: Vec<bool>,
}

impl Spectrogram {
    pub fn bin_hz(&self) -> f64 {
        if self.freqs.len() > 1 {
            self.freqs[1] - self.freqs[0]
        } else {
            0.0
        }
    }
}

pub fn spectrogram(a: &Audio, p: &SpecParams) -> Spectrogram {
    let rate = f64::from(a.sample_rate);
    let win = (p.window_s * rate).round() as usize;
    let hop = ((p.hop_s * rate).round() as usize).max(1);
    let empty = Spectrogram {
        times: vec![],
        freqs: vec![],
        power_db: vec![],
        whitened: vec![],
        mask: vec![],
    };
    if win < 2 || a.samples.len() < win || a.sample_rate == 0 {
        return empty;
    }
    let nfft = (win * p.zero_pad.max(1)).next_power_of_two();
    let df = rate / nfft as f64;
    let b0 = (p.f_min / df).ceil().max(0.0) as usize;
    let b1 = ((p.f_max / df).floor() as usize).min(nfft / 2);
    if b1 < b0 {
        return empty;
    }
    let freqs: Vec<f64> = (b0..=b1).map(|b| b as f64 * df).collect();
    let hann: Vec<f64> = (0..win)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / win as f64).cos())
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(nfft);
    let mut buf = vec![Complex::new(0.0, 0.0); nfft];
    let mut times = Vec::new();
    let mut power_db = Vec::new();
    let mut start = 0;
    while start + win <= a.samples.len() {
        for (i, c) in buf.iter_mut().enumerate() {
            *c = if i < win {
                Complex::new(f64::from(a.samples[start + i]) * hann[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.process(&mut buf);
        power_db.push(
            buf[b0..=b1]
                .iter()
                .map(|c| (10.0 * (c.norm_sqr() + 1e-20).log10()) as f32)
                .collect::<Vec<f32>>(),
        );
        times.push((start as f64 + win as f64 / 2.0) / rate);
        start += hop;
    }
    let whitened = whiten_robust(&power_db, df);
    let mut s = Spectrogram {
        times,
        freqs,
        power_db,
        whitened,
        mask: vec![],
    };
    s.mask = super::lines::persistent_lines(&s);
    s
}

/// Several window lengths on one frequency grid (a shared FFT length) and one hop grid, each
/// whitened on its own and max-combined: short windows follow the steep part of a pass near CPA,
/// long ones lift faint, slowly changing asymptotes. `power_db` comes from the window nearest
/// 1 s (used for sub-bin frequency). Windows longer than the clip are skipped.
pub fn multi_spectrogram(a: &Audio, windows_s: &[f64], p: &SpecParams) -> Spectrogram {
    let empty = Spectrogram {
        times: vec![],
        freqs: vec![],
        power_db: vec![],
        whitened: vec![],
        mask: vec![],
    };
    let rate = f64::from(a.sample_rate);
    if a.sample_rate == 0 {
        return empty;
    }
    let wins: Vec<usize> = windows_s
        .iter()
        .map(|w| (w * rate).round() as usize)
        .filter(|&w| w >= 2 && w <= a.samples.len())
        .collect();
    let Some(&wmax) = wins.iter().max() else {
        return empty;
    };
    let hop = ((p.hop_s * rate).round() as usize).max(1);
    let reference = *wins
        .iter()
        .min_by_key(|&&w| (w as f64 - rate).abs() as u64)
        .unwrap_or(&wmax);
    // The frequency grid is the reference window's (zero-padded as usual). Longer windows use
    // the same FFT length (less zero padding); shorter ones their own, interpolated onto it.
    let nfft = (reference * p.zero_pad.max(1)).next_power_of_two();
    let df = rate / nfft as f64;
    let b0 = (p.f_min / df).ceil().max(0.0) as usize;
    let b1 = ((p.f_max / df).floor() as usize).min(nfft / 2);
    if b1 < b0 {
        return empty;
    }
    let freqs: Vec<f64> = (b0..=b1).map(|b| b as f64 * df).collect();
    // Frame centres (samples): the longest window must fit around each.
    let centres: Vec<usize> = (wmax / 2..=a.samples.len() - (wmax - wmax / 2))
        .step_by(hop)
        .collect();
    let times: Vec<f64> = centres.iter().map(|&c| c as f64 / rate).collect();
    let mut planner = FftPlanner::<f64>::new();
    let mut frames_for = |win: usize| -> Vec<Vec<f32>> {
        let n = (win * p.zero_pad.max(1))
            .next_power_of_two()
            .clamp(win.next_power_of_two(), nfft.max(win.next_power_of_two()));
        let fft = planner.plan_fft_forward(n);
        let mut buf = vec![Complex::new(0.0, 0.0); n];
        let hann: Vec<f64> = (0..win)
            .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / win as f64).cos())
            .collect();
        let dfw = rate / n as f64;
        centres
            .iter()
            .map(|&c| {
                let start = c - win / 2;
                for (i, z) in buf.iter_mut().enumerate() {
                    *z = if i < win {
                        Complex::new(f64::from(a.samples[start + i]) * hann[i], 0.0)
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                }
                fft.process(&mut buf);
                let db = |k: usize| (10.0 * (buf[k.min(n / 2)].norm_sqr() + 1e-20).log10()) as f32;
                if n == nfft {
                    (b0..=b1).map(db).collect()
                } else {
                    // Coarser grid: linear interpolation in dB at each grid frequency.
                    freqs
                        .iter()
                        .map(|&f| {
                            let x = f / dfw;
                            let k = x.floor() as usize;
                            let w = (x - k as f64) as f32;
                            db(k) * (1.0 - w) + db(k + 1) * w
                        })
                        .collect()
                }
            })
            .collect()
    };
    // Peak memory: one power field (the reference's), one whitened field per window at a time,
    // and the running max — the reference's noise statistics are kept, not its whole field.
    let mut power_db: Vec<Vec<f32>> = Vec::new();
    let mut whitened: Vec<Vec<f32>> = Vec::new();
    let mut ref_stats: Option<(f32, f32)> = None;
    for &w in &wins {
        let pw = frames_for(w);
        let wh = whiten_robust(&pw, df);
        if w == reference {
            ref_stats = noise_quantiles(&wh);
            power_db = pw;
        }
        if whitened.is_empty() {
            whitened = wh;
        } else {
            for (acc, fr) in whitened.iter_mut().zip(&wh) {
                for (x, y) in acc.iter_mut().zip(fr) {
                    *x = x.max(*y);
                }
            }
        }
    }
    if wins.len() > 1
        && let Some(r) = ref_stats
    {
        restandardise(&mut whitened, r);
    }
    let mut s = Spectrogram {
        times,
        freqs,
        power_db,
        whitened,
        mask: vec![],
    };
    s.mask = super::lines::persistent_lines(&s);
    s
}

/// Median and 99th percentile of the finite cells of `f` (a sample of every 7th bin), or `None`
/// when there are none. Non-finite cells (a corrupt decode) are ignored, not sorted to the top.
pub(crate) fn noise_quantiles(f: &[Vec<f32>]) -> Option<(f32, f32)> {
    let mut v: Vec<f32> = f
        .iter()
        .flat_map(|fr| fr.iter().step_by(7).copied())
        .filter(|x| x.is_finite())
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f32::total_cmp);
    Some((v[v.len() / 2], v[(v.len() - 1) * 99 / 100]))
}

/// The max of several noise fields has a higher median and a heavier upper tail than one, so
/// thresholds tuned on one window would pass more noise. Map `combined` linearly so its median
/// and 99th percentile match `reference` (from `noise_quantiles` of the reference window).
pub(crate) fn restandardise(combined: &mut [Vec<f32>], reference: (f32, f32)) {
    let Some((cm, c99)) = noise_quantiles(combined) else {
        return;
    };
    let (rm, r99) = reference;
    let k = (r99 - rm) / (c99 - cm);
    if !(k.is_finite() && k > 0.0) {
        return;
    }
    for fr in combined.iter_mut() {
        for x in fr.iter_mut() {
            *x = rm + (*x - cm) * k;
        }
    }
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Half-width of the spectral neighbourhood the floor is taken over.
pub const FLOOR_HALF_HZ: f64 = 10.0;
/// Bins next to the centre left out of the neighbourhood (a line's own main lobe: a 1 s Hann
/// window is ±2 Hz wide).
pub const FLOOR_CORE_HZ: f64 = 2.0;
/// The floor is computed every this many Hz and interpolated between (speed).
pub const FLOOR_STEP_HZ: f64 = 2.0;
/// Clip (dB): no single cell can dominate a sum along a curve.
pub const CLIP_DB: (f32, f32) = (-15.0, 40.0);

/// Robust whitening of `p` ([frame][bin] dB, bin width `df`):
/// 1. per frame, subtract the median of the bins within ±FLOOR_HALF_HZ (excluding ±FLOOR_CORE_HZ);
/// 2. clip to CLIP_DB.
///
/// (A per-bin spread rescale was tried and dropped: bins a slow pass dwells in get a large
/// spread, so the pass itself was scaled down.)
pub fn whiten_robust(p: &[Vec<f32>], df: f64) -> Vec<Vec<f32>> {
    let Some(nb) = p.first().map(Vec::len) else {
        return Vec::new();
    };
    if nb == 0 || df <= 0.0 {
        return p.to_vec();
    }
    let half = ((FLOOR_HALF_HZ / df).round() as usize).max(2);
    let core = (FLOOR_CORE_HZ / df).round() as usize;
    let step = ((FLOOR_STEP_HZ / df).round() as usize).max(1);
    let anchors: Vec<usize> = (0..nb)
        .step_by(step)
        .chain(std::iter::once(nb - 1))
        .collect();
    let mut nbuf: Vec<f32> = Vec::with_capacity(2 * half + 1);
    let rel: Vec<Vec<f32>> = p
        .iter()
        .map(|fr| {
            let floors: Vec<f32> = anchors
                .iter()
                .map(|&c| {
                    nbuf.clear();
                    let (lo, hi) = (c.saturating_sub(half), (c + half).min(nb - 1));
                    nbuf.extend((lo..=hi).filter(|&b| b.abs_diff(c) > core).map(|b| fr[b]));
                    if nbuf.is_empty() {
                        fr[c]
                    } else {
                        median(&mut nbuf)
                    }
                })
                .collect();
            (0..nb)
                .map(|b| {
                    let i = (b / step).min(anchors.len() - 1);
                    let (b0, f0) = (anchors[i], floors[i]);
                    let (b1, f1) = anchors
                        .get(i + 1)
                        .map_or((b0, f0), |&b1| (b1, floors[i + 1]));
                    let floor = if b1 > b0 {
                        f0 + (f1 - f0) * (b - b0) as f32 / (b1 - b0) as f32
                    } else {
                        f0
                    };
                    fr[b] - floor
                })
                .collect()
        })
        .collect();
    let (lo, hi) = CLIP_DB;
    rel.into_iter()
        .map(|fr| fr.into_iter().map(|v| v.clamp(lo, hi)).collect())
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn synth(rate: u32, secs: f64, f: impl Fn(f64) -> f64) -> Audio {
        let n = (secs * f64::from(rate)) as usize;
        let mut phase = 0.0;
        let samples = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                phase += std::f64::consts::TAU * f(t) / f64::from(rate);
                phase.sin() as f32
            })
            .collect();
        Audio::new(rate, samples)
    }

    /// White noise (σ = 1, deterministic) plus optional tone (f, amplitude) and broadband
    /// bursts (start s, length s) at 10× the noise level.
    pub(crate) fn noisy(
        rate: u32,
        secs: f64,
        tone: Option<(f64, f64)>,
        bursts: &[(f64, f64)],
    ) -> Audio {
        let n = (secs * f64::from(rate)) as usize;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut gauss = || {
            // Box–Muller on a xorshift stream.
            let mut u = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        let samples = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let burst = bursts.iter().any(|&(t0, d)| t >= t0 && t < t0 + d);
                let mut v = gauss() * if burst { 10.0 } else { 1.0 };
                if let Some((f, a)) = tone {
                    v += a * (std::f64::consts::TAU * f * t).sin();
                }
                (v * 0.05) as f32
            })
            .collect();
        Audio::new(rate, samples)
    }

    fn bin_of(s: &Spectrogram, f: f64) -> usize {
        ((f - s.freqs[0]) / s.bin_hz()).round() as usize
    }

    #[test]
    fn steady_tone_survives_whitening() {
        // A tone present the whole time is still a tone; median-over-time whitening erased it.
        let s = spectrogram(
            &noisy(4000, 120.0, Some((120.0, 0.3)), &[]),
            &SpecParams::default(),
        );
        let b = bin_of(&s, 120.0);
        let mean: f32 = s.whitened.iter().map(|fr| fr[b]).sum::<f32>() / s.whitened.len() as f32;
        assert!(mean >= 6.0, "mean whitened at the tone {mean}");
    }

    #[test]
    fn restandardise_ignores_nan() {
        // A few NaN cells (a corrupt decode) must not turn the whole field into NaN.
        let mut field: Vec<Vec<f32>> = (0..200)
            .map(|t| {
                (0..100)
                    .map(|b| ((t * 31 + b * 17) % 23) as f32 - 5.0)
                    .collect()
            })
            .collect();
        for t in (0..200).step_by(10) {
            for b in (0..100).step_by(5) {
                field[t][b] = f32::NAN;
            }
        }
        let reference = noise_quantiles(&field).expect("finite cells exist");
        assert!(reference.0.is_finite() && reference.1.is_finite());
        restandardise(&mut field, reference);
        let finite = field.iter().flatten().filter(|x| x.is_finite()).count();
        assert!(finite >= 200 * 100 - 20 * 20, "{finite}");
    }

    #[test]
    fn short_clip_multi_window() {
        // 1.5 s: the 2 s window can't fit; the 0.5 s and 1 s windows still give frames.
        let a = synth(4000, 1.5, |_| 200.0);
        let s = multi_spectrogram(&a, &[0.5, 1.0, 2.0], &SpecParams::default());
        assert!(!s.times.is_empty() && s.whitened.len() == s.times.len());
        assert!(
            multi_spectrogram(
                &synth(4000, 0.3, |_| 200.0),
                &[0.5, 1.0],
                &SpecParams::default()
            )
            .times
            .is_empty()
        );
    }

    #[test]
    fn tone_peak_bin() {
        let s = spectrogram(&synth(4000, 10.0, |_| 97.3), &SpecParams::default());
        assert!((s.bin_hz() - 4000.0 / 16384.0).abs() < 1e-12);
        let frame = &s.power_db[5];
        let (imax, _) = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert!(
            (s.freqs[imax] - 97.3).abs() <= s.bin_hz() / 2.0 + 1e-9,
            "{}",
            s.freqs[imax]
        );
        assert!(s.freqs[0] >= 30.0 && *s.freqs.last().unwrap() <= 1600.0);
    }

    #[test]
    fn frame_times_centred() {
        let s = spectrogram(&synth(4000, 10.0, |_| 100.0), &SpecParams::default());
        for (k, t) in s.times.iter().enumerate() {
            assert!((t - (k as f64 * 0.25 + 0.5)).abs() < 1e-12);
        }
        assert_eq!(s.times.len(), 37);
        assert!(
            spectrogram(&synth(4000, 0.5, |_| 100.0), &SpecParams::default())
                .times
                .is_empty()
        );
    }

    #[test]
    fn constant_line_masked_not_erased() {
        let rate = 4000;
        let line = synth(rate, 60.0, |_| 120.0);
        let sweep = synth(rate, 60.0, |t| 300.0 + 3.0 * t);
        let a = Audio {
            sample_rate: rate,
            samples: line
                .samples
                .iter()
                .zip(&sweep.samples)
                .map(|(x, y)| x + y)
                .collect(),
            warnings: Vec::new(),
        };
        let s = spectrogram(&a, &SpecParams::default());
        let bin120 = s
            .freqs
            .iter()
            .position(|f| (f - 120.0).abs() <= s.bin_hz() / 2.0)
            .unwrap();
        // The steady line stays visible (it is evidence of a machine, not noise) and is masked
        // as a persistent line so ridge finding ignores it.
        let high = s.whitened.iter().filter(|fr| fr[bin120] >= 6.0).count();
        assert!(
            high as f64 >= 0.9 * s.whitened.len() as f64,
            "{high}/{}",
            s.whitened.len()
        );
        assert!(crate::audio::lines::persistent_lines(&s)[bin120]);
        let strong = s
            .times
            .iter()
            .zip(&s.whitened)
            .filter(|(t, fr)| {
                let f = 300.0 + 3.0 * **t;
                let b = ((f - s.freqs[0]) / s.bin_hz()).round() as usize;
                fr[b.saturating_sub(1)..=b + 1]
                    .iter()
                    .cloned()
                    .fold(f32::MIN, f32::max)
                    > 10.0
            })
            .count();
        assert!(strong as f64 >= 0.9 * s.times.len() as f64, "{strong}");
    }
}
