//! The guide image for guided curve scoring: the whitened multi-window spectrogram, harmonic-
//! summed, on a log-frequency axis — so a Doppler curve f₀·k(t) is a fixed shift of f₀'s bin
//! by GUIDE_BPO·log₂ k(t), whatever f₀ is.

use super::comb::{clipped_ratio, harmonic_sum_frame, harmonic_sum_frame_either};
use super::decode::Audio;
use super::events::EventParams;
use super::resample::{ANALYSIS_RATE, decimate};
use super::spectrogram::{SpecParams, multi_spectrogram};

/// Log-frequency bins per octave.
pub const GUIDE_BPO: f64 = 192.0;
/// The axis: GUIDE_F_LO·2^(q/GUIDE_BPO) up to GUIDE_F_HI.
pub const GUIDE_F_LO: f64 = 30.0;
pub const GUIDE_F_HI: f64 = 1600.0;
/// Harmonics summed per bin (`comb::harmonic_sum`).
pub const GUIDE_HARMONICS: usize = 4;

/// How a guide image is built: the analysis rate, the STFT windows (max-combined after
/// whitening), the log axis's range, and the harmonics summed per bin.
#[derive(Clone, Debug)]
pub struct GuideParams {
    pub rate: u32,
    pub spec: SpecParams,
    pub windows_s: Vec<f64>,
    pub f_lo: f64,
    pub f_hi: f64,
    pub harmonics: usize,
    /// Also build a score image from noise-relative power clipped at this ratio (the curve sum
    /// uses it; measuring and coverage keep the dB image). `None`: score on the dB image.
    pub linear_clip: Option<f32>,
    /// The score image's harmonic sum also tries the comb without its fundamental.
    pub either_fundamental: bool,
    /// Subtract a per-bin temporal median over this span before standardising.
    pub temporal_median_s: Option<f64>,
}

impl GuideParams {
    /// Propellers and rotors: discovery's front end, 30–1600 Hz, four harmonics summed.
    pub fn prop() -> GuideParams {
        let p = EventParams::default();
        GuideParams {
            rate: ANALYSIS_RATE,
            spec: p.spec,
            windows_s: p.windows_s,
            f_lo: GUIDE_F_LO,
            f_hi: GUIDE_F_HI,
            harmonics: GUIDE_HARMONICS,
            // Front-end analysis: quiet's far PC12 z 14.7 → 17.2 with the null unchanged; the
            // linear image must not feed measuring (that lost the pass).
            linear_clip: Some(10.0),
            either_fundamental: true,
            // Measured to add junk ridges and lose suite/quiet in discovery: props keep none.
            temporal_median_s: None,
        }
    }

    /// Jets: airframe and engine tones at 300–3400 Hz (B §2; on real street audio a jet at
    /// 1.2 km carried a 2030 → 1100 Hz tone). Short windows (a 2 kHz tone at 1 km sweeps tens
    /// of Hz/s near CPA), no zero padding (keeps ~100 MB per hour), the same 0.25 s frame grid,
    /// no harmonic sum (their harmonics fall outside the band).
    pub fn jet() -> GuideParams {
        GuideParams {
            rate: 8000,
            spec: SpecParams {
                window_s: 0.5,
                hop_s: 0.25,
                zero_pad: 1,
                f_min: 280.0,
                f_max: 3400.0,
            },
            windows_s: vec![0.25, 0.5],
            f_lo: 300.0,
            f_hi: 3400.0,
            harmonics: 1,
            // Measured worse for single jet tones (300–1200 Hz, h = 1): keep the dB score.
            linear_clip: None,
            either_fundamental: false,
            temporal_median_s: Some(30.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuideImage {
    /// Frame centres, seconds from the first sample; evenly spaced by `hop_s`.
    pub times: Vec<f64>,
    pub hop_s: f64,
    /// The log axis's origin: bin q is f_lo·2^(q/bpo).
    pub f_lo: f64,
    /// Log bins per octave of this image (GUIDE_BPO, or fewer on a `coarsen`ed image).
    pub bpo: f64,
    /// [frame][log bin], standardised like `level`, from clipped linear power: what curve sums
    /// add up. Empty: they use `level`.
    pub score: Vec<Vec<f32>>,
    /// [frame][log bin], standardised per bin over time: (level − median) / robust spread, so
    /// noise reads ~0 ± 1 at every frequency whatever the harmonic sum and log pooling did.
    pub level: Vec<Vec<f32>>,
}

impl Default for GuideImage {
    fn default() -> Self {
        GuideImage {
            times: Vec::new(),
            hop_s: 0.0,
            f_lo: 0.0,
            bpo: GUIDE_BPO,
            score: Vec::new(),
            level: Vec::new(),
        }
    }
}

impl GuideImage {
    pub fn n_bins(&self) -> usize {
        self.level.first().map_or(0, Vec::len)
    }
    /// Fractional log bin of frequency `f`.
    pub fn bin_of(&self, f: f64) -> f64 {
        self.bpo * (f / self.f_lo).log2()
    }
    pub fn freq_of(&self, q: f64) -> f64 {
        self.f_lo * (q / self.bpo).exp2()
    }
    /// The row a curve sum adds up at frame `j`.
    pub fn score_row(&self, j: usize) -> &[f32] {
        if self.score.is_empty() {
            &self.level[j]
        } else {
            &self.score[j]
        }
    }
}

/// The guide image of `a`: decimated, multi-window whitened spectrogram (as discovery uses),
/// persistent lines zeroed, harmonic-summed, then on the log axis — each log bin the largest
/// linear bin inside it where bins are coarser than the linear grid, interpolated where finer.
/// Built frame by frame: only the log image (~6× smaller) outlives the spectrogram.
pub fn guide_image(a: &Audio) -> GuideImage {
    guide_image_with(a, &GuideParams::prop())
}

/// `guide_image` with explicit parameters (`GuideParams::prop`, `GuideParams::jet`).
pub fn guide_image_with(a: &Audio, p: &GuideParams) -> GuideImage {
    let d = decimate(a, p.rate);
    let s = multi_spectrogram(&d, &p.windows_s, &p.spec);
    let axis = GuideImage {
        f_lo: p.f_lo,
        ..GuideImage::default()
    };
    let (Some(&f_lo), Some(&f_hi)) = (s.freqs.first(), s.freqs.last()) else {
        return GuideImage::default();
    };
    let df = s.bin_hz();
    if df <= 0.0 {
        return GuideImage::default();
    }
    let top = p.f_hi.min(f_hi);
    let n_bins = axis.bin_of(top).floor().max(0.0) as usize + 1;
    // Each log bin's linear range [lo, hi] (fractional linear bin indices).
    let ranges: Vec<(f64, f64)> = (0..n_bins)
        .map(|q| {
            let edge = |x: f64| (axis.freq_of(x) - f_lo) / df;
            (edge(q as f64 - 0.5).max(0.0), edge(q as f64 + 0.5))
        })
        .collect();
    let last = s.freqs.len() - 1;
    let pool = |h: &[f32]| -> Vec<f32> {
        ranges
            .iter()
            .enumerate()
            .map(|(q, &(lo, hi))| {
                let (a, b) = (lo.ceil() as usize, (hi.floor() as usize).min(last));
                if a <= b && b - a >= 1 {
                    h[a..=b].iter().copied().fold(f32::NEG_INFINITY, f32::max)
                } else {
                    let x = (axis.freq_of(q as f64) - f_lo) / df;
                    let k = (x.floor().max(0.0) as usize).min(last);
                    let w = (x - k as f64).clamp(0.0, 1.0) as f32;
                    h[k] * (1.0 - w) + h[(k + 1).min(last)] * w
                }
            })
            .collect()
    };
    let (mut level, mut score): (Vec<Vec<f32>>, Vec<Vec<f32>>) = s
        .whitened
        .iter()
        .map(|fr| {
            let fr: Vec<f32> = fr
                .iter()
                .zip(&s.mask)
                .map(|(&x, &m)| if m { 0.0 } else { x })
                .collect();
            let db = pool(&harmonic_sum_frame(&fr, &s.freqs, p.harmonics));
            let lin = p.linear_clip.map(|c| {
                let r = clipped_ratio(&fr, c);
                pool(&if p.either_fundamental {
                    harmonic_sum_frame_either(&r, &s.freqs, p.harmonics)
                } else {
                    harmonic_sum_frame(&r, &s.freqs, p.harmonics)
                })
            });
            (db, lin.unwrap_or_default())
        })
        .unzip();
    let hop = if s.times.len() > 1 {
        s.times[1] - s.times[0]
    } else {
        p.spec.hop_s
    };
    if let Some(span) = p.temporal_median_s {
        subtract_temporal_median(&mut level, hop, span);
    }
    standardise(&mut level);
    if p.linear_clip.is_some() {
        standardise(&mut score);
    } else {
        score.clear();
    }
    let hop_s = if s.times.len() > 1 {
        s.times[1] - s.times[0]
    } else {
        p.spec.hop_s
    };
    GuideImage {
        times: s.times,
        hop_s,
        f_lo: p.f_lo,
        bpo: GUIDE_BPO,
        score,
        level,
    }
}

/// `img` on a coarser grid for track-first's stage 1: frames averaged in groups of
/// round(`hop_s` / img.hop_s) (NaN skipped; time = the group's centre), and each coarse log bin
/// (`bpo` per octave, same origin) the max of the fine bins whose centres fall inside it — so a
/// curve straddling fine bins loses nothing. `score` and `level` alike; an empty `score` stays
/// empty.
pub fn coarsen(img: &GuideImage, hop_s: f64, bpo: f64) -> GuideImage {
    let g = if img.hop_s > 0.0 {
        ((hop_s / img.hop_s).round() as usize).max(1)
    } else {
        1
    };
    let out_axis = GuideImage {
        f_lo: img.f_lo,
        bpo,
        ..GuideImage::default()
    };
    let nb = img.n_bins();
    // Coarse bin of each fine bin's centre.
    let map: Vec<usize> = (0..nb)
        .map(|q| out_axis.bin_of(img.freq_of(q as f64)).round().max(0.0) as usize)
        .collect();
    let n_out = map.last().map_or(0, |&m| m + 1);
    let pool = |rows: &[Vec<f32>]| -> Vec<Vec<f32>> {
        rows.chunks(g)
            .map(|grp| {
                let mut sum = vec![0f64; nb];
                let mut cnt = vec![0u32; nb];
                for r in grp {
                    for (q, &x) in r.iter().enumerate().take(nb) {
                        if !x.is_nan() {
                            sum[q] += f64::from(x);
                            cnt[q] += 1;
                        }
                    }
                }
                let mut o = vec![f32::NEG_INFINITY; n_out];
                for q in 0..nb {
                    let v = if cnt[q] > 0 {
                        (sum[q] / f64::from(cnt[q])) as f32
                    } else {
                        f32::NAN
                    };
                    let k = map[q];
                    if !v.is_nan() && v > o[k] {
                        o[k] = v;
                    }
                }
                o.iter_mut()
                    .filter(|x| x.is_infinite())
                    .for_each(|x| *x = f32::NAN);
                o
            })
            .collect()
    };
    let times = img
        .times
        .chunks(g)
        .map(|t| t.iter().sum::<f64>() / t.len() as f64)
        .collect();
    GuideImage {
        times,
        hop_s: img.hop_s * g as f64,
        f_lo: img.f_lo,
        bpo,
        score: pool(&img.score),
        level: pool(&img.level),
    }
}

/// Per bin, subtract the median over the surrounding `span_s` (computed on 5 s blocks and
/// interpolated): slowly drifting background — traffic rumble, machinery, a jet's own broadband
/// roar — goes; a tone sweeping through the bin in a few seconds stays. Front-end analysis on
/// real street audio: jet passes at the top of their nulls 2/20 → 6/20 in 300–1200 Hz.
fn subtract_temporal_median(rows: &mut [Vec<f32>], hop_s: f64, span_s: f64) {
    let n = rows.len();
    let nb = rows.first().map_or(0, Vec::len);
    if n == 0 || hop_s <= 0.0 {
        return;
    }
    let step = ((5.0 / hop_s).round() as usize).max(1);
    let half = ((span_s / 2.0 / hop_s).round() as usize).max(1);
    let centres: Vec<usize> = (0..n)
        .step_by(step)
        .map(|c| (c + step / 2).min(n - 1))
        .collect();
    let mut col = Vec::with_capacity(2 * half + 1);
    for q in 0..nb {
        let meds: Vec<f32> = centres
            .iter()
            .map(|&c| {
                col.clear();
                col.extend(
                    rows[c.saturating_sub(half)..(c + half + 1).min(n)]
                        .iter()
                        .map(|r| r[q]),
                );
                col.sort_by(f32::total_cmp);
                col[col.len() / 2]
            })
            .collect();
        for (j, r) in rows.iter_mut().enumerate() {
            let b = (j / step).min(meds.len() - 1);
            let (c0, m0) = (centres[b], meds[b]);
            let m = if j >= c0 && b + 1 < meds.len() {
                let (c1, m1) = (centres[b + 1], meds[b + 1]);
                m0 + (m1 - m0) * (j - c0) as f32 / (c1 - c0).max(1) as f32
            } else if j < c0 && b > 0 {
                let (cp, mp) = (centres[b - 1], meds[b - 1]);
                mp + (m0 - mp) * (j - cp) as f32 / (c0 - cp).max(1) as f32
            } else {
                m0
            };
            r[q] -= m;
        }
    }
}

/// Per log bin: subtract the median over frames and divide by the normalised MAD (floored).
fn standardise(level: &mut [Vec<f32>]) {
    let n = level.first().map_or(0, Vec::len);
    let mut col = Vec::with_capacity(level.len());
    for q in 0..n {
        col.clear();
        col.extend(level.iter().map(|r| r[q]).filter(|x| x.is_finite()));
        if col.is_empty() {
            continue;
        }
        col.sort_by(f32::total_cmp);
        let med = col[col.len() / 2];
        for x in col.iter_mut() {
            *x = (*x - med).abs();
        }
        col.sort_by(f32::total_cmp);
        let spread = (1.4826 * col[col.len() / 2]).max(0.1);
        for r in level.iter_mut() {
            r[q] = if r[q].is_finite() {
                (r[q] - med) / spread
            } else {
                0.0
            };
        }
    }
}

/// Ridge points along a curve: for each `(frame, predicted Hz)`, the strongest bin within
/// ±`tol` (fraction of frequency), refined by a parabola, kept when ≥ `min_db`. Points are
/// (frame time, Hz).
pub fn measure_along(
    img: &GuideImage,
    curve: &[(usize, f64)],
    tol: f64,
    min_db: f32,
) -> Vec<(f64, f64)> {
    let n = img.n_bins() as i64;
    // ±tol in whole log bins (0.4 % → ±1 bin, ±0.36 %): no wider, or noise picks inside a
    // generous window still trace the guide.
    let half = (img.bpo * (1.0 + tol).log2()).round().max(1.0) as i64;
    curve
        .iter()
        .filter_map(|&(j, f)| {
            let row = img.level.get(j)?;
            if f.is_nan() || f <= 0.0 {
                return None;
            }
            let c = img.bin_of(f).round() as i64;
            let (lo, hi) = ((c - half).max(1), (c + half).min(n - 2));
            if lo > hi {
                return None;
            }
            let k =
                (lo..=hi).max_by(|&a, &b| row[a as usize].total_cmp(&row[b as usize]))? as usize;
            if row[k] < min_db {
                return None;
            }
            let (y0, y1, y2) = (
                f64::from(row[k - 1]),
                f64::from(row[k]),
                f64::from(row[k + 1]),
            );
            let den = y0 - 2.0 * y1 + y2;
            let off = if den < 0.0 {
                (0.5 * (y0 - y2) / den).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            Some((img.times[j], img.freq_of(k as f64 + off)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_median_removes_slow_background_keeps_sweep() {
        // Bin 5 holds a slowly drifting background (+6 for the whole minute); bin 9 a short
        // sweep (+6 for 2 s). After the 30 s temporal median: the background ≈ 0, the sweep stays.
        let mut rows: Vec<Vec<f32>> = (0..240).map(|_| vec![0.0; 16]).collect();
        for (j, r) in rows.iter_mut().enumerate() {
            r[5] = 6.0 + 0.01 * j as f32;
            if (100..108).contains(&j) {
                r[9] = 6.0;
            }
        }
        subtract_temporal_median(&mut rows, 0.25, 30.0);
        assert!(rows.iter().all(|r| r[5].abs() < 0.5), "{:?}", rows[120][5]);
        assert!((rows[104][9] - 6.0).abs() < 0.5, "{}", rows[104][9]);
    }

    #[test]
    fn guide_image_and_measure_follow_a_tone() {
        // A tone sweeping 80 → 120 Hz over 60 s (as a pass does: a steady tone is hum, masked
        // and standardised away), with a weaker 2nd harmonic, in noise: the image peaks on the
        // sweep, and measuring along a curve 0.3 % above it returns the sweep.
        let rate = 8000u32;
        let mut x: u64 = 7;
        let (f_a, f_b, dur) = (80.0f64, 120.0f64, 60.0f64);
        let sweep = move |t: f64| f_a + (f_b - f_a) * t / dur;
        let samples: Vec<f32> = (0..rate as usize * dur as usize)
            .map(|i| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((x >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.5;
                let t = i as f64 / f64::from(rate);
                // Phase of a linear chirp: 2π(f_a t + (f_b − f_a) t² / 2dur).
                let ph = std::f64::consts::TAU * (f_a * t + (f_b - f_a) * t * t / (2.0 * dur));
                (ph.sin() * 0.1 + (2.0 * ph).sin() * 0.05) as f32 + noise
            })
            .collect();
        let img = guide_image(&Audio::new(rate, samples));
        assert!(img.level.len() > 50 && (img.hop_s - 0.25).abs() < 1e-9);
        let j = img.level.len() / 2;
        let row = &img.level[j];
        let peak = (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap();
        let want = img.bin_of(sweep(img.times[j]));
        assert!((peak as f64 - want).abs() <= 2.0, "peak {peak} vs {want}");
        let curve: Vec<(usize, f64)> = (10..img.level.len() - 10)
            .map(|j| (j, sweep(img.times[j]) * 1.003))
            .collect();
        let pts = measure_along(&img, &curve, 0.004, 3.0);
        assert!(
            pts.len() * 10 >= curve.len() * 9,
            "{} of {}",
            pts.len(),
            curve.len()
        );
        // The 2 s window smears a 0.67 Hz/s sweep and log bins quantise it: typical error
        // well under a bin, never beyond the ±0.4 % search window plus that smear.
        let mut err: Vec<f64> = pts.iter().map(|p| (p.1 - sweep(p.0)).abs()).collect();
        err.sort_by(f64::total_cmp);
        assert!(err[err.len() / 2] < 0.3, "median {}", err[err.len() / 2]);
        assert!(err[err.len() - 1] < 1.0, "max {}", err[err.len() - 1]);
    }

    fn tone_image() -> GuideImage {
        // 8 frames at 0.25 s, 192 bpo; one bright bin at q = 100 in frames 0..4 only.
        let mut img = GuideImage {
            hop_s: 0.25,
            f_lo: 30.0,
            bpo: 192.0,
            ..Default::default()
        };
        img.times = (0..8).map(|j| 0.125 + 0.25 * f64::from(j)).collect();
        img.level = (0..8)
            .map(|j| {
                let mut r = vec![0.0f32; 400];
                if j < 4 {
                    r[100] = 4.0;
                }
                r
            })
            .collect();
        img.score = img.level.clone();
        img
    }

    #[test]
    fn default_image_has_the_guide_axis() {
        assert_eq!(GuideImage::default().bpo, GUIDE_BPO);
    }

    #[test]
    fn coarsen_pools_max_in_frequency_mean_in_time() {
        let img = tone_image();
        let c = coarsen(&img, 1.0, 48.0);
        assert_eq!(c.times.len(), 2);
        assert_eq!(c.bpo, 48.0);
        assert!((c.hop_s - 1.0).abs() < 1e-12);
        assert!((c.times[0] - 0.5).abs() < 1e-12);
        let q = c.bin_of(img.freq_of(100.0)).round() as usize;
        // Mean of four 4.0 frames; the max over the fine bins inside the coarse one.
        assert!((c.score[0][q] - 4.0).abs() < 1e-6, "{}", c.score[0][q]);
        assert!((c.level[0][q] - 4.0).abs() < 1e-6);
        assert!(c.score[1][q].abs() < 1e-6);
    }

    #[test]
    fn coarsen_keeps_frequency_axis() {
        let img = tone_image();
        let c = coarsen(&img, 1.0, 48.0);
        for f in [40.0, 100.0, 400.0] {
            assert!((c.freq_of(c.bin_of(f)) - f).abs() < 1e-9);
            assert!((c.bin_of(f) - img.bin_of(f) / 4.0).abs() < 1e-9);
        }
        assert!(c.n_bins() * 4 >= img.n_bins() - 3 && c.n_bins() * 4 <= img.n_bins() + 4);
    }
}
