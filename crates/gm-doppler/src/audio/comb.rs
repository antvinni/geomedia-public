//! Harmonic sum: a propeller's tone is a comb of harmonics. Summing the whitened level at
//! f, 2f, 3f, … lifts a pass whose harmonics are each too weak to see, and scores the true
//! fundamental even when a codec's high-pass removed it.

use super::spectrogram::Spectrogram;

/// Per frame and bin f: max(W(f), Σ_{h=1..H} W(h·f) / √n), W linearly interpolated, terms above
/// the spectrogram's top frequency skipped (n = terms present).
pub fn harmonic_sum(s: &Spectrogram, h_max: usize) -> Vec<Vec<f32>> {
    s.whitened
        .iter()
        .map(|fr| harmonic_sum_frame(fr, &s.freqs, h_max))
        .collect()
}

/// `harmonic_sum` of one frame `fr` on the evenly spaced bin frequencies `freqs`.
pub fn harmonic_sum_frame(fr: &[f32], freqs: &[f64], h_max: usize) -> Vec<f32> {
    let (Some(&f_lo), Some(&f_hi)) = (freqs.first(), freqs.last()) else {
        return fr.to_vec();
    };
    let df = if freqs.len() > 1 {
        freqs[1] - freqs[0]
    } else {
        0.0
    };
    if h_max < 2 || df <= 0.0 || fr.len() != freqs.len() {
        return fr.to_vec();
    }
    let at = |fr: &[f32], f: f64| -> f32 {
        let x = (f - f_lo) / df;
        let k = x.floor() as usize;
        if k + 1 >= fr.len() {
            return fr[fr.len() - 1];
        }
        let w = (x - k as f64) as f32;
        fr[k] * (1.0 - w) + fr[k + 1] * w
    };
    freqs
        .iter()
        .enumerate()
        .map(|(b, &f)| {
            let (mut sum, mut n) = (0.0f32, 0usize);
            for h in 1..=h_max {
                let fh = f * h as f64;
                if fh > f_hi {
                    break;
                }
                sum += at(fr, fh);
                n += 1;
            }
            let comb = if n > 1 {
                sum / (n as f32).sqrt()
            } else {
                fr[b]
            };
            fr[b].max(comb)
        })
        .collect()
}

/// Whitened dB → noise-relative power ratio (noise median 1), clipped at `clip`: a bounded-
/// influence energy detector (research A §3.2, C §1.2) — one +40 dB cell adds at most `clip`.
pub fn clipped_ratio(fr: &[f32], clip: f32) -> Vec<f32> {
    fr.iter().map(|&w| 10f32.powf(w / 10.0).min(clip)).collect()
}

/// Like `harmonic_sum_frame`, but each f also scores the comb without its fundamental,
/// Σ_{h=2..H+1} — a phone or codec high-pass (60–200 Hz) can remove f itself — and the better
/// of the two is kept (dropping h = 1 outright loses passes whose fundamental is strongest).
pub fn harmonic_sum_frame_either(fr: &[f32], freqs: &[f64], h_max: usize) -> Vec<f32> {
    let with = harmonic_sum_frame(fr, freqs, h_max);
    let (Some(&f_lo), Some(&f_hi)) = (freqs.first(), freqs.last()) else {
        return with;
    };
    let df = if freqs.len() > 1 {
        freqs[1] - freqs[0]
    } else {
        0.0
    };
    if h_max < 2 || df <= 0.0 || fr.len() != freqs.len() {
        return with;
    }
    let at = |f: f64| -> f32 {
        let x = (f - f_lo) / df;
        let k = x.floor() as usize;
        if k + 1 >= fr.len() {
            return fr[fr.len() - 1];
        }
        let w = (x - k as f64) as f32;
        fr[k] * (1.0 - w) + fr[k + 1] * w
    };
    freqs
        .iter()
        .zip(with)
        .map(|(&f, w)| {
            let (mut sum, mut n) = (0.0f32, 0usize);
            for h in 2..=h_max + 1 {
                let fh = f * h as f64;
                if fh > f_hi {
                    break;
                }
                sum += at(fh);
                n += 1;
            }
            if n > 1 {
                w.max(sum / (n as f32).sqrt())
            } else {
                w
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_linear_lifts_a_faint_tone_more_than_db() {
        // A tone at ρ = 0.5 (−3 dB) on exponential noise, summed along a known track over many
        // frames: the clipped-linear sum's deflection beats the dB sum's (theory ≈ 1.5×).
        let mut x: u64 = 5;
        let mut exp = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((x >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            -u.ln()
        };
        let (frames, bins) = (4000, 64);
        let mut db_on = 0.0;
        let mut lin_on = 0.0;
        let mut db_off = Vec::new();
        let mut lin_off = Vec::new();
        for _ in 0..frames {
            let row: Vec<f32> = (0..bins)
                .map(|b| {
                    let p = exp() * if b == 10 { 1.5 } else { 1.0 };
                    (10.0 * p.log10()) as f32
                })
                .collect();
            let lin = clipped_ratio(&row, 10.0);
            db_on += f64::from(row[10]);
            lin_on += f64::from(lin[10]);
            db_off.push(row[40..].iter().map(|&v| f64::from(v)).sum::<f64>() / 24.0);
            lin_off.push(lin[40..].iter().map(|&v| f64::from(v)).sum::<f64>() / 24.0);
        }
        // Deflection in units of a single cell's spread, from the off-track cells.
        let defl = |on: f64, off: &[f64], cell_sd: f64| {
            let mean = off.iter().sum::<f64>() / off.len() as f64;
            (on / frames as f64 - mean) / (cell_sd / (frames as f64).sqrt())
        };
        let (d_db, d_lin) = (defl(db_on, &db_off, 5.57), defl(lin_on, &lin_off, 1.0));
        assert!(d_lin >= 1.3 * d_db, "linear {d_lin:.1} vs dB {d_db:.1}");
        // One huge cell adds at most the clip.
        let big = clipped_ratio(&[40.0, 0.0], 10.0);
        assert_eq!(big, vec![10.0, 1.0]);
    }

    #[test]
    fn comb_without_fundamental_still_scores_at_f0() {
        // Harmonics 2–5 of 100 Hz at +4, the 100 Hz line removed (a high-pass): "with" gives
        // (0+4+4+4)/√4 = 6, "without" (h = 2..5) 16/√4 = 8; the either-sum keeps 8, and 100 Hz
        // scores at least as high as 50 Hz or 200 Hz.
        let freqs: Vec<f64> = (0..2000).map(|b| 30.0 + 0.25 * f64::from(b)).collect();
        let fr: Vec<f32> = freqs
            .iter()
            .map(|f| {
                if (2..=5).any(|h| (f - 100.0 * f64::from(h)).abs() < 0.13) {
                    4.0
                } else {
                    0.0
                }
            })
            .collect();
        let c = harmonic_sum_frame_either(&fr, &freqs, 4);
        let at = |f: f64| c[((f - 30.0) / 0.25).round() as usize];
        assert!((at(100.0) - 8.0).abs() < 0.05, "{}", at(100.0));
        assert!((harmonic_sum_frame(&fr, &freqs, 4)[280] - 6.0).abs() < 0.05);
        assert!(at(100.0) >= at(50.0) && at(100.0) >= at(200.0));
        // With the fundamental present the plain sum is never lowered.
        assert!(
            c.iter()
                .zip(&harmonic_sum_frame(&fr, &freqs, 4))
                .all(|(a, b)| a >= b)
        );
    }

    #[test]
    fn harmonic_sum_lifts_a_weak_comb() {
        // Four harmonics at +4 dB each, noise 0: the sum at the fundamental is 4·4/√4 = 8 dB.
        let freqs: Vec<f64> = (0..2000).map(|b| 30.0 + 0.25 * f64::from(b)).collect();
        let f0 = 100.0;
        let fr: Vec<f32> = freqs
            .iter()
            .map(|f| {
                if (1..=4).any(|h| (f - f0 * f64::from(h)).abs() < 0.13) {
                    4.0
                } else {
                    0.0
                }
            })
            .collect();
        let s = Spectrogram {
            times: vec![0.0],
            freqs: freqs.clone(),
            power_db: vec![fr.clone()],
            whitened: vec![fr],
            mask: vec![false; freqs.len()],
        };
        let c = harmonic_sum(&s, 4);
        let b = ((f0 - 30.0) / 0.25).round() as usize;
        assert!((c[0][b] - 8.0).abs() < 0.01, "{}", c[0][b]);
        // A single harmonic alone is never lowered (max with the plain value).
        assert!(c[0].iter().zip(&s.whitened[0]).all(|(x, y)| x >= y));
        assert_eq!(harmonic_sum(&s, 1), s.whitened);
    }
}
