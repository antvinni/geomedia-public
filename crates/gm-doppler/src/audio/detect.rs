//! Stage 1: find aircraft sounds — tone or not — as slow loudness humps above the recording's own
//! rolling background, with every rejected candidate logged with its reason.

use serde::{Deserialize, Serialize};

use super::decode::Audio;

/// Octave bands (Hz): 50–100, 100–200, …, 1600–3200. "Mid" is bands 1..=4 (100–1600 Hz).
pub const BANDS: [(f64, f64); 6] = [
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 400.0),
    (400.0, 800.0),
    (800.0, 1600.0),
    (1600.0, 3200.0),
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectParams {
    pub frame_s: f64,
    pub hop_s: f64,
    /// Rolling background: this percentile of each band over ±`floor_half_s`.
    pub floor_pct: f64,
    pub floor_half_s: f64,
    /// Main detector: mid-band excess averaged over this many seconds.
    pub smooth_s: f64,
    /// Short-event channel: mid-band excess averaged over this many seconds (logs vehicles etc.).
    pub light_s: f64,
    pub threshold_db: f64,
    pub min_pass_s: f64,
    pub merge_gap_s: f64,
    pub max_jitter_db: f64,
    pub max_low_share: f64,
    pub max_speech_share: f64,
    /// Click-robust level: each frame's band power is this quantile of the band powers of its
    /// `sub_block_s` sub-blocks (Hann, 50 % overlap, inside the frame) instead of the frame's
    /// mean. Clicks fill few sub-blocks; an aircraft's noise is steady on that scale. `None`: the
    /// frame's mean (one FFT per frame).
    #[serde(default)]
    pub sub_quantile: Option<f64>,
    #[serde(default = "default_sub_block_s")]
    pub sub_block_s: f64,
    /// Sustained check: the main detector's level is this running quantile of the per-frame
    /// excess over `smooth_s` instead of its running mean, so a few loud seconds in the window
    /// (one car, a door) cannot carry it. `None`: the running mean.
    #[serde(default)]
    pub smooth_quantile: Option<f64>,
    /// Air absorption: reject a candidate whose added 1.6–3.2 kHz power density is more than
    /// this many dB re its added 200–800 Hz density (`Pass::hf_db`). A distant aircraft loses
    /// its highs to the air (ISO 9613-1: ≈ 10 dB/km at 2 kHz, ≈ 1 dB/km at 250 Hz); a near
    /// sound keeps them. `None`: no check.
    #[serde(default)]
    pub max_hf_db: Option<f64>,
    /// Reject a candidate whose tones rise (an engine revving, a gear change): a Doppler curve
    /// only falls. Needs a spectrogram and ridge pass over the recording (prepare time). Off by
    /// default: on the tuning split it removed no street false pass and cost AeroSonicDB passes
    /// (aircraft spooling up for take-off raise their tones too).
    #[serde(default)]
    pub reject_engine: bool,
}

fn default_sub_block_s() -> f64 {
    0.1
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            frame_s: 1.0,
            hop_s: 0.5,
            floor_pct: 20.0,
            floor_half_s: 120.0,
            smooth_s: 30.0,
            light_s: 3.0,
            // Set on AeroSonicDB hours 1–3 (tuning split): row F1 0.845, 26/33 labelled runs,
            // 0.5 false passes per silent hour (docs/evaluation.md, broadband pipeline).
            threshold_db: 5.0,
            min_pass_s: 20.0,
            merge_gap_s: 8.0,
            max_jitter_db: 4.0,
            max_low_share: 0.6,
            max_speech_share: 0.5,
            // Click-robust level, set on the tuning split (AeroSonicDB hours 1–3, the phone clip,
            // Piccadilly; sweep 2026-11-09): q 0.3 over 0.1-s sub-blocks took strict false passes
            // 6 → 4 and AeroSonicDB runs 26 → 28/33 (row F1 0.845 → 0.890), AeroSonicDB false
            // passes unchanged. q 0.2 tied; 0.3 is the less extreme quantile.
            sub_quantile: Some(0.3),
            sub_block_s: default_sub_block_s(),
            // Sustained check, set on the tuning split (sweep 2026-11-09, on top of the
            // click-robust level): the running 0.4 quantile took strict false passes 4 → 2
            // (AeroSonicDB 1 → 0, the phone clip 2 → 1), runs held at 28/33, row F1 0.890 → 0.900;
            // the median (0.5) removed none.
            smooth_quantile: Some(0.4),
            max_hf_db: None,
            reject_engine: false,
        }
    }
}

impl DetectParams {
    /// The PR #24 detector, with every street-sound cue off: the fixed reference row of the
    /// stage-1 sweep, so a cue's allowance is never judged against an earlier cue's defaults.
    pub fn legacy() -> Self {
        Self {
            sub_quantile: None,
            smooth_quantile: None,
            max_hf_db: None,
            reject_engine: false,
            ..Self::default()
        }
    }
}

/// Per-frame levels over the rolling background (dB). Frame k is centred at `t0 + k·dt`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Envelope {
    pub t0: f64,
    pub dt: f64,
    /// Frame length (s): frames `dt` apart overlap when `frame_s > dt`.
    #[serde(default = "one_second")]
    pub frame_s: f64,
    /// Mid-band excess averaged over `smooth_s`.
    pub mid_db: Vec<f32>,
    /// Mid-band excess per frame.
    pub raw_mid_db: Vec<f32>,
    /// Excess per octave band per frame (NaN for a band above the recording's Nyquist).
    pub bands_db: Vec<[f32; 6]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pass {
    pub id: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub t_peak: f64,
    /// Peak of the smoothed mid-band excess (dB).
    pub excess_db: f64,
    /// Mean smoothed excess over the pass (dB).
    pub score: f64,
    /// `cut_at_start`, `cut_at_end`.
    pub flags: Vec<String>,
    /// Added 1.6–3.2 kHz power density re the added 200–800 Hz density (dB); `None` when the
    /// recording has no 1.6–3.2 kHz band or nothing was added below.
    #[serde(default)]
    pub hf_db: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rejected {
    pub t_start: f64,
    pub t_end: f64,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Detection {
    pub passes: Vec<Pass>,
    pub rejected: Vec<Rejected>,
    pub envelope: Envelope,
    pub notes: Vec<String>,
}

fn one_second() -> f64 {
    1.0
}

/// Analysis rate: octave bands up to 3200 Hz.
const RATE: u32 = 8000;
/// Background percentile for recordings shorter than two floor windows (whole-recording floor).
const SHORT_FLOOR_PCT: f64 = 5.0;
/// The short-event channel logs a sound only this far above the pass threshold.
const SHORT_EXTRA_DB: f64 = 6.0;

/// Find aircraft passes in `a`: runs of the mid-band (100–1600 Hz) excess over the rolling
/// background, averaged over `smooth_s`, above `threshold_db`; each candidate is then judged
/// (duration, burstiness, low-frequency share, speech-like modulation) and either kept as a
/// pass or logged as rejected with its reason. Sounds too short for the main detector are found
/// by a lightly smoothed channel and logged as "too short".
pub fn detect(a: &Audio, p: &DetectParams) -> Detection {
    detect_prepared(&prepare(a, p), p)
}

/// The costly part of detection (band levels, rolling background, fine envelope), computed once
/// per recording; `detect_prepared` judges it under any thresholds (used for tuning).
#[derive(Clone, Debug, Default)]
pub struct Prepared {
    pub duration_s: f64,
    pub t0: f64,
    pub hop: f64,
    pub bands: Vec<[f32; 6]>,
    pub floor: Vec<[f32; 6]>,
    pub excess: Vec<[f32; 6]>,
    pub raw_mid: Vec<f32>,
    pub env20: Vec<f32>,
    /// Spans of rising tones (with `reject_engine`), seconds.
    pub engine_spans: Vec<(f64, f64)>,
    pub notes: Vec<String>,
    /// Set when the recording is too short to judge (then nothing else is filled).
    pub too_short: Option<Rejected>,
}

/// Band levels, background and envelopes of `a` (frame, hop and background from `p`).
pub fn prepare(a: &Audio, p: &DetectParams) -> Prepared {
    let audio = if a.sample_rate > RATE {
        super::resample::decimate(a, RATE)
    } else {
        a.clone()
    };
    let sr = f64::from(audio.sample_rate.max(1));
    let x = &audio.samples;
    let dur = x.len() as f64 / sr;
    let mut prep = Prepared {
        duration_s: dur,
        t0: p.frame_s / 2.0,
        hop: p.hop_s,
        ..Default::default()
    };
    let short = Rejected {
        t_start: 0.0,
        t_end: dur,
        reason: format!("too short to judge (recording {dur:.1} s)"),
    };
    if dur < p.min_pass_s.max(2.0 * p.frame_s) {
        prep.too_short = Some(short);
        return prep;
    }
    let bands = band_levels(x, sr, p);
    let nf = bands.len();
    if nf < 3 {
        prep.too_short = Some(short);
        return prep;
    }
    let hop = p.hop_s;
    // Rolling background per band.
    let whole = dur < 2.0 * p.floor_half_s;
    if whole {
        prep.notes.push(format!(
            "recording shorter than {:.0} s: whole-recording background ({SHORT_FLOOR_PCT:.0}th percentile)",
            2.0 * p.floor_half_s
        ));
    }
    let half = (p.floor_half_s / hop).round() as usize;
    let mut floor = vec![[f32::NAN; 6]; nf];
    for b in 0..6 {
        let col: Vec<f32> = bands.iter().map(|r| r[b]).collect();
        if col.iter().all(|v| !v.is_finite()) {
            continue;
        }
        if whole {
            // A short clip is often trimmed around its pass: only the quietest frames are background.
            let q = percentile(&col, SHORT_FLOOR_PCT);
            for f in floor.iter_mut() {
                f[b] = q;
            }
        } else {
            // Every 4th frame, then held (the background moves slowly).
            let mut last = f32::NAN;
            for (k, f) in floor.iter_mut().enumerate() {
                if k % 4 == 0 {
                    let (lo, hi) = (k.saturating_sub(half), (k + half + 1).min(nf));
                    last = percentile(&col[lo..hi], p.floor_pct);
                }
                f[b] = last;
            }
        }
    }
    let excess: Vec<[f32; 6]> = bands
        .iter()
        .zip(&floor)
        .map(|(r, f)| {
            let mut e = [f32::NAN; 6];
            for b in 0..6 {
                e[b] = r[b] - f[b];
            }
            e
        })
        .collect();
    let raw_mid: Vec<f32> = excess
        .iter()
        .map(|e| {
            let v: Vec<f32> = e[1..=4].iter().copied().filter(|v| v.is_finite()).collect();
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        })
        .collect();
    prep.env20 = fine_envelope(x, sr);
    if p.reject_engine {
        prep.engine_spans = engine_spans(&audio);
    }
    prep.bands = bands;
    prep.floor = floor;
    prep.excess = excess;
    prep.raw_mid = raw_mid;
    prep
}

/// Judge a prepared recording under `p`'s thresholds: passes and logged rejections.
pub fn detect_prepared(prep: &Prepared, p: &DetectParams) -> Detection {
    let mut det = Detection {
        notes: prep.notes.clone(),
        ..Default::default()
    };
    if let Some(r) = &prep.too_short {
        det.rejected.push(r.clone());
        return det;
    }
    let (t0, hop) = (prep.t0, prep.hop);
    let t_of = |k: usize| t0 + k as f64 * hop;
    let (bands, floor, excess, raw_mid, env20) = (
        &prep.bands,
        &prep.floor,
        &prep.excess,
        &prep.raw_mid,
        &prep.env20,
    );
    let nf = bands.len();
    let w = (p.smooth_s / hop).round() as usize;
    let mid = match p.smooth_quantile {
        Some(q) if nf >= w => moving_quantile(raw_mid, w, q),
        Some(_) => {
            det.notes
                .push("too short for the sustained check: running mean used".into());
            moving_mean(raw_mid, w)
        }
        None => moving_mean(raw_mid, w),
    };
    let light = moving_mean(raw_mid, (p.light_s / hop).round() as usize);
    det.envelope = Envelope {
        t0,
        dt: hop,
        frame_s: p.frame_s,
        mid_db: mid.clone(),
        raw_mid_db: raw_mid.clone(),
        bands_db: excess.clone(),
    };
    // The absorption check needs the 1.6–3.2 kHz band (a sample rate of 6.4 kHz or more).
    let has_hf = bands.iter().any(|r| r[5].is_finite());
    if p.max_hf_db.is_some() && !has_hf {
        det.notes
            .push("no 1.6–3.2 kHz band at this sample rate: absorption check skipped".into());
    }
    let thr = p.threshold_db as f32;
    let gap = (p.merge_gap_s / hop).round() as usize;
    let mains = runs(&mid, thr, gap);
    // The short-event channel: loud, brief sounds outside every main candidate.
    for (s, e) in runs(&light, thr + SHORT_EXTRA_DB as f32, gap) {
        if mains.iter().any(|&(ms, me)| s <= me && e >= ms) {
            continue;
        }
        let len = (e - s + 1) as f64 * hop;
        // Long enough for a pass, yet no main candidate: loud in bursts, not sustained.
        let reason = if len >= p.min_pass_s {
            format!(
                "not sustained (loud in bursts over {len:.1} s; the {:.0}-s level stays below the threshold)",
                p.smooth_s
            )
        } else {
            format!("too short ({len:.1} s < {:.0} s)", p.min_pass_s)
        };
        det.rejected.push(Rejected {
            t_start: t_of(s),
            t_end: t_of(e),
            reason,
        });
    }
    for (s, e) in mains {
        let (ts, te) = (t_of(s), t_of(e));
        let reject = |reason: String| Rejected {
            t_start: ts,
            t_end: te,
            reason,
        };
        // Duration of the sound itself (the lightly smoothed level above threshold).
        let above = (s..=e).filter(|&k| light[k] >= thr).count() as f64 * hop;
        if above < p.min_pass_s {
            det.rejected.push(reject(format!(
                "too short ({above:.1} s < {:.0} s)",
                p.min_pass_s
            )));
            continue;
        }
        let d: Vec<f64> = (s + 1..=e)
            .map(|k| f64::from(raw_mid[k] - raw_mid[k - 1]))
            .collect();
        // Median |Δ| so a few clicks can't decide it (bursty sources change level every frame).
        let jitter = median_abs(&d);
        if jitter > p.max_jitter_db {
            det.rejected
                .push(reject(format!("too bursty (jitter {jitter:.1} dB)")));
            continue;
        }
        // Share of the added sound power (band power minus its background, linear) below 100 Hz.
        let added = |b: usize| -> f64 {
            let mut sum = 0.0;
            for k in s..=e {
                let (l, f) = (f64::from(bands[k][b]), f64::from(floor[k][b]));
                if l.is_finite() && f.is_finite() {
                    sum += (10f64.powf(l / 10.0) - 10f64.powf(f / 10.0)).max(0.0);
                }
            }
            sum
        };
        let low = added(0);
        let total: f64 = (0..6).map(added).sum();
        let hf_db = {
            let (hi, mid) = (added(5), added(2) + added(3));
            let v = 10.0 * (hi / mid).log10();
            (has_hf && mid > 0.0 && v.is_finite()).then_some(v)
        };
        let share = if total > 0.0 { low / total } else { 0.0 };
        if share > p.max_low_share {
            det.rejected.push(reject(format!(
                "mostly below 100 Hz (wind-like, {:.0} % of the added power)",
                100.0 * share
            )));
            continue;
        }
        let speech = modulation_share(env20, ts, te);
        if speech > p.max_speech_share {
            det.rejected.push(reject(format!(
                "speech-like modulation ({:.0} % at 3–6 Hz)",
                100.0 * speech
            )));
            continue;
        }
        if p.reject_engine && covered(&prep.engine_spans, ts, te) >= ENGINE_COVER * (te - ts) {
            det.rejected
                .push(reject("engine: rising tones (rpm), not Doppler".into()));
            continue;
        }
        if let (Some(t), Some(v)) = (p.max_hf_db, hf_db)
            && v > t
        {
            det.rejected.push(reject(format!(
                "near sound: keeps high frequencies (1.6–3.2 kHz density {v:.0} dB re 200–800 Hz; a distant aircraft loses them)"
            )));
            continue;
        }
        let k_peak = (s..=e)
            .max_by(|&i, &j| light[i].total_cmp(&light[j]))
            .unwrap_or(s);
        let mut flags = Vec::new();
        if s == 0 {
            flags.push("cut_at_start".to_string());
        }
        if e + 1 == nf {
            flags.push("cut_at_end".to_string());
        }
        det.passes.push(Pass {
            id: det.passes.len() + 1,
            t_start: ts,
            t_end: te,
            t_peak: t_of(k_peak),
            excess_db: f64::from(mid[s..=e].iter().copied().fold(f32::MIN, f32::max)),
            score: f64::from(mid[s..=e].iter().sum::<f32>()) / (e - s + 1) as f64,
            flags,
            hf_db,
        });
    }
    det
}

/// Octave-band levels (dB) per frame; NaN for a band above Nyquist. With `p.sub_quantile`, the
/// click-robust level (a quantile over the frame's sub-blocks); otherwise the frame's mean.
fn band_levels(x: &[f32], sr: f64, p: &DetectParams) -> Vec<[f32; 6]> {
    let n = (p.frame_s * sr).round() as usize;
    let m = (p.sub_block_s * sr).round() as usize;
    match p.sub_quantile {
        Some(q) if m >= 16 && m <= n => sub_block_levels(x, sr, p, q),
        _ => frame_levels(x, sr, p),
    }
}

/// FFT bin range [lo, hi) of each octave band at this FFT size (None above Nyquist).
fn band_bins(sr: f64, nfft: usize) -> Vec<Option<(usize, usize)>> {
    let df = sr / nfft as f64;
    BANDS
        .iter()
        .map(|&(lo, hi)| {
            (hi <= sr / 2.0).then(|| {
                (
                    (lo / df).ceil() as usize,
                    ((hi / df).ceil() as usize).max(1),
                )
            })
        })
        .collect()
}

/// Per-bin mean power of each band in every `m`-sample Hann block starting at multiples of
/// `step` (NaN for a band above Nyquist).
fn block_powers(x: &[f32], sr: f64, m: usize, step: usize) -> Vec<[f64; 6]> {
    use rustfft::{FftPlanner, num_complex::Complex};
    let nfft = m.next_power_of_two();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(nfft);
    let w: Vec<f32> = (0..m)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / m as f64).cos()) as f32)
        .collect();
    let idx = band_bins(sr, nfft);
    let mut buf = vec![Complex::new(0f32, 0f32); nfft];
    let mut out = Vec::new();
    let mut s = 0;
    while s + m <= x.len() {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = if i < m {
                Complex::new(x[s + i] * w[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.process(&mut buf);
        let mut r = [f64::NAN; 6];
        for (b, ix) in idx.iter().enumerate() {
            if let Some((lo, hi)) = *ix {
                let hi = hi.min(nfft / 2);
                if hi > lo {
                    r[b] = buf[lo..hi]
                        .iter()
                        .map(|c| f64::from(c.norm_sqr()))
                        .sum::<f64>()
                        / (hi - lo) as f64;
                }
            }
        }
        out.push(r);
        s += step;
    }
    out
}

/// The click-robust levels: per frame and band, the `q` quantile of its sub-blocks' powers.
fn sub_block_levels(x: &[f32], sr: f64, p: &DetectParams, q: f64) -> Vec<[f32; 6]> {
    let n = (p.frame_s * sr).round() as usize;
    let h = ((p.hop_s * sr).round() as usize).max(1);
    let m = (p.sub_block_s * sr).round() as usize;
    let step = (m / 2).max(1);
    if n < 16 || x.len() < n {
        return Vec::new();
    }
    let blocks = block_powers(x, sr, m, step);
    let mut out = Vec::new();
    let mut vals = Vec::new();
    let mut s = 0;
    while s + n <= x.len() {
        // Sub-blocks lying wholly inside [s, s + n).
        let g_lo = s.div_ceil(step);
        let g_hi = ((s + n - m) / step).min(blocks.len().saturating_sub(1));
        let mut r = [f32::NAN; 6];
        for (b, rb) in r.iter_mut().enumerate() {
            vals.clear();
            vals.extend(
                blocks[g_lo..=g_hi.max(g_lo)]
                    .iter()
                    .map(|v| v[b])
                    .filter(|v| v.is_finite()),
            );
            if vals.is_empty() {
                continue;
            }
            vals.sort_by(f64::total_cmp);
            let i = (q.clamp(0.0, 1.0) * (vals.len() - 1) as f64).round() as usize;
            *rb = (10.0 * (vals[i] + 1e-30).log10()) as f32;
        }
        out.push(r);
        s += h;
    }
    out
}

/// One FFT per frame: each band's per-bin mean power (dB).
fn frame_levels(x: &[f32], sr: f64, p: &DetectParams) -> Vec<[f32; 6]> {
    use rustfft::{FftPlanner, num_complex::Complex};
    let n = (p.frame_s * sr).round() as usize;
    let h = ((p.hop_s * sr).round() as usize).max(1);
    if n < 16 || x.len() < n {
        return Vec::new();
    }
    let nfft = n.next_power_of_two();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(nfft);
    let w: Vec<f32> = (0..n)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
        .collect();
    let df = sr / nfft as f64;
    let idx: Vec<Option<(usize, usize)>> = BANDS
        .iter()
        .map(|&(lo, hi)| {
            (hi <= sr / 2.0).then(|| {
                (
                    (lo / df).ceil() as usize,
                    ((hi / df).ceil() as usize).max(1),
                )
            })
        })
        .collect();
    let mut buf = vec![Complex::new(0f32, 0f32); nfft];
    let mut out = Vec::new();
    let mut s = 0;
    while s + n <= x.len() {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = if i < n {
                Complex::new(x[s + i] * w[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.process(&mut buf);
        let mut r = [f32::NAN; 6];
        for (b, ix) in idx.iter().enumerate() {
            if let Some((lo, hi)) = *ix {
                let hi = hi.min(nfft / 2);
                if hi > lo {
                    let pw: f64 = buf[lo..hi]
                        .iter()
                        .map(|c| f64::from(c.norm_sqr()))
                        .sum::<f64>()
                        / (hi - lo) as f64;
                    r[b] = (10.0 * (pw + 1e-30).log10()) as f32;
                }
            }
        }
        out.push(r);
        s += h;
    }
    out
}

/// `pct` percentile of the finite values in `v` (NaN if none).
fn percentile(v: &[f32], pct: f64) -> f32 {
    let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        return f32::NAN;
    }
    f.sort_by(f32::total_cmp);
    let i = ((pct / 100.0) * (f.len() - 1) as f64).round() as usize;
    f[i.min(f.len() - 1)]
}

/// Centred moving mean over `w` frames (over the frames available at the ends).
fn moving_mean(v: &[f32], w: usize) -> Vec<f32> {
    let half = w / 2;
    let mut pre = vec![0f64; v.len() + 1];
    for (i, x) in v.iter().enumerate() {
        pre[i + 1] = pre[i] + f64::from(*x);
    }
    (0..v.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(v.len()));
            ((pre[b] - pre[a]) / (b - a) as f64) as f32
        })
        .collect()
}

/// Share of a candidate that rising tones must cover to mark it an engine.
const ENGINE_COVER: f64 = 0.3;

/// Ridge linking for engine tones: rpm chirps change far faster than any Doppler curve (up to
/// 0.3 of their frequency per second) and last seconds, not minutes.
fn engine_ridges() -> super::RidgeParams {
    super::RidgeParams {
        max_rel_slope_per_s: 0.3,
        // The first link predicts no slope: allow a miss as large as the slope limit, but bridge
        // no gaps (a tone has a peak in every frame; noise linked across gaps wanders).
        miss_rel_per_s: 0.3,
        max_gap_s: 0.25,
        min_duration_s: 2.0,
        min_drop_frac: 0.0,
        min_mean_snr_db: 10.0,
        ..super::RidgeParams::default()
    }
}

/// Spans of rising tones: linked spectral peaks (kept or not by the Doppler rules) with mean SNR
/// ≥ 10 dB whose frequency climbs ≥ 10 % within 10 s. A passing source only lowers its pitch.
fn engine_spans(a: &Audio) -> Vec<(f64, f64)> {
    let s = super::spectrogram(
        a,
        &super::SpecParams {
            window_s: 0.25,
            hop_s: ENGINE_HOP_S,
            ..super::SpecParams::default()
        },
    );
    let rp = engine_ridges();
    super::ridges::ridges_explained(&s, &rp)
        .into_iter()
        .map(|(r, _)| r)
        .filter(|r| {
            r.mean_snr_db() >= rp.min_mean_snr_db && r.t_end() - r.t_start() >= rp.min_duration_s
        })
        .flat_map(|r| rising_spans(&r))
        .collect()
}

/// Frame step of the engine spectrogram (s).
const ENGINE_HOP_S: f64 = 0.125;

/// The stretches of a ridge where its frequency (running median over ±0.5 s, so a stray point
/// cannot decide it) climbs ≥ 10 % within 10 s, with a peak in at least half the frames between
/// (a tone; noise peaks linked across gaps are sparse).
fn rising_spans(r: &super::Ridge) -> Vec<(f64, f64)> {
    let n = r.points.len();
    let mut w = Vec::new();
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let t = r.points[i].t;
            w.clear();
            w.extend(
                r.points
                    .iter()
                    .filter(|q| (q.t - t).abs() <= 0.5)
                    .map(|q| q.f),
            );
            w.sort_by(f64::total_cmp);
            w[w.len() / 2]
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..n {
        let t0 = r.points[i].t;
        let end = (i + 1..n)
            .take_while(|&j| r.points[j].t - t0 <= 10.0)
            .filter(|&j| {
                let dense = (j - i + 1) as f64 >= 0.5 * ((r.points[j].t - t0) / ENGINE_HOP_S + 1.0);
                f[j] >= 1.1 * f[i] && dense
            })
            .last();
        if let Some(j) = end {
            out.push((t0, r.points[j].t));
        }
    }
    out
}

/// Seconds of [t0, t1] covered by the union of `spans`.
fn covered(spans: &[(f64, f64)], t0: f64, t1: f64) -> f64 {
    let mut v: Vec<(f64, f64)> = spans
        .iter()
        .map(|&(a, b)| (a.max(t0), b.min(t1)))
        .filter(|(a, b)| b > a)
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut total, mut end) = (0.0, f64::NEG_INFINITY);
    for (a, b) in v {
        if b > end {
            total += b - a.max(end);
            end = b;
        }
    }
    total
}

/// Centred running `q` quantile over `w` frames (over the frames available at the ends).
fn moving_quantile(v: &[f32], w: usize, q: f64) -> Vec<f32> {
    let half = w / 2;
    let mut buf: Vec<f32> = Vec::with_capacity(w + 1);
    (0..v.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(v.len()));
            buf.clear();
            buf.extend_from_slice(&v[a..b]);
            buf.sort_by(f32::total_cmp);
            let k = (q.clamp(0.0, 1.0) * (buf.len() - 1) as f64).round() as usize;
            buf[k]
        })
        .collect()
}

/// Maximal runs of `v ≥ thr` (inclusive frame indices), merged across gaps ≤ `gap` frames.
fn runs(v: &[f32], thr: f32, gap: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        if v[i] >= thr {
            let s = i;
            while i + 1 < v.len() && v[i + 1] >= thr {
                i += 1;
            }
            match out.last_mut() {
                Some(last) if s - last.1 - 1 <= gap => last.1 = i,
                _ => out.push((s, i)),
            }
        }
        i += 1;
    }
    out
}

fn median_abs(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut a: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}

/// 200–1000 Hz level (dB) every 50 ms, for the speech check.
fn fine_envelope(x: &[f32], sr: f64) -> Vec<f32> {
    let w0 = std::f64::consts::TAU * 450.0 / sr;
    let alpha = w0.sin() / (2.0 * 0.7);
    let a0 = 1.0 + alpha;
    let (b0, b2, a1, a2) = (
        alpha / a0,
        -alpha / a0,
        -2.0 * w0.cos() / a0,
        (1.0 - alpha) / a0,
    );
    let (mut x1, mut x2, mut y1, mut y2) = (0f64, 0f64, 0f64, 0f64);
    let block = ((0.05 * sr) as usize).max(1);
    let mut out = Vec::with_capacity(x.len() / block + 1);
    let mut acc = 0f64;
    for (i, &v) in x.iter().enumerate() {
        let v = f64::from(v);
        let y = b0 * v + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = v;
        y2 = y1;
        y1 = y;
        acc += y * y;
        if (i + 1) % block == 0 {
            out.push((10.0 * (acc / block as f64 + 1e-30).log10()) as f32);
            acc = 0.0;
        }
    }
    out
}

/// Share of the fine envelope's modulation power at 3–6 Hz among 0.5–10 Hz, over [t0, t1].
fn modulation_share(env20: &[f32], t0: f64, t1: f64) -> f64 {
    use rustfft::{FftPlanner, num_complex::Complex};
    let (a, b) = (
        ((t0 / 0.05) as usize).min(env20.len()),
        ((t1 / 0.05) as usize).min(env20.len()),
    );
    if b <= a + 40 {
        return 0.0;
    }
    let seg = &env20[a..b];
    let m = seg.iter().map(|v| f64::from(*v)).sum::<f64>() / seg.len() as f64;
    let n = seg.len().next_power_of_two();
    let mut buf: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(seg.get(i).map_or(0.0, |v| f64::from(*v) - m), 0.0))
        .collect();
    FftPlanner::<f64>::new()
        .plan_fft_forward(n)
        .process(&mut buf);
    let df = 20.0 / n as f64;
    let (mut band, mut all) = (0.0, 0.0);
    for (k, c) in buf.iter().enumerate().take(n / 2) {
        let f = k as f64 * df;
        let pw = c.norm_sqr();
        if (0.5..=10.0).contains(&f) {
            all += pw;
            if (3.0..=6.0).contains(&f) {
                band += pw;
            }
        }
    }
    if all > 0.0 { band / all } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 8000;

    struct Lcg(u64);
    impl Lcg {
        fn normal(&mut self) -> f64 {
            // Box–Muller on a 64-bit LCG (deterministic test noise).
            let mut u = || {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
    }

    /// `x` plus 5-ms clicks (noise bursts 40 dB above `sigma`) at seeded Poisson times, `rate`
    /// per second on average, over [t0, t1).
    fn with_clicks(
        mut x: Vec<f64>,
        sigma: f64,
        (t0, t1): (f64, f64),
        rate: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut r = Lcg(seed);
        let mut t = t0;
        let len = (0.005 * f64::from(SR)) as usize;
        loop {
            t += -r.uniform().ln() / rate;
            if t >= t1 {
                return x;
            }
            let at = (t * f64::from(SR)) as usize;
            for (i, v) in noise(0.005, 100.0 * sigma, r.0)
                .iter()
                .take(len)
                .enumerate()
            {
                if at + i < x.len() {
                    x[at + i] += v;
                }
            }
        }
    }

    /// The highest smoothed mid-band excess over [t0, t1].
    fn mid_peak(d: &Detection, t0: f64, t1: f64) -> f32 {
        let e = &d.envelope;
        e.mid_db
            .iter()
            .enumerate()
            .filter(|(k, _)| {
                let t = e.t0 + *k as f64 * e.dt;
                t >= t0 && t <= t1
            })
            .map(|(_, v)| *v)
            .fold(f32::MIN, f32::max)
    }

    /// The PR #24 detector with the click-robust level at quantile `q`, sub-blocks `block_s`.
    fn robust(q: f64, block_s: f64) -> DetectParams {
        DetectParams {
            sub_quantile: Some(q),
            sub_block_s: block_s,
            ..DetectParams::legacy()
        }
    }

    fn noise(seconds: f64, sigma: f64, seed: u64) -> Vec<f64> {
        let mut r = Lcg(seed);
        (0..(seconds * f64::from(SR)) as usize)
            .map(|_| sigma * r.normal())
            .collect()
    }

    /// One-pole low-pass (cut-off `fc`), for wind-like and shaped noise.
    fn lowpass(x: &[f64], fc: f64) -> Vec<f64> {
        let k = 1.0 - (-std::f64::consts::TAU * fc / f64::from(SR)).exp();
        let mut y = 0.0;
        x.iter()
            .map(|&v| {
                y += k * (v - y);
                y
            })
            .collect()
    }

    fn audio(x: &[f64]) -> Audio {
        Audio::new(SR, x.iter().map(|&v| v as f32).collect())
    }

    /// A broadband jet pass from the simulator: CPA (geometric) at `cpa_s`, duration `dur_s`.
    fn jet(dur_s: f64, cpa_s: f64, level_db: f64) -> (Vec<f64>, f64) {
        let s = gm_sim::scenario::parse(&format!(
            r#"
seed = 11
sample_rate = 8000
duration_s = {dur_s}
start_utc = "2026-09-08T20:10:00Z"
[observer]
lat_deg = 40.88
lon_deg = -73.52
h_ell_m = 100.0
mic_height_m = 1.5
[atmosphere]
c = 343.0
wind_enu = [0.0, 0.0, 0.0]
[noise]
snr_db = 0.0
[[aircraft]]
id = "jet"
icao24 = "a1b2c3"
type_code = "B738"
f0 = 100.0
harmonics = [0.0]
amplitude = 1.0
[aircraft.broadband]
level_db = {level_db}
low_hz = 100.0
high_hz = 3000.0
[aircraft.straight]
heading_deg = 45.0
speed_mps = 120.0
alt_agl_m = 1000.0
cpa_offset_m = 3000.0
cpa_time_s = {cpa_s}
half_span_s = 150.0
"#
        ))
        .unwrap();
        let r = gm_sim::render::render(&s).unwrap();
        (
            r.samples.iter().map(|&v| f64::from(v)).collect(),
            r.truth.events[0].cpa_arrival_audio_s,
        )
    }

    #[test]
    fn sparse_clicks_lift_the_mean_level_but_not_the_quantile() {
        // Handling clicks with nothing else: today's mean level rises over the threshold (the
        // jitter rule may still reject the candidate, so the level is what is asserted).
        let sigma = 0.01;
        let x = with_clicks(noise(180.0, sigma, 21), sigma, (60.0, 120.0), 3.0, 22);
        let a = audio(&x);
        let mean = detect(
            &a,
            &DetectParams {
                sub_quantile: None,
                ..DetectParams::legacy()
            },
        );
        let thr = DetectParams::legacy().threshold_db as f32;
        assert!(
            mid_peak(&mean, 60.0, 120.0) >= thr,
            "{}",
            mid_peak(&mean, 60.0, 120.0)
        );
        for block in [0.05, 0.1] {
            let q = detect(&a, &robust(0.5, block));
            assert!(
                mid_peak(&q, 60.0, 120.0) < thr,
                "{block}: {}",
                mid_peak(&q, 60.0, 120.0)
            );
            assert!(q.passes.is_empty(), "{block}: {:?}", q.passes);
        }
    }

    /// A 60-s hump (5-s raised-cosine ramps) from 90 s in 240 s of noise: noise low-passed at
    /// 800 Hz, gated on/off at 20 Hz (a rotor's slap), mean power `level_db` over the background.
    fn pulsed_hump(level_db: f64) -> Vec<f64> {
        let sigma = 0.01;
        let mut x = noise(240.0, sigma, 31);
        let src = lowpass(&noise(60.0, 1.0, 32), 800.0);
        let p_src = src.iter().map(|v| v * v).sum::<f64>() / src.len() as f64;
        // Gated at 50 % duty: twice the power while on gives `level_db` on average.
        let gain = (2.0 * 10f64.powf(level_db / 10.0) * sigma * sigma / p_src).sqrt();
        let at = 90 * SR as usize;
        for (i, v) in src.iter().enumerate() {
            let t = i as f64 / f64::from(SR);
            let ramp = (t / 5.0).min((60.0 - t) / 5.0).min(1.0);
            let w = 0.5 - 0.5 * (std::f64::consts::PI * ramp).cos();
            let on = ((t * 40.0) as usize).is_multiple_of(2);
            if on {
                x[at + i] += gain * w * v;
            }
        }
        x
    }

    #[test]
    fn pulsed_20hz_source_is_kept() {
        let a = audio(&pulsed_hump(15.0));
        for block in [0.05, 0.1] {
            let d = detect(&a, &robust(0.5, block));
            assert_eq!(
                d.passes.len(),
                1,
                "{block}: {:?} {:?}",
                d.passes,
                d.rejected
            );
        }
    }

    #[test]
    fn aircraft_hump_is_kept_with_sub_quantile() {
        let (j, arrival) = jet(300.0, 150.0, 40.0);
        let bg = lowpass(&noise(300.0, 0.05, 3), 800.0);
        let x: Vec<f64> = j.iter().zip(&bg).map(|(a, b)| a + b).collect();
        for block in [0.05, 0.1] {
            let d = detect(&audio(&x), &robust(0.5, block));
            assert_eq!(
                d.passes.len(),
                1,
                "{block}: {:?} {:?}",
                d.passes,
                d.rejected
            );
            assert!((d.passes[0].t_peak - arrival).abs() <= 4.0);
        }
    }

    #[test]
    fn steady_prop_tones_are_kept_with_sub_quantile() {
        // Prop harmonics (120 Hz and its multiples to 1.2 kHz), falling 8 % across a 60-s
        // raised-cosine hump from 120 s, over noise. (A fundamental below 100 Hz would trip the
        // wind rule in both paths; that is not what this test is about.)
        let mut x = noise(300.0, 0.01, 41);
        let (t0, len) = (120.0, 60.0);
        let n = (len * f64::from(SR)) as usize;
        let mut phase = [0f64; 10];
        for i in 0..n {
            let t = i as f64 / f64::from(SR);
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * t / len).cos();
            let fall = 1.04 - 0.08 * t / len;
            let mut v = 0.0;
            for (k, ph) in phase.iter_mut().enumerate() {
                let f = 120.0 * (k + 1) as f64 * fall;
                *ph += std::f64::consts::TAU * f / f64::from(SR);
                v += 0.05 / (k + 1) as f64 * ph.sin();
            }
            x[(t0 * f64::from(SR)) as usize + i] += w * v;
        }
        let a = audio(&x);
        let plain = detect(
            &a,
            &DetectParams {
                sub_quantile: None,
                ..DetectParams::legacy()
            },
        );
        assert_eq!(
            plain.passes.len(),
            1,
            "{:?} {:?}",
            plain.passes,
            plain.rejected
        );
        for block in [0.05, 0.1] {
            let d = detect(&a, &robust(0.5, block));
            assert_eq!(
                d.passes.len(),
                1,
                "{block}: {:?} {:?}",
                d.passes,
                d.rejected
            );
            assert!(
                (d.passes[0].t_peak - plain.passes[0].t_peak).abs() <= 3.0,
                "{block}: {} vs {}",
                d.passes[0].t_peak,
                plain.passes[0].t_peak
            );
        }
    }

    /// The PR #24 detector with the sustained check at quantile `q`.
    fn sustained(q: f64) -> DetectParams {
        DetectParams {
            smooth_quantile: Some(q),
            ..DetectParams::legacy()
        }
    }

    /// `dur` s of noise with bursts of 800-Hz low-passed noise, `on_s` on / `off_s` off over
    /// [t0, t1), each `level_db` over the background (0.2-s ramps).
    fn duty_bursts(
        dur: f64,
        (t0, t1): (f64, f64),
        on_s: f64,
        off_s: f64,
        level_db: f64,
    ) -> Vec<f64> {
        let sigma = 0.01;
        let mut x = noise(dur, sigma, 51);
        let src = lowpass(&noise(t1 - t0, 1.0, 52), 800.0);
        let p_src = src.iter().map(|v| v * v).sum::<f64>() / src.len() as f64;
        let gain = (10f64.powf(level_db / 10.0) * sigma * sigma / p_src).sqrt();
        let at = (t0 * f64::from(SR)) as usize;
        for (i, v) in src.iter().enumerate() {
            let t = (i as f64 / f64::from(SR)) % (on_s + off_s);
            if t < on_s {
                let ramp = (t / 0.2).min((on_s - t) / 0.2).min(1.0);
                x[at + i] += gain * ramp * v;
            }
        }
        x
    }

    #[test]
    fn duty_cycle_bursts_make_no_pass_when_sustained() {
        // A car every 8 s for a minute: loud 3 s in 8. The running mean of the level lifts over
        // the threshold; the running median stays on the background.
        let a = audio(&duty_bursts(240.0, (60.0, 124.0), 3.0, 5.0, 20.0));
        let thr = DetectParams::legacy().threshold_db as f32;
        let mean = detect(
            &a,
            &DetectParams {
                smooth_quantile: None,
                ..DetectParams::legacy()
            },
        );
        assert!(
            mid_peak(&mean, 60.0, 124.0) >= thr,
            "{}",
            mid_peak(&mean, 60.0, 124.0)
        );
        let q = detect(&a, &sustained(0.5));
        assert!(
            mid_peak(&q, 60.0, 124.0) < thr,
            "{}",
            mid_peak(&q, 60.0, 124.0)
        );
        assert!(q.passes.is_empty(), "{:?}", q.passes);
    }

    #[test]
    fn intermittent_sound_is_logged_as_not_sustained() {
        // The bursts are no main candidate under the sustained check, but the light channel
        // still sees a minute of loud sound: its reason must say why, not "too short".
        let d = detect(
            &audio(&duty_bursts(240.0, (60.0, 124.0), 3.0, 5.0, 20.0)),
            &sustained(0.5),
        );
        assert!(
            d.rejected
                .iter()
                .any(|r| r.reason.starts_with("not sustained")
                    && r.t_start <= 64.0
                    && r.t_end >= 116.0),
            "{:?}",
            d.rejected
        );
        assert!(
            !d.rejected.iter().any(|r| r.reason.starts_with("too short")
                && r.t_end - r.t_start >= DetectParams::legacy().min_pass_s),
            "{:?}",
            d.rejected
        );
    }

    #[test]
    fn aircraft_hump_survives_sustained_check() {
        let (j, arrival) = jet(300.0, 150.0, 40.0);
        let bg = lowpass(&noise(300.0, 0.05, 3), 800.0);
        let x: Vec<f64> = j.iter().zip(&bg).map(|(a, b)| a + b).collect();
        let a = audio(&x);
        let mean = detect(
            &a,
            &DetectParams {
                smooth_quantile: None,
                ..DetectParams::legacy()
            },
        );
        let q = detect(&a, &sustained(0.5));
        assert_eq!(q.passes.len(), 1, "{:?} {:?}", q.passes, q.rejected);
        assert!((q.passes[0].t_peak - mean.passes[0].t_peak).abs() <= 3.0);
        assert!((q.passes[0].t_peak - arrival).abs() <= 4.0);
    }

    #[test]
    fn short_clip_falls_back_to_mean_with_note() {
        let (x, _) = jet(25.0, 12.0, 30.0);
        let d = detect(&audio(&x), &sustained(0.5));
        assert!(
            d.notes
                .iter()
                .any(|n| n == "too short for the sustained check: running mean used"),
            "{:?}",
            d.notes
        );
    }

    /// Today's detector with the absorption check at `t` dB.
    fn absorbing(t: f64) -> DetectParams {
        DetectParams {
            max_hf_db: Some(t),
            ..DetectParams::default()
        }
    }

    /// 240 s of white noise with a 60-s raised-cosine hump of white noise from 90 s, `level_db`
    /// over the background; the hump is first attenuated by ISO 9613-1 air absorption over
    /// `dist_m` (15 °C, 70 % RH, 101.325 kPa), bin by bin.
    fn white_hump(level_db: f64, dist_m: f64) -> Vec<f64> {
        use rustfft::{FftPlanner, num_complex::Complex};
        let sigma = 0.01;
        let mut x = noise(240.0, sigma, 61);
        let src = noise(60.0, sigma * 10f64.powf(level_db / 20.0), 62);
        let n = src.len();
        let mut buf: Vec<Complex<f64>> = src.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let mut planner = FftPlanner::<f64>::new();
        planner.plan_fft_forward(n).process(&mut buf);
        for (k, c) in buf.iter_mut().enumerate() {
            let f = k.min(n - k) as f64 * f64::from(SR) / n as f64;
            let db = crate::physics::absorption::alpha_db_per_m(f, 15.0, 70.0, 101.325) * dist_m;
            *c *= 10f64.powf(-db / 20.0) / n as f64;
        }
        planner.plan_fft_inverse(n).process(&mut buf);
        let at = 90 * SR as usize;
        for (i, c) in buf.iter().enumerate() {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
            x[at + i] += w * c.re;
        }
        x
    }

    #[test]
    fn flat_broadband_hump_is_rejected_as_near() {
        let d = detect(&audio(&white_hump(15.0, 0.0)), &absorbing(-6.0));
        assert!(d.passes.is_empty(), "{:?}", d.passes);
        assert!(
            d.rejected
                .iter()
                .any(|r| r.reason.starts_with("near sound")),
            "{:?}",
            d.rejected
        );
    }

    #[test]
    fn absorbed_hump_is_kept() {
        let d = detect(&audio(&white_hump(15.0, 3000.0)), &absorbing(-6.0));
        assert_eq!(d.passes.len(), 1, "{:?} {:?}", d.passes, d.rejected);
        let hf = d.passes[0].hf_db.expect("hf_db measured");
        assert!(hf < -6.0, "{hf}");
    }

    #[test]
    fn low_rate_recording_skips_absorption_with_note() {
        let a = super::super::resample::decimate(&audio(&white_hump(15.0, 0.0)), 6000);
        let d = detect(&a, &absorbing(-6.0));
        assert_eq!(d.passes.len(), 1, "{:?} {:?}", d.passes, d.rejected);
        assert_eq!(d.passes[0].hf_db, None);
        assert!(
            d.notes
                .iter()
                .any(|n| n == "no 1.6–3.2 kHz band at this sample rate: absorption check skipped"),
            "{:?}",
            d.notes
        );
    }

    /// 240 s of noise; from 90 s a 60-s raised-cosine hump (800-Hz low-passed noise, +15 dB)
    /// carrying a tone at `f(t)` (t from the hump's start).
    fn hump_with_tone(f: impl Fn(f64) -> f64) -> Vec<f64> {
        let sigma = 0.01;
        let mut x = noise(240.0, sigma, 71);
        let src = lowpass(&noise(60.0, 1.0, 72), 800.0);
        let p_src = src.iter().map(|v| v * v).sum::<f64>() / src.len() as f64;
        let gain = (10f64.powf(1.5) * sigma * sigma / p_src).sqrt();
        let at = 90 * SR as usize;
        let mut ph = 0.0;
        for (i, v) in src.iter().enumerate() {
            let t = i as f64 / f64::from(SR);
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * t / 60.0).cos();
            ph += std::f64::consts::TAU * f(t) / f64::from(SR);
            x[at + i] += w * (gain * v + 0.1 * ph.sin());
        }
        x
    }

    fn engine(on: bool) -> DetectParams {
        DetectParams {
            reject_engine: on,
            ..DetectParams::default()
        }
    }

    #[test]
    fn rising_chirps_mark_an_engine() {
        // rpm chirps: 110 → 210 Hz every 4 s, the way an engine revs through its gears.
        let chirps = audio(&hump_with_tone(|t| 110.0 + 25.0 * (t % 4.0)));
        let today = detect(&chirps, &engine(false));
        assert_eq!(
            today.passes.len(),
            1,
            "{:?} {:?}",
            today.passes,
            today.rejected
        );
        let d = detect(&chirps, &engine(true));
        assert!(d.passes.is_empty(), "{:?}", d.passes);
        assert!(
            d.rejected
                .iter()
                .any(|r| r.reason == "engine: rising tones (rpm), not Doppler"),
            "{:?}",
            d.rejected
        );
        // A Doppler tone (falling 8 % across the hump) on the same hump is kept.
        let doppler = audio(&hump_with_tone(|t| 160.0 * (1.04 - 0.08 * t / 60.0)));
        let d = detect(&doppler, &engine(true));
        assert_eq!(d.passes.len(), 1, "{:?} {:?}", d.passes, d.rejected);
    }

    #[test]
    fn the_shipped_defaults_keep_steady_and_pulsed_sources() {
        // Review Focus 2 at the defaults (sub-block quantile 0.3, sustained quantile 0.4), which
        // are stricter for keeping a pass than the 0.5 the mechanism tests use.
        let d = DetectParams::default();
        let pulsed = detect(&audio(&pulsed_hump(15.0)), &d);
        assert_eq!(
            pulsed.passes.len(),
            1,
            "rotor slap: {:?} {:?}",
            pulsed.passes,
            pulsed.rejected
        );
        let (j, arrival) = jet(300.0, 150.0, 40.0);
        let bg = lowpass(&noise(300.0, 0.05, 3), 800.0);
        let x: Vec<f64> = j.iter().zip(&bg).map(|(a, b)| a + b).collect();
        let jet_d = detect(&audio(&x), &d);
        assert_eq!(
            jet_d.passes.len(),
            1,
            "jet: {:?} {:?}",
            jet_d.passes,
            jet_d.rejected
        );
        assert!((jet_d.passes[0].t_peak - arrival).abs() <= 4.0);
        let prop = detect(
            &audio(&hump_with_tone(|t| 120.0 * (1.04 - 0.08 * t / 60.0))),
            &d,
        );
        assert_eq!(
            prop.passes.len(),
            1,
            "prop tone: {:?} {:?}",
            prop.passes,
            prop.rejected
        );
    }

    #[test]
    fn legacy_is_the_pr24_detector() {
        let l = DetectParams::legacy();
        assert_eq!(
            (
                l.frame_s,
                l.hop_s,
                l.floor_pct,
                l.floor_half_s,
                l.smooth_s,
                l.light_s
            ),
            (1.0, 0.5, 20.0, 120.0, 30.0, 3.0)
        );
        assert_eq!(
            (l.threshold_db, l.min_pass_s, l.merge_gap_s, l.max_jitter_db),
            (5.0, 20.0, 8.0, 4.0)
        );
        assert_eq!((l.max_low_share, l.max_speech_share), (0.6, 0.5));
    }

    #[test]
    fn broadband_pass_is_detected() {
        let (x, arrival) = jet(300.0, 150.0, 30.0);
        let d = detect(&audio(&x), &DetectParams::default());
        assert_eq!(d.passes.len(), 1, "{:?} {:?}", d.passes, d.rejected);
        let p = &d.passes[0];
        assert!(
            (p.t_peak - arrival).abs() <= 3.0,
            "peak {} vs {arrival}",
            p.t_peak
        );
        assert!(p.t_start < arrival && p.t_end > arrival);
    }

    #[test]
    fn silence_gives_no_pass() {
        let d = detect(&audio(&noise(300.0, 0.01, 1)), &DetectParams::default());
        assert!(d.passes.is_empty(), "{:?}", d.passes);
    }

    #[test]
    fn steady_noise_gives_no_pass() {
        let x = lowpass(&noise(300.0, 0.3, 2), 800.0);
        let d = detect(&audio(&x), &DetectParams::default());
        assert!(d.passes.is_empty(), "{:?}", d.passes);
    }

    #[test]
    fn hump_over_steady_noise_is_found() {
        let (j, arrival) = jet(300.0, 150.0, 40.0);
        let bg = lowpass(&noise(300.0, 0.05, 3), 800.0);
        let x: Vec<f64> = j.iter().zip(&bg).map(|(a, b)| a + b).collect();
        let d = detect(&audio(&x), &DetectParams::default());
        assert_eq!(d.passes.len(), 1, "{:?} {:?}", d.passes, d.rejected);
        assert!((d.passes[0].t_peak - arrival).abs() <= 4.0);
    }

    #[test]
    fn car_like_hump_is_rejected_too_short() {
        let mut x = noise(200.0, 0.01, 4);
        let burst = lowpass(&noise(3.0, 1.0, 5), 1200.0);
        let at = 100 * SR as usize;
        let n = burst.len();
        for (i, b) in burst.iter().enumerate() {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
            x[at + i] += b * w;
        }
        let d = detect(&audio(&x), &DetectParams::default());
        assert!(d.passes.is_empty(), "{:?}", d.passes);
        assert!(
            d.rejected.iter().any(|r| r.reason.starts_with("too short")
                && r.t_start <= 102.0
                && r.t_end >= 100.0),
            "{:?}",
            d.rejected
        );
    }

    #[test]
    fn truck_hump_5s_at_25db_is_rejected() {
        let mut x = noise(200.0, 0.01, 12);
        let burst = lowpass(&noise(5.0, 0.18, 13), 1500.0); // ≈ +25 dB over the floor
        let at = 90 * SR as usize;
        let n = burst.len();
        for (i, b) in burst.iter().enumerate() {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
            x[at + i] += b * w;
        }
        let d = detect(&audio(&x), &DetectParams::default());
        assert!(d.passes.is_empty(), "{:?}", d.passes);
        assert!(
            d.rejected.iter().any(|r| r.reason.starts_with("too short")),
            "{:?}",
            d.rejected
        );
    }

    #[test]
    fn clicks_do_not_make_a_pass_bursty() {
        // Handling clicks on top of a real pass: the pass must survive.
        let (mut x, arrival) = jet(300.0, 150.0, 30.0);
        for c in 0..10 {
            let at = ((110.0 + 8.0 * c as f64) * f64::from(SR)) as usize;
            let click = noise(0.1, 2.0, 100 + c);
            for (i, v) in click.iter().enumerate() {
                x[at + i] += v;
            }
        }
        let d = detect(&audio(&x), &DetectParams::default());
        assert_eq!(d.passes.len(), 1, "{:?} {:?}", d.passes, d.rejected);
        assert!((d.passes[0].t_peak - arrival).abs() <= 6.0);
    }

    #[test]
    fn hump_filling_most_of_a_short_clip_is_found() {
        // A 60-s clip trimmed around the pass (AeroSonicDB-style): background only at the ends.
        let (x, arrival) = jet(60.0, 22.0, 30.0);
        let d = detect(&audio(&x), &DetectParams::default());
        assert_eq!(
            d.passes.len(),
            1,
            "{:?} {:?} {:?}",
            d.passes,
            d.rejected,
            d.notes
        );
        assert!((d.passes[0].t_peak - arrival).abs() <= 5.0);
    }

    #[test]
    fn wind_bursts_are_rejected() {
        // 60 s of bursty sub-100 Hz noise: 0.5–2 s gusts.
        let mut x = noise(300.0, 0.01, 6);
        let mut r = Lcg(7);
        let mut t = 120.0;
        while t < 180.0 {
            let len = 0.5 + 1.5 * (r.normal().abs().min(1.0));
            let g = lowpass(&noise(len, 3.0, (t * 10.0) as u64), 60.0);
            let at = (t * f64::from(SR)) as usize;
            for (i, v) in g.iter().enumerate() {
                if at + i < x.len() {
                    x[at + i] += v;
                }
            }
            t += len + 0.3 + r.normal().abs();
        }
        // Today's detector: the sustained check rejects the gusts before the wind rule sees them.
        let d = detect(&audio(&x), &DetectParams::default());
        assert!(d.passes.is_empty(), "{:?}", d.passes);
        assert!(
            d.rejected.iter().any(|r| r.reason.starts_with("too bursty")
                || r.reason.starts_with("mostly below 100 Hz")
                || r.reason.starts_with("not sustained")),
            "{:?}",
            d.rejected
        );
        // Without it, the wind rule (or the jitter rule) still catches them.
        let d = detect(
            &audio(&x),
            &DetectParams {
                smooth_quantile: None,
                ..DetectParams::default()
            },
        );
        assert!(d.passes.is_empty(), "{:?}", d.passes);
        assert!(
            d.rejected.iter().any(|r| r.reason.starts_with("too bursty")
                || r.reason.starts_with("mostly below 100 Hz")),
            "{:?}",
            d.rejected
        );
    }

    #[test]
    fn pass_cut_by_recording_start_is_found() {
        // CPA arrival ~1 s after the start: the recording opens at the loudest point.
        let (x, arrival) = jet(300.0, -7.0, 30.0);
        assert!(arrival < 3.0, "{arrival}");
        let d = detect(&audio(&x), &DetectParams::default());
        assert_eq!(d.passes.len(), 1, "{:?} {:?}", d.passes, d.rejected);
        assert!(d.passes[0].flags.iter().any(|f| f == "cut_at_start"));
        assert!(d.passes[0].t_start <= 1.0);
    }

    #[test]
    fn short_recording_is_judged_or_logged() {
        for secs in [10.0, 6.0] {
            let d = detect(&audio(&noise(secs, 0.01, 8)), &DetectParams::default());
            assert!(d.passes.is_empty());
            assert!(
                d.rejected
                    .iter()
                    .any(|r| r.reason.starts_with("too short to judge")),
                "{secs} s: {:?}",
                d.rejected
            );
        }
    }

    #[test]
    fn speech_like_modulation_is_rejected() {
        // 40 s of a 150 Hz harmonic comb switched on and off at 4 Hz, over quiet noise.
        let mut x = noise(240.0, 0.01, 9);
        let sr = f64::from(SR);
        let (a, b) = ((100.0 * sr) as usize, (140.0 * sr) as usize);
        for (i, v) in x.iter_mut().enumerate().take(b).skip(a) {
            let t = i as f64 / sr;
            let gate = if (t * 4.0).fract() < 0.5 { 1.0 } else { 0.05 };
            let comb: f64 = (1..=8)
                .map(|k| (std::f64::consts::TAU * 150.0 * k as f64 * t).sin() / k as f64)
                .sum();
            *v += 0.5 * gate * comb;
        }
        let d = detect(&audio(&x), &DetectParams::default());
        assert!(d.passes.is_empty(), "{:?}", d.passes);
        assert!(
            d.rejected
                .iter()
                .any(|r| r.reason.starts_with("speech-like")),
            "{:?}",
            d.rejected
        );
    }
}
