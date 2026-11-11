//! Blind Doppler templates: discovery without ADS-B. On the log-frequency guide image a passing
//! source's curve f₀·k(t; M, τ) is f₀'s bin shifted by GUIDE_BPO·log₂ k — its shape depends only
//! on M = v/c and τ = d/v, and f₀ and the arrival time only move it. So a bank of (M, τ)
//! templates, slid over every time and f₀, finds passes too faint to follow as a ridge
//! (research A, ranks 3–4; discovery analysis §2.1). The impossible rising templates — the same
//! curves reversed in time — give the recording's own null: a passing source only falls.

use super::decode::Audio;
use super::events::{Event, EventParams, event_from_points, vehicle_like};
use super::guide::{GUIDE_BPO, GuideImage, GuideParams, guide_image_with, measure_along};
use super::tracker::{TrackParams, track_paths};
use crate::physics::cv::shape;

/// M = v/c and τ = d/v grids (log-spaced): the aircraft region (cars are τ < 4 s, M < 0.12).
pub const T_MACH: (f64, f64, usize) = (0.08, 0.55, 12);
pub const T_TAU_S: (f64, f64, usize) = (4.0, 120.0, 14);
/// f₀ swept at the comb's fundamental (the prop image is harmonic-summed), and on the jet image
/// (airframe/engine tones, no harmonic sum).
pub const T_F0: (f64, f64) = (40.0, 400.0);
pub const T_F0_JET: (f64, f64) = (300.0, 3000.0);
/// Per-cell clip (σ): a curve must be lit consistently (research C §1.2).
pub const T_CLIP: (f32, f32) = (-1.0, 3.0);
/// Acceptance: z ≥ T_MIN_Z and ≥ the rising null + T_NULL_MARGIN.
pub const T_MIN_Z: f64 = 8.0;
pub const T_NULL_MARGIN: f64 = 2.0;
/// Below this much audio there is no usable rising null: abstain.
pub const T_MIN_AUDIO_S: f64 = 300.0;
/// The prop image's tracker band ends here (Hz).
pub const TRACK_PROP_MAX_HZ: f64 = 1000.0;
/// Centres every this many frames (refined ±this around accepted peaks).
const T_STEP: usize = 4;
const C: f64 = 343.0;

struct Run {
    a: i64,
    b: i64,
    s: i64,
}

struct Template {
    mach: f64,
    tau: f64,
    half: i64,
    runs: Vec<Run>,
}

pub(super) fn logspace((a, b, n): (f64, f64, usize)) -> Vec<f64> {
    (0..n)
        .map(|i| a * (b / a).powf(i as f64 / (n - 1).max(1) as f64))
        .collect()
}

/// The (M, τ) curve's log-bin shift at frame offsets −half..=half (rising: time reversed),
/// run-length encoded.
fn template(mach: f64, tau: f64, hop: f64, rising: bool) -> Template {
    let (v, d) = (mach * C, tau * mach * C);
    let support = (3.5 * tau * (1.0 - mach * mach).sqrt()).clamp(15.0, 150.0);
    let half = (support / hop).round() as i64;
    let mut runs: Vec<Run> = Vec::new();
    for x in -half..=half {
        let t = x as f64 * hop * if rising { -1.0 } else { 1.0 };
        let s = (GUIDE_BPO * shape(t, v, d, 0.0, C).log2()).round() as i64;
        match runs.last_mut() {
            Some(r) if r.s == s && r.b == x => r.b = x + 1,
            _ => runs.push(Run { a: x, b: x + 1, s }),
        }
    }
    Template {
        mach,
        tau,
        half,
        runs,
    }
}

/// Median and normalised MAD of `v` (reorders a copy).
fn robust(v: &[f32], buf: &mut Vec<f32>) -> (f64, f64) {
    buf.clear();
    buf.extend_from_slice(v);
    let m = buf.len() / 2;
    let med = *buf.select_nth_unstable_by(m, f32::total_cmp).1;
    for x in buf.iter_mut() {
        *x = (*x - med).abs();
    }
    let mad = *buf.select_nth_unstable_by(m, f32::total_cmp).1;
    (f64::from(med), 1.4826 * f64::from(mad))
}

#[derive(Clone, Copy, Debug)]
struct Best {
    z: f64,
    t: usize,
    q: usize,
}

/// For every centre, the best template (z, index, f₀ bin) of `bank`.
fn scan(
    p: &[Vec<f32>],
    bank: &[Template],
    q_lo: usize,
    q_hi: usize,
    centres: &[usize],
) -> Vec<Best> {
    let n = p.len() as i64 - 1;
    let nb = p.first().map_or(0, Vec::len) as i64;
    let width = q_hi - q_lo + 1;
    let threads = std::thread::available_parallelism().map_or(4, |x| x.get());
    let chunk = centres.len().div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        let hs: Vec<_> = centres
            .chunks(chunk)
            .map(|cs| {
                sc.spawn(move || {
                    let mut b = vec![0f32; width];
                    let mut buf = Vec::with_capacity(width);
                    cs.iter()
                        .map(|&c| {
                            let mut best = Best {
                                z: f64::NEG_INFINITY,
                                t: 0,
                                q: q_lo,
                            };
                            for (ti, tm) in bank.iter().enumerate() {
                                b.iter_mut().for_each(|x| *x = 0.0);
                                let mut used = 0;
                                for r in &tm.runs {
                                    let (a, e) = ((c as i64 + r.a).max(0), (c as i64 + r.b).min(n));
                                    if a >= e {
                                        continue;
                                    }
                                    let lo = (q_lo as i64).max(-r.s);
                                    let hi = (q_hi as i64).min(nb - 1 - r.s);
                                    if lo > hi {
                                        continue;
                                    }
                                    used += e - a;
                                    let (pa, pe) = (&p[a as usize], &p[e as usize]);
                                    for q in lo..=hi {
                                        let k = (q + r.s) as usize;
                                        b[(q - q_lo as i64) as usize] += pe[k] - pa[k];
                                    }
                                }
                                // Most of the curve inside the recording (a blind pass cut
                                // in half by the file's edge is too easy to fake).
                                if 2 * used < 3 * tm.half {
                                    continue;
                                }
                                let (med, mad) = robust(&b, &mut buf);
                                if mad <= 0.0 {
                                    continue;
                                }
                                let (k, &bmax) = b
                                    .iter()
                                    .enumerate()
                                    .max_by(|x, y| x.1.total_cmp(y.1))
                                    .unwrap();
                                let z = (f64::from(bmax) - med) / mad;
                                if z > best.z {
                                    best = Best {
                                        z,
                                        t: ti,
                                        q: q_lo + k,
                                    };
                                }
                            }
                            best
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// Local maxima of `z` (index into `centres`) within ±`radius` frames, strongest first.
fn peaks(z: &[Best], centres: &[usize], radius: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..z.len())
        .filter(|&i| {
            z[i].z.is_finite()
                && (0..z.len())
                    .all(|k| centres[k].abs_diff(centres[i]) > radius || z[k].z <= z[i].z)
        })
        .collect();
    v.sort_by(|a, b| z[*b].z.total_cmp(&z[*a].z));
    v
}

/// Same pass: overlapping in time, tones at a small-integer ratio within 1 %.
fn same_pass(a: &Event, b: &Event) -> bool {
    let overlap = a.t_start <= b.t_end && b.t_start <= a.t_end;
    let (x, y) = (a.fit.f0.max(b.fit.f0), a.fit.f0.min(b.fit.f0));
    if !(overlap && y > 0.0) {
        return false;
    }
    let r = x / y;
    r.round() <= 4.0 && (r - r.round()).abs() / r <= 0.01
}

/// Does `g` ride on `e`'s ridge: ≥ 20 % of its points within half a frame of one of `e`'s, at a
/// simple harmonic ratio p/q (p, q ≤ 4) within 0.5 %? Loud passes light up their tails, octaves
/// and harmonic cousins; a template curve fitted through those points is the same pass again.
fn rides_on(g: &Event, e: &Event) -> bool {
    if g.points.is_empty() || e.points.is_empty() {
        return false;
    }
    let ratios: Vec<f64> = (1..=4)
        .flat_map(|p| (1..=4).map(move |q| f64::from(p) / f64::from(q)))
        .collect();
    let on = g
        .points
        .iter()
        .filter(|&&(t, f)| {
            let i = e.points.partition_point(|p| p.0 < t - 0.13);
            e.points[i..]
                .iter()
                .take_while(|p| p.0 <= t + 0.13)
                .any(|p| p.1 > 0.0 && ratios.iter().any(|r| ((f / p.1) / r - 1.0).abs() <= 0.005))
        })
        .count();
    on * 5 >= g.points.len()
}

/// Does `g` lie on `e`'s modelled curve family: ≥ 30 % of `g`'s points within 0.6 % of
/// r·f₀·shape(t) of `e`'s fit, for a simple ratio r? The model extends past `e`'s measured points,
/// so a loud pass's tails and harmonic copies — the template stage's duplicates — are caught.
fn on_family(g: &Event, e: &Event) -> bool {
    const RATIOS: [f64; 8] = [0.5, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.5, 2.0, 3.0, 4.0];
    if g.points.is_empty() {
        return false;
    }
    let f = &e.fit;
    let on = g
        .points
        .iter()
        .filter(|&&(t, fg)| {
            let m = f.f0 * shape(t, f.v, f.d, f.tc, C);
            m > 0.0 && RATIOS.iter().any(|r| (fg / (r * m) - 1.0).abs() <= 0.006)
        })
        .count();
    on * 10 >= g.points.len() * 3
}

/// Why the template stage kept or dropped a falling peak (the rejection log behind
/// `template_events`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateVerdict {
    BelowBar,
    FewPoints(usize),
    NoFit,
    Drifted,
    VehicleLike,
    Duplicate,
    Accepted,
}

/// One falling peak the template stage considered.
#[derive(Clone, Copy, Debug)]
pub struct TemplateTrial {
    pub tc: f64,
    pub f0: f64,
    pub mach: f64,
    pub tau: f64,
    pub z: f64,
    pub bar: f64,
    pub verdict: TemplateVerdict,
    /// Proposed by the path tracker (`tracker.rs`), not by a template.
    pub tracked: bool,
}

/// Peaks listed in explain mode: down to this z, at most T_EXPLAIN_MAX per image.
pub const T_EXPLAIN_MIN_Z: f64 = 3.0;
pub const T_EXPLAIN_MAX: usize = 20;

/// Template events in `img` not already among `existing` (ridge events). Each carries its z and
/// the rising null in `guide_z` / `guide_null_z` (`guided` stays false: no ADS-B was used).
pub fn template_events(
    img: &GuideImage,
    existing: &[Event],
    duration_s: f64,
    f0_range: (f64, f64),
) -> Vec<Event> {
    trials(img, existing, duration_s, f0_range, None)
        .into_iter()
        .filter_map(|(_, e)| e)
        .collect()
}

/// Every falling peak's verdict. With `window`, peaks whose CPA lies in it are all listed (down
/// to T_EXPLAIN_MIN_Z, at most T_EXPLAIN_MAX), not only those above the bar.
fn trials(
    img: &GuideImage,
    existing: &[Event],
    duration_s: f64,
    f0_range: (f64, f64),
    window: Option<(f64, f64)>,
) -> Vec<(TemplateTrial, Option<Event>)> {
    let n = img.times.len();
    if duration_s < T_MIN_AUDIO_S || n < 2 * T_STEP || img.hop_s <= 0.0 {
        return Vec::new();
    }
    let nb = img.n_bins();
    let q_lo = img.bin_of(f0_range.0).ceil().max(0.0) as usize;
    let q_hi = (img.bin_of(f0_range.1).floor().max(0.0) as usize).min(nb.saturating_sub(1));
    if q_lo >= q_hi {
        return Vec::new();
    }
    // Prefix sums over time of the clipped score image.
    let mut p: Vec<Vec<f32>> = Vec::with_capacity(n + 1);
    p.push(vec![0.0; nb]);
    for j in 0..n {
        let row = img.score_row(j);
        let next: Vec<f32> = p[j]
            .iter()
            .zip(row)
            .map(|(acc, &x)| acc + x.clamp(T_CLIP.0, T_CLIP.1))
            .collect();
        p.push(next);
    }
    let hop = img.hop_s;
    let mut fall: Vec<Template> = Vec::new();
    let mut rise: Vec<Template> = Vec::new();
    for &m in &logspace(T_MACH) {
        for &tau in &logspace(T_TAU_S) {
            fall.push(template(m, tau, hop, false));
            rise.push(template(m, tau, hop, true));
        }
    }
    let centres: Vec<usize> = (0..n).step_by(T_STEP).collect();
    let zf = scan(&p, &fall, q_lo, q_hi, &centres);
    let zr = scan(&p, &rise, q_lo, q_hi, &centres);
    let radius = (30.0 / hop).round() as usize;
    let rise_peaks = peaks(&zr, &centres, radius);
    // The recording's rising null: its strongest rising peak (the second past 30 min).
    let k = usize::from(duration_s > 1800.0);
    let Some(null) = rise_peaks.get(k).or(rise_peaks.first()).map(|&i| zr[i].z) else {
        return Vec::new();
    };
    let bar = T_MIN_Z.max(null + T_NULL_MARGIN);
    let mut out: Vec<Event> = Vec::new();
    let mut log: Vec<(TemplateTrial, Option<Event>)> = Vec::new();
    for i in peaks(&zf, &centres, radius) {
        let Best { z, t, q } = zf[i];
        let tm = &fall[t];
        let (v, d) = (tm.mach * C, tm.tau * tm.mach * C);
        let c = centres[i];
        let f0 = img.freq_of(q as f64);
        let tc = img.times[c];
        let mut trial = TemplateTrial {
            tc,
            f0,
            mach: tm.mach,
            tau: tm.tau,
            z,
            bar,
            verdict: TemplateVerdict::BelowBar,
            tracked: false,
        };
        let listed = |w: (f64, f64)| tc >= w.0 && tc <= w.1;
        if z < bar {
            match window {
                None => break,
                Some(_) if z < T_EXPLAIN_MIN_Z || log.len() >= T_EXPLAIN_MAX => break,
                Some(w) => {
                    if listed(w) {
                        log.push((trial, None));
                    }
                    continue;
                }
            }
        }
        let mut record = |verdict: TemplateVerdict, e: Option<Event>| {
            trial.verdict = verdict;
            if window.is_none_or(listed) {
                log.push((trial, e));
            }
        };
        let curve: Vec<(usize, f64)> = (c.saturating_sub(tm.half as usize)
            ..(c + tm.half as usize + 1).min(n))
            .map(|j| (j, f0 * shape(img.times[j] - tc, v, d, 0.0, C)))
            .collect();
        let pts = measure_along(img, &curve, 0.004, 1.0);
        if pts.len() < 40 {
            record(TemplateVerdict::FewPoints(pts.len()), None);
            continue;
        }
        let Some(mut e) = event_from_points(pts, z as f32, &EventParams::default()) else {
            record(TemplateVerdict::NoFit, None);
            continue;
        };
        // The fitted pass must be the template's own: CPA time, speed and distance agreeing
        // (a refit that drifts to another geometry is a coincidence of points, not this pass).
        let near = |x: f64, y: f64| x > 0.0 && y > 0.0 && (x / y).ln().abs() <= 2f64.ln();
        if vehicle_like(&e.fit) {
            record(TemplateVerdict::VehicleLike, None);
            continue;
        }
        if (e.fit.tc - tc).abs() > 10.0 || !near(e.fit.v, v) || !near(e.fit.d, d) {
            record(TemplateVerdict::Drifted, None);
            continue;
        }
        e.guided = false;
        e.guide_z = Some(z);
        e.guide_null_z = Some(null);
        if existing.iter().chain(&out).all(|x| {
            !same_pass(x, &e)
                && !rides_on(&e, x)
                && !on_family(&e, x)
                && (x.fit.tc - e.fit.tc).abs() > 10.0
        }) {
            out.push(e.clone());
            record(TemplateVerdict::Accepted, Some(e));
        } else {
            record(TemplateVerdict::Duplicate, None);
        }
    }
    // Tracked paths: partial and one-sided passes no full template fits. Each is measured,
    // fitted, and judged by the same statistic and bar: its fitted curve's sum against the same
    // curve shifted in f₀.
    // The tracker follows tones at their own frequency, over the image's band less its edges
    // (a 10 % guard: the anti-alias roll-off at the top standardises into heavy-tailed noise
    // that free paths ride).
    let guard = (GUIDE_BPO * 1.1f64.log2()).round() as usize;
    // On the prop image only up to TRACK_PROP_MAX_HZ: above it there is no harmonic sum and
    // the long windows make noise persist across frames; the jet image covers that band.
    let top = if f0_range.1 <= T_F0.1 {
        (img.bin_of(TRACK_PROP_MAX_HZ).floor().max(0.0) as usize).min(nb.saturating_sub(guard + 1))
    } else {
        nb.saturating_sub(guard + 1)
    };
    let (t_lo, t_hi) = (guard, top);
    for path in track_paths(img, t_lo, t_hi, &TrackParams::default()) {
        let curve: Vec<(usize, f64)> = path
            .cells
            .iter()
            .map(|&(j, q)| (j, img.freq_of(q as f64)))
            .collect();
        let (j0, j1) = (path.cells[0].0, path.cells[path.cells.len() - 1].0);
        let pts = measure_along(img, &curve, 0.004, 1.0);
        let mut trial = TemplateTrial {
            tc: img.times[(j0 + j1) / 2],
            f0: curve[0].1,
            mach: 0.0,
            tau: 0.0,
            z: 0.0,
            bar,
            verdict: TemplateVerdict::FewPoints(pts.len()),
            tracked: true,
        };
        let listed = |t: &TemplateTrial| window.is_none_or(|w| t.tc >= w.0 && t.tc <= w.1);
        if pts.len() < 32 {
            if listed(&trial) {
                log.push((trial, None));
            }
            continue;
        }
        let Some(mut e) = event_from_points(pts, 0.0, &EventParams::default()) else {
            trial.verdict = TemplateVerdict::NoFit;
            if listed(&trial) {
                log.push((trial, None));
            }
            continue;
        };
        let f = e.fit;
        trial.tc = f.tc;
        trial.f0 = f.f0;
        trial.mach = f.v / C;
        trial.tau = f.d / f.v.max(1e-9);
        trial.z = curve_z(img, &p, &f, (j0, j1), (t_lo, t_hi));
        trial.verdict = if vehicle_like(&f) {
            TemplateVerdict::VehicleLike
        } else if trial.z < bar {
            TemplateVerdict::BelowBar
        } else if existing.iter().chain(&out).any(|x| {
            same_pass(x, &e)
                || rides_on(&e, x)
                || on_family(&e, x)
                || (x.fit.tc - f.tc).abs() <= 10.0
        }) {
            TemplateVerdict::Duplicate
        } else {
            TemplateVerdict::Accepted
        };
        let accepted = trial.verdict == TemplateVerdict::Accepted;
        if accepted {
            e.guided = false;
            e.guide_z = Some(trial.z);
            e.guide_null_z = Some(null);
            e.mean_snr_db = trial.z as f32;
            out.push(e.clone());
        }
        if listed(&trial) && (window.is_some() || accepted) {
            log.push((trial, accepted.then_some(e)));
        }
    }
    log
}

/// The z of a fitted pass over frames `j0..=j1`: its curve's clipped sum on the score image
/// (prefix sums `p`), against the same curve shifted to every f₀ bin in `q_lo..=q_hi` — the
/// template statistic for a curve of any length.
fn curve_z(
    img: &GuideImage,
    p: &[Vec<f32>],
    f: &crate::audio::CvFit,
    (j0, j1): (usize, usize),
    (q_lo, q_hi): (usize, usize),
) -> f64 {
    let nb = img.n_bins() as i64;
    let width = q_hi - q_lo + 1;
    let mut b = vec![0f32; width];
    for j in j0..=j1 {
        let k = shape(img.times[j], f.v, f.d, f.tc, C);
        if k.is_nan() || k <= 0.0 {
            continue;
        }
        let s = (GUIDE_BPO * k.log2()).round() as i64;
        let (row_a, row_b) = (&p[j], &p[j + 1]);
        for (i, x) in b.iter_mut().enumerate() {
            let q = (q_lo + i) as i64 + s;
            if (0..nb).contains(&q) {
                *x += row_b[q as usize] - row_a[q as usize];
            }
        }
    }
    let mut buf = Vec::with_capacity(width);
    let (med, mad) = robust(&b, &mut buf);
    let q0 = img.bin_of(f.f0).round() as i64 - q_lo as i64;
    let at = (q0 - 2..=q0 + 2)
        .filter(|&i| (0..width as i64).contains(&i))
        .map(|i| b[i as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    if mad <= 0.0 || !at.is_finite() {
        return 0.0;
    }
    (f64::from(at) - med) / mad
}

/// The template stage's rejection log for `window` (seconds), on both guide images computed over
/// the whole recording; `existing` are the ridge-stage events.
pub fn explain_templates(a: &Audio, existing: &[Event], window: (f64, f64)) -> Vec<String> {
    let duration = a.samples.len() as f64 / f64::from(a.sample_rate.max(1));
    if duration < T_MIN_AUDIO_S {
        return vec![format!(
            "templates: abstain (audio {duration:.0} s < {T_MIN_AUDIO_S:.0} s)"
        )];
    }
    let mut lines = Vec::new();
    let mut seen: Vec<Event> = existing.to_vec();
    for (name, params, f0) in [
        ("prop", GuideParams::prop(), T_F0),
        ("jet", GuideParams::jet(), T_F0_JET),
    ] {
        let img = guide_image_with(a, &params);
        for (t, e) in trials(&img, &seen, duration, f0, Some(window)) {
            let verdict = match t.verdict {
                TemplateVerdict::BelowBar => "below bar".to_string(),
                TemplateVerdict::FewPoints(n) => format!("few points ({n})"),
                TemplateVerdict::NoFit => "no fit".into(),
                TemplateVerdict::Drifted => "refit drifted from the template".into(),
                TemplateVerdict::VehicleLike => "road-vehicle geometry".into(),
                TemplateVerdict::Duplicate => "duplicate of an earlier event".into(),
                TemplateVerdict::Accepted => "accepted".into(),
            };
            let stage = if t.tracked { "tracker" } else { "templates" };
            lines.push(format!(
                "{stage}/{name}  tc {:.1} s  f0 {:.1} Hz  M {:.2} τ {:.1} s  z {:.1} (bar {:.1})  {verdict}",
                t.tc, t.f0, t.mach, t.tau, t.z, t.bar
            ));
            if let Some(e) = e {
                seen.push(e);
            }
        }
    }
    lines
}

/// `template_events` on `a`'s own guide images: propeller/rotor combs, then jet tones (on the
/// 8 kHz jet image, 300–3400 Hz), each deduplicated against everything found before it.
pub fn discover_templates(a: &Audio, existing: &[Event]) -> Vec<Event> {
    let duration = a.samples.len() as f64 / f64::from(a.sample_rate.max(1));
    if duration < T_MIN_AUDIO_S {
        return Vec::new();
    }
    let mut out = template_events(
        &guide_image_with(a, &GuideParams::prop()),
        existing,
        duration,
        T_F0,
    );
    let seen: Vec<Event> = existing.iter().chain(&out).cloned().collect();
    out.extend(template_events(
        &guide_image_with(a, &GuideParams::jet()),
        &seen,
        duration,
        T_F0_JET,
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_explain_abstains_on_short_audio() {
        let a = Audio::new(8000, vec![0.0; 8000 * 60]);
        let lines = explain_templates(&a, &[], (0.0, 60.0));
        assert_eq!(
            lines,
            vec!["templates: abstain (audio 60 s < 300 s)".to_string()]
        );
    }

    #[test]
    fn tracker_finds_faint_jet_tone() {
        // The find must come from the tracker itself (not a template that drifted over its bar).
        let (a, truth) = crate::audio::ridges::tests::sim("faint_jet.toml", |s| s);
        let t = &truth.events[0];
        let duration = a.samples.len() as f64 / f64::from(a.sample_rate);
        let img = guide_image_with(&a, &GuideParams::jet());
        let near = |x: f64, y: f64| (x / y).ln().abs() <= 2f64.ln();
        let found = trials(&img, &[], duration, T_F0_JET, None);
        assert!(
            found.iter().any(|(tr, e)| tr.tracked
                && tr.verdict == TemplateVerdict::Accepted
                && e.as_ref()
                    .is_some_and(|e| (e.fit.tc - t.cpa_arrival_audio_s).abs() <= 3.0
                        && near(e.fit.v, t.speed_mps)
                        && near(e.fit.d, t.slant_m))),
            "{:?}",
            found
                .iter()
                .map(|(tr, _)| (tr.tracked, tr.tc, tr.z, tr.verdict))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn template_explain_lists_accepted_pass() {
        // one_pass.toml stretched to 600 s with its pass at 300 s: listed as accepted at its CPA.
        let (a, truth) = crate::audio::ridges::tests::sim("one_pass.toml", |s| {
            s.replace("duration_s = 120.0", "duration_s = 600.0")
                .replace("cpa_time_s = 55.0", "cpa_time_s = 300.0")
        });
        let t = truth.events[0].cpa_arrival_audio_s;
        let lines = explain_templates(&a, &[], (t - 30.0, t + 30.0));
        assert!(lines.iter().any(|l| l.contains("accepted")), "{lines:#?}");
    }

    #[test]
    fn rising_template_is_the_time_reverse() {
        let (f, r) = (
            template(0.3, 20.0, 0.25, false),
            template(0.3, 20.0, 0.25, true),
        );
        let flat = |t: &Template| -> Vec<i64> {
            t.runs
                .iter()
                .flat_map(|r| (r.a..r.b).map(move |_| r.s))
                .collect()
        };
        let (a, mut b) = (flat(&f), flat(&r));
        b.reverse();
        assert_eq!(a, b);
        // Falling: early frames above the rest frequency, late ones below.
        assert!(a[0] > 0 && a[a.len() - 1] < 0);
    }
}
