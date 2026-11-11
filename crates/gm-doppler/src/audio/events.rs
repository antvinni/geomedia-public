//! Doppler events: ridges fitted with the constant-velocity model, fragments and harmonics of
//! the same aircraft grouped by agreeing fits, then refitted together.

use serde::Serialize;

use super::decode::Audio;
use super::fit::{CvFit, fit_cv};
use super::resample::{ANALYSIS_RATE, decimate};
use super::ridges::{Ridge, RidgeParams, fragments, ridges};
use super::spectrogram::{SpecParams, Spectrogram, multi_spectrogram, spectrogram};
use crate::physics::cv::{best_f0, shape};

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct Event {
    pub id: usize,
    pub t_start: f64,
    pub t_end: f64,
    pub fit: CvFit,
    /// Harmonic number of the reference ridge relative to the inferred fundamental.
    pub harmonic: u32,
    pub f0_fundamental: f64,
    pub ridges: usize,
    pub mean_snr_db: f32,
    #[serde(default)]
    /// (t, f) of every member ridge, scaled onto the reference harmonic (so `fit.f0·shape(t)`
    /// models them), outliers removed, sorted by t.
    pub points: Vec<(f64, f64)>,
    /// Found by guided curve scoring (a real aircraft's predicted curve, measured along), not
    /// by free discovery.
    #[serde(default)]
    pub guided: bool,
    /// The aircraft whose predicted curve proposed a guided event, and that curve's score.
    #[serde(default)]
    pub guide_icao: Option<String>,
    #[serde(default)]
    pub guide_z: Option<f64>,
    /// The largest z the curve's null reaches: the same curve at other times of the recording,
    /// and decoy aircraft re-timed onto it.
    #[serde(default)]
    pub guide_null_z: Option<f64>,
    /// How many values that null took: (in-recording shifts, decoy aircraft).
    #[serde(default)]
    pub guide_null_n: Option<(usize, usize)>,
}

/// What kind of source an event's tone suggests, from features discovery already has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SourceClass {
    /// A harmonic comb (several harmonics sharing one Doppler ratio) or a low tone: propellers
    /// and rotors.
    Prop,
    /// A single tone above the comb band, with no lower harmonic: an airframe or engine tone.
    Jet,
    Unknown,
}

/// Below this a lone tone is a blade-pass fundamental, not a jet tone.
pub const CLASS_PROP_MAX_HZ: f64 = 300.0;
/// Above this a lone tone with no lower harmonic is a jet tone.
pub const CLASS_JET_MIN_HZ: f64 = 400.0;

/// The source class `e`'s tone suggests (Phase 5: hand rules first).
pub fn source_class(e: &Event) -> SourceClass {
    let comb = e.harmonic > 1 || e.f0_fundamental < 0.9 * e.fit.f0;
    if comb || e.fit.f0 < CLASS_PROP_MAX_HZ {
        SourceClass::Prop
    } else if e.fit.f0 >= CLASS_JET_MIN_HZ {
        SourceClass::Jet
    } else {
        SourceClass::Unknown
    }
}

/// How ridges become events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    /// Pass pursuit (`pursuit.rs`): fragments of any length, grouped on one curve.
    #[default]
    Pursuit,
    /// The earlier stage: whole ridges, each fitted alone, then grouped by harmonic ratio.
    Ridges,
}

/// The band above the first one: single engine and airframe tones. An 1800 Hz tone passing at
/// 80 m/s is heard from 2350 Hz down to 1460 Hz; without this band pass pursuit gets only the
/// end of its fall, and fits a slower, nearer pass to it.
#[derive(Clone, Debug)]
pub struct HighBand {
    /// Sample rate the band is analysed at. A recording sampled below it is analysed as it
    /// is: the band then ends where the recording does, and may be empty.
    pub rate: u32,
    pub spec: SpecParams,
    /// STFT window lengths (s), max-combined after whitening.
    pub windows_s: Vec<f64>,
}

impl Default for HighBand {
    fn default() -> Self {
        Self {
            rate: 8000,
            // From the top of the first band (no overlap: a tone in both would count twice)
            // to the top of the template stage's jet band.
            spec: SpecParams {
                zero_pad: 2,
                f_min: SpecParams::default().f_max,
                f_max: 3400.0,
                ..SpecParams::default()
            },
            // A pass sweeps a tone in proportion to its frequency: windows half as long as
            // the first band's.
            windows_s: vec![0.25, 0.5, 1.0],
        }
    }
}

#[derive(Clone, Debug)]
pub struct EventParams {
    pub stage: Stage,
    pub spec: SpecParams,
    /// STFT window lengths (s), max-combined after whitening; empty = `spec.window_s` only.
    pub windows_s: Vec<f64>,
    /// A second band above `spec`'s, for pass pursuit; `None` = `spec`'s band only.
    pub high: Option<HighBand>,
    pub ridge: RidgeParams,
    /// Sound speed used by the fit.
    pub c: f64,
}

impl Default for EventParams {
    fn default() -> Self {
        Self {
            stage: Stage::default(),
            spec: SpecParams::default(),
            // Long windows lift faint, slow sections (suite/quiet far pass: +0.7 dB mean along
            // its exact curve); the 0.5 s window keeps the steep sweep near CPA in one piece
            // (without it two_props' B350 harmonic breaks up around CPA).
            windows_s: vec![0.5, 1.0, 2.0],
            high: Some(HighBand::default()),
            ridge: RidgeParams::default(),
            c: 343.0,
        }
    }
}

/// All events in `a`; `windows` (seconds) restricts the search, empty = whole file.
pub fn discover(a: &Audio, p: &EventParams, windows: &[(f64, f64)]) -> Vec<Event> {
    let d = decimate(a, ANALYSIS_RATE);
    // The harmonic sum (`comb.rs`) is not used here: in free discovery it multiplies noise
    // ridges and sub-harmonic copies (two_props: 25 → 90 ridges, 4 junk events). It serves the
    // guided curve scoring, which sums along modelled harmonics only.
    let spec = if p.windows_s.is_empty() {
        spectrogram(&d, &p.spec)
    } else {
        multi_spectrogram(&d, &p.windows_s, &p.spec)
    };
    let mut events = match p.stage {
        Stage::Pursuit => {
            super::pursuit::events_from_fragments(pursuit_fragments(a, &spec, p), p, windows)
        }
        Stage::Ridges => events_from_ridges(ridges(&spec, &p.ridge), p, windows),
    };
    // Blind Doppler templates find passes too faint to follow as ridges (whole file only:
    // `windows` restrict ridges to user-chosen spans).
    if windows.is_empty() {
        let more = super::templates::discover_templates(a, &events);
        events.extend(more);
    }
    events.sort_by(|x, y| x.fit.tc.total_cmp(&y.fit.tc));
    for (i, e) in events.iter_mut().enumerate() {
        e.id = i + 1;
    }
    events
}

/// The fragments pass pursuit works on: those of `spec` (the first band's spectrogram of `a`)
/// and those of the band above it, by start time. A tone that falls from one band into the
/// other arrives as two fragments, which pursuit joins like any two pieces of one pass.
fn pursuit_fragments(a: &Audio, spec: &Spectrogram, p: &EventParams) -> Vec<Ridge> {
    let mut out = fragments(spec, &p.ridge);
    if let Some(h) = &p.high {
        let s = multi_spectrogram(&decimate(a, h.rate), &h.windows_s, &h.spec);
        out.extend(fragments(&s, &p.ridge));
        out.sort_by(|x, y| x.t_start().total_cmp(&y.t_start()));
    }
    out
}

/// The ridge stage of discovery: fit each ridge, group harmonics of one pass, and keep the
/// groups whose combined fit is an aircraft's (ids unset, unsorted).
pub fn events_from_ridges(rs: Vec<Ridge>, p: &EventParams, windows: &[(f64, f64)]) -> Vec<Event> {
    let in_window =
        |t: f64| windows.is_empty() || windows.iter().any(|&(lo, hi)| t >= lo && t <= hi);
    let mut cands: Vec<(Ridge, Option<CvFit>)> = rs
        .into_iter()
        .filter_map(|mut r| {
            r.points.retain(|q| in_window(q.t));
            if r.points.len() < 8 {
                return None;
            }
            let pts: Vec<(f64, f64)> = r.points.iter().map(|q| (q.t, q.f)).collect();
            let (t0, t1) = (r.t_start(), r.t_end());
            let fit = fit_cv(&pts, p.c)
                .filter(|f| plausible(f, p.c) && f.tc > t0 - 15.0 && f.tc < t1 + 15.0);
            Some((r, fit))
        })
        .collect();
    // Strongest first: well-fitted, long, loud ridges seed events; the rest attach to them.
    let weight =
        |(r, f): &(Ridge, Option<CvFit>)| (f.is_some(), r.points.len() as f32 * r.mean_snr_db());
    cands.sort_by(|a, b| {
        let (wa, wb) = (weight(a), weight(b));
        wb.0.cmp(&wa.0).then(wb.1.total_cmp(&wa.1))
    });
    let mut seeds: Vec<(CvFit, Vec<usize>)> = Vec::new();
    for (idx, (r, fit)) in cands.iter().enumerate() {
        let attach = seeds.iter_mut().find(|(seed, _)| fits_model(r, seed, p.c));
        match (attach, fit) {
            (Some((_, members)), _) => members.push(idx),
            // A ridge whose own fit is a road vehicle's may join an aircraft's group (it fits
            // that curve) but never starts one: refitted with its neighbours, a vehicle group
            // drifts past the veto below (street audio: 31 m/s at 135 m → 34 m/s at 228 m).
            (None, Some(f)) if !vehicle_like(f) => seeds.push((*f, vec![idx])),
            (None, _) => {}
        }
    }
    // A seed fitted on one fragment may miss the rest of its own pass; refit each group on all
    // its ridges, merge groups whose curves agree, and re-offer every ridge, until stable.
    for _ in 0..5 {
        let mut changed = false;
        for (seed, members) in seeds.iter_mut() {
            if let Some(f) = refit(&cands, seed, members, p.c) {
                *seed = f;
            }
        }
        let mut i = 0;
        while i < seeds.len() {
            let mut j = i + 1;
            while j < seeds.len() {
                let (head, tail) = seeds.split_at_mut(j);
                let (a, b) = (&head[i], &tail[0]);
                if b.1.iter().all(|&m| fits_model(&cands[m].0, &a.0, p.c))
                    || joint_fit_ok(&cands, a, b, p.c)
                {
                    let moved = seeds.remove(j).1;
                    seeds[i].1.extend(moved);
                    changed = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        let assigned: Vec<usize> = seeds.iter().flat_map(|(_, m)| m.iter().copied()).collect();
        for idx in (0..cands.len()).filter(|i| !assigned.contains(i)) {
            if let Some((_, members)) = seeds
                .iter_mut()
                .find(|(seed, _)| fits_model(&cands[idx].0, seed, p.c))
            {
                members.push(idx);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let corr = (p.spec.window_s / p.spec.hop_s).max(1.0).sqrt();
    seeds
        .iter()
        .filter_map(|(seed, m)| combine(&cands, seed, m, p.c))
        .filter(|e| !vehicle_like(&e.fit))
        .map(|mut e| {
            inflate_sigma(&mut e.fit, corr);
            e
        })
        .collect()
}

/// The log of discovery for `window` (seconds). Pass pursuit: one line per seed (its vote,
/// joint fit, tracks and verdict), per mirrored seed, and per pass held against the mirror,
/// from the search that `discover` runs with the same `windows`.
/// The ridge stage: one line per merged ridge piece, saying why it was dropped — at the ridge
/// stage (length, SNR, drop, direction) or at the fit (no fit, implausible, CPA outside the
/// ridge) — or that it became a candidate.
pub fn explain(
    a: &Audio,
    p: &EventParams,
    windows: &[(f64, f64)],
    window: (f64, f64),
) -> Vec<String> {
    let d = decimate(a, ANALYSIS_RATE);
    let spec = if p.windows_s.is_empty() {
        spectrogram(&d, &p.spec)
    } else {
        multi_spectrogram(&d, &p.windows_s, &p.spec)
    };
    if p.stage == Stage::Pursuit {
        // The pursuit works on looser fragments than the ridge lines further down describe.
        return super::pursuit::explain_fragments(
            pursuit_fragments(a, &spec, p),
            p,
            windows,
            window,
        );
    }
    super::ridges::ridges_explained(&spec, &p.ridge)
        .into_iter()
        .filter(|(r, _)| r.t_end() >= window.0 && r.t_start() <= window.1)
        .map(|(r, verdict)| {
            let (f_a, f_b) = (r.points[0].f, r.points[r.points.len() - 1].f);
            let head = format!(
                "{:7.1}-{:7.1} s  {:6.1}->{:6.1} Hz  {:4} pts  snr {:4.1} dB",
                r.t_start(),
                r.t_end(),
                f_a,
                f_b,
                r.points.len(),
                r.mean_snr_db()
            );
            let why = match verdict {
                Err(e) => format!("ridge: {e}"),
                Ok(()) => {
                    let pts: Vec<(f64, f64)> = r.points.iter().map(|q| (q.t, q.f)).collect();
                    match fit_cv(&pts, p.c) {
                        None => "fit: none (no Doppler drop or fit failed)".into(),
                        Some(f) if !plausible(&f, p.c) => format!(
                            "fit: implausible (v {:.0} m/s, d {:.0} m, rms {:.2} Hz)",
                            f.v, f.d, f.rms_hz
                        ),
                        Some(f) if f.tc <= r.t_start() - 15.0 || f.tc >= r.t_end() + 15.0 => {
                            format!("fit: CPA {:.1} s outside the ridge ±15 s", f.tc)
                        }
                        Some(f) if vehicle_like(&f) => format!(
                            "fit: road-vehicle geometry (v {:.0} m/s, d {:.0} m)",
                            f.v, f.d
                        ),
                        Some(f) => format!(
                            "candidate (tc {:.1} s, v {:.0} m/s, d {:.0} m)",
                            f.tc, f.v, f.d
                        ),
                    }
                }
            };
            format!("{head}  {why}")
        })
        .collect()
}

/// An event from (t, f) points measured along a known curve (guided scoring): fitted, checked
/// for plausibility like a discovered ridge, and its σ inflated for frame overlap. `None` when
/// the points don't fit a pass.
pub fn event_from_points(
    mut points: Vec<(f64, f64)>,
    mean_snr_db: f32,
    p: &EventParams,
) -> Option<Event> {
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (t0, t1) = (points.first()?.0, points.last()?.0);
    let mut fit = fit_cv(&points, p.c)
        .filter(|f| plausible(f, p.c) && f.tc > t0 - 15.0 && f.tc < t1 + 15.0)?;
    inflate_sigma(&mut fit, (p.spec.window_s / p.spec.hop_s).max(1.0).sqrt());
    let cut = (5.0 * fit.rms_hz).max(1.0);
    points.retain(|&(t, f)| (f - fit.f0 * shape(t, fit.v, fit.d, fit.tc, p.c)).abs() <= cut);
    Some(Event {
        id: 0,
        t_start: t0,
        t_end: t1,
        fit,
        harmonic: 1,
        f0_fundamental: fit.f0,
        ridges: 0,
        mean_snr_db,
        points,
        guided: true,
        guide_icao: None,
        guide_z: None,
        guide_null_z: None,
        guide_null_n: None,
    })
}

/// Road-vehicle geometry (B §5: cars sweep in ~2 s at 6–15 %/s; aircraft over 15–60 s): a
/// fitted closest approach under VEHICLE_MAX_D_M, or a fast sweep (τ = d/v under VEHICLE_TAU_S)
/// at road speed (under VEHICLE_MAX_V_MPS). Applies to free discovery only: a guided event comes
/// from a real ADS-B track (a low helicopter is legitimate).
pub const VEHICLE_MAX_D_M: f64 = 150.0;
pub const VEHICLE_TAU_S: f64 = 4.0;
pub const VEHICLE_MAX_V_MPS: f64 = 40.0;

pub fn vehicle_like(f: &CvFit) -> bool {
    f.d < VEHICLE_MAX_D_M || (f.d / f.v < VEHICLE_TAU_S && f.v < VEHICLE_MAX_V_MPS)
}

/// A pass's audible span around CPA: ±(this × d/v) + PASS_SPAN_PAD_S.
pub const PASS_SPAN_D_OVER_V: f64 = 3.0;
pub const PASS_SPAN_PAD_S: f64 = 60.0;

/// Does ridge `r` follow the seed's curve at some small-integer harmonic scale?
fn fits_model(r: &Ridge, seed: &CvFit, c: f64) -> bool {
    // Only ridges overlapping this pass's own time span: far from CPA the curve is a flat
    // asymptote that any steady tone at a harmonic-looking ratio would "fit".
    let half = PASS_SPAN_D_OVER_V * seed.d / seed.v.max(1.0) + PASS_SPAN_PAD_S;
    if r.t_end() < seed.tc - half || r.t_start() > seed.tc + half {
        return false;
    }
    let pts: Vec<(f64, f64)> = r.points.iter().map(|q| (q.t, q.f)).collect();
    let Some(q) = best_f0(&pts, seed.v, seed.d, seed.tc, c) else {
        return false;
    };
    if harmonic_ratio(q, seed.f0).is_none() {
        return false;
    }
    let rms = (pts
        .iter()
        .map(|&(t, f)| (f - q * shape(t, seed.v, seed.d, seed.tc, c)).powi(2))
        .sum::<f64>()
        / pts.len() as f64)
        .sqrt();
    // Residuals scale with frequency; allow 3× the seed's (or 0.2 Hz) at the seed's tone.
    rms <= 3.0 * seed.rms_hz.max(0.2) * (q / seed.f0).max(1.0)
}

/// Overlapping frames are not independent (×√(window/hop)), and the constant-velocity, still-air
/// model has its own error (wind, curvature), so add a floor. Keeps σ honest for scoring.
pub(super) fn inflate_sigma(f: &mut CvFit, corr: f64) {
    f.sigma_v = (f.sigma_v * corr).max(0.005 * f.v + 0.2);
    f.sigma_d = (f.sigma_d * corr).max(0.01 * f.d + 10.0);
    f.sigma_tc = (f.sigma_tc * corr).max(0.05);
}

/// Two groups close in time are one pass if a joint fit (b scaled onto a's harmonic) is about
/// as good as either alone: halves of a pass fit each other's curve badly on their own, but
/// a joint fit of two different aircraft is poor.
fn joint_fit_ok(
    cands: &[(Ridge, Option<CvFit>)],
    a: &(CvFit, Vec<usize>),
    b: &(CvFit, Vec<usize>),
    c: f64,
) -> bool {
    let span = |m: &[usize]| {
        m.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &i| {
                (lo.min(cands[i].0.t_start()), hi.max(cands[i].0.t_end()))
            })
    };
    let ((a0, a1), (b0, b1)) = (span(&a.1), span(&b.1));
    if b0 > a1 + 10.0 || a0 > b1 + 10.0 {
        return false;
    }
    // Candidate scales for b → a: the junction frequency ratio, the f0 ratio, and 1:1,
    // each snapped to a small-integer harmonic ratio; one-sided fits bias f0, junctions can be
    // contaminated, so try all and keep the best joint fit.
    let junction = if b0 <= a1 {
        0.5 * (b0.max(a0) + a1.min(b1))
    } else {
        0.5 * (a1 + b0)
    };
    let near = |m: &[usize], t: f64| {
        m.iter()
            .flat_map(|&i| cands[i].0.points.iter())
            .min_by(|x, y| (x.t - t).abs().total_cmp(&(y.t - t).abs()))
            .map(|q| q.f)
    };
    let mut scales = vec![1.0];
    let mut add = |x: f64, y: f64| {
        if let Some((k, m)) = harmonic_ratio(x, y) {
            let r = f64::from(k) / f64::from(m);
            scales.push(if y > x { 1.0 / r } else { r });
        }
    };
    if let (Some(fa), Some(fb)) = (near(&a.1, junction), near(&b.1, junction)) {
        add(fa, fb);
    }
    add(a.0.f0, b.0.f0);
    scales.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
    let limit = 1.5 * a.0.rms_hz.max(b.0.rms_hz) + 0.1;
    scales.into_iter().any(|scale| {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for &i in &a.1 {
            pts.extend(cands[i].0.points.iter().map(|q| (q.t, q.f)));
        }
        for &i in &b.1 {
            pts.extend(cands[i].0.points.iter().map(|q| (q.t, q.f * scale)));
        }
        pts.sort_by(|x, y| x.0.total_cmp(&y.0));
        fit_cv(&pts, c)
            .filter(|f| plausible(f, c))
            .is_some_and(|j| j.rms_hz <= limit)
    })
}

/// The group's fit on all its ridges, each scaled onto the seed's harmonic.
fn refit(
    cands: &[(Ridge, Option<CvFit>)],
    seed: &CvFit,
    members: &[usize],
    c: f64,
) -> Option<CvFit> {
    if members.len() < 2 {
        return None;
    }
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for &i in members {
        let rp: Vec<(f64, f64)> = cands[i].0.points.iter().map(|q| (q.t, q.f)).collect();
        let tone = best_f0(&rp, seed.v, seed.d, seed.tc, c)?;
        let scale = seed.f0 / tone;
        pts.extend(rp.iter().map(|&(t, f)| (t, f * scale)));
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    fit_cv(&pts, c).filter(|f| plausible(f, c))
}

/// A fit worth keeping: finite, constrained, and not a bound-hugging artefact.
pub(super) fn plausible(f: &CvFit, c: f64) -> bool {
    // Fits pinned at the search bounds (20 m/s, 0.8c, 50 m, 20 km) are artefacts, not passes.
    let pinned = f.v <= 20.0 * 1.05 || f.v >= 0.8 * c * 0.97 || f.d <= 60.0 || f.d >= 19_000.0;
    !pinned
        && [
            f.v, f.d, f.tc, f.f0, f.rms_hz, f.sigma_v, f.sigma_d, f.sigma_tc,
        ]
        .iter()
        .all(|x| x.is_finite())
        && f.sigma_d < f.d
        && f.sigma_v < 0.2 * f.v
        && f.f0 > 0.0
        && f.rms_hz <= 0.01 * f.f0.max(100.0)
}

/// The small-integer ratio k/m (k, m ≤ 6) within 2% of hi/lo, if any.
pub(super) fn harmonic_ratio(x: f64, y: f64) -> Option<(u32, u32)> {
    let r = x.max(y) / x.min(y);
    (1..=6u32)
        .flat_map(|m| (m..=6 * m).map(move |k| (k, m)))
        .filter(|&(k, _)| k <= 6)
        .find(|&(k, m)| {
            let q = f64::from(k) / f64::from(m);
            (r - q).abs() / q < 0.02
        })
}

/// Refit a group's ridges together, each scaled onto the reference ridge's harmonic.
fn combine(
    cands: &[(Ridge, Option<CvFit>)],
    seed: &CvFit,
    members: &[usize],
    c: f64,
) -> Option<Event> {
    // Each member's harmonic scale against the seed, solved in closed form.
    let scale_of = |i: usize| {
        let pts: Vec<(f64, f64)> = cands[i].0.points.iter().map(|q| (q.t, q.f)).collect();
        best_f0(&pts, seed.v, seed.d, seed.tc, c).unwrap_or(seed.f0)
    };
    let tones: Vec<f64> = members.iter().map(|&i| scale_of(i)).collect();
    let mut pts: Vec<(f64, f64)> = Vec::new();
    let (mut t0, mut t1, mut snr_sum, mut snr_n) =
        (f64::INFINITY, f64::NEG_INFINITY, 0.0f32, 0usize);
    for (&i, &tone) in members.iter().zip(&tones) {
        let r = &cands[i].0;
        let scale = seed.f0 / tone;
        pts.extend(r.points.iter().map(|q| (q.t, q.f * scale)));
        t0 = t0.min(r.t_start());
        t1 = t1.max(r.t_end());
        snr_sum += r.points.iter().map(|q| q.snr_db).sum::<f32>();
        snr_n += r.points.len();
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let fit = if members.len() == 1 {
        *seed
    } else {
        fit_cv(&pts, c).filter(|f| plausible(f, c)).unwrap_or(*seed)
    };
    // The fundamental: the lowest tone, divided down if the tones share an unseen lower base.
    let lowest = tones.iter().copied().fold(f64::INFINITY, f64::min);
    let base = tones
        .iter()
        .filter_map(|&t| harmonic_ratio(t, lowest))
        .map(|(_, m)| m)
        .max()
        .unwrap_or(1);
    let f0_fundamental = lowest / f64::from(base);
    let harmonic = (fit.f0 / f0_fundamental).round().max(1.0) as u32;
    Some(Event {
        id: 0,
        t_start: t0,
        t_end: t1,
        fit,
        harmonic,
        f0_fundamental,
        ridges: members.len(),
        mean_snr_db: snr_sum / snr_n.max(1) as f32,
        guided: false,
        guide_icao: None,
        guide_z: None,
        guide_null_z: None,
        guide_null_n: None,
        points: {
            // Drop points far off the final fit (contamination the robust fit trimmed).
            let cut = (5.0 * fit.rms_hz).max(1.0);
            pts.into_iter()
                .filter(|&(t, f)| (f - fit.f0 * shape(t, fit.v, fit.d, fit.tc, c)).abs() <= cut)
                .collect()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::ridges::tests::sim;

    fn close(ev: &Event, truth: &gm_sim::TruthEvent) {
        assert!(
            (ev.fit.tc - truth.cpa_arrival_audio_s).abs() < 0.3,
            "tc {ev:?} vs {truth:?}"
        );
        assert!(
            (ev.fit.v - truth.speed_mps).abs() / truth.speed_mps < 0.03,
            "v {ev:?} vs {truth:?}"
        );
        assert!(
            (ev.fit.d - truth.slant_m).abs() / truth.slant_m < 0.07,
            "d {ev:?} vs {truth:?}"
        );
    }

    #[test]
    fn source_class_separates_comb_from_jet_tone() {
        // The faint jet bench cell: its 650 Hz tone is a Jet; two_props' combs are Props.
        let (a, _) = sim("faint_jet.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        assert!(
            ev.iter().any(|e| source_class(e) == SourceClass::Jet),
            "{ev:?}"
        );
        let (a, _) = sim("two_props.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        assert!(!ev.is_empty() && ev.iter().all(|e| source_class(e) == SourceClass::Prop));
    }

    /// one_pass.toml with the aircraft's straight-line geometry replaced.
    fn pass_with(speed: f64, alt: f64, offset: f64) -> String {
        format!("speed_mps = {speed:.1}\nalt_agl_m = {alt:.1}\ncpa_offset_m = {offset:.1}")
    }

    #[test]
    fn vehicle_pass_vetoed() {
        // A road vehicle's geometry, as fitted on real street audio: 28 m/s at ~78 m (τ = d/v
        // ≈ 2.8 s). Its Doppler sweep is real, but it is not an aircraft: no event.
        let (a, _) = sim("one_pass.toml", |s| {
            s.replace(
                "speed_mps = 140.0\nalt_agl_m = 1800.0\ncpa_offset_m = 1500.0",
                &pass_with(28.0, 50.0, 60.0),
            )
        });
        let ev = discover(&a, &EventParams::default(), &[]);
        assert!(ev.is_empty(), "{ev:#?}");
    }

    #[test]
    fn vehicle_seed_cannot_grow_into_an_event() {
        // Real street audio (a city-centre walk, 250–310 s): a ~38 Hz engine rumble whose own
        // ridge fits as a road vehicle (31 m/s at 135 m). Grouped with its neighbours, the refit
        // drifted to 34 m/s at 228 m and passed the veto as a "Prop" event. Ridge points only.
        let raw: Vec<Vec<[f64; 3]>> = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/ridges/street_vehicle.json"
        ))
        .unwrap();
        let rs: Vec<Ridge> = raw
            .into_iter()
            .map(|r| Ridge {
                points: r
                    .into_iter()
                    .map(|[t, f, snr]| crate::audio::RidgePoint {
                        t,
                        f,
                        snr_db: snr as f32,
                    })
                    .collect(),
            })
            .collect();
        let ev = events_from_ridges(rs.clone(), &EventParams::default(), &[]);
        assert!(ev.is_empty(), "{ev:#?}");
        let ev = crate::audio::pursuit::events_from_fragments(rs, &EventParams::default(), &[]);
        assert!(ev.is_empty(), "{ev:#?}");
    }

    #[test]
    fn low_slow_aircraft_kept() {
        // 40 m/s at 300 m (τ 7.5 s): a slow, low aircraft or helicopter, kept.
        let (a, truth) = sim("one_pass.toml", |s| {
            s.replace(
                "speed_mps = 140.0\nalt_agl_m = 1800.0\ncpa_offset_m = 1500.0",
                &pass_with(40.0, 300.0, 0.0),
            )
        });
        let ev = discover(&a, &EventParams::default(), &[]);
        assert!(
            ev.iter()
                .any(|e| (e.fit.tc - truth.events[0].cpa_arrival_audio_s).abs() < 2.0),
            "{ev:#?}"
        );
    }

    #[test]
    fn steady_tone_makes_no_event() {
        use crate::audio::spectrogram::{SpecParams, spectrogram, tests::noisy};
        let a = noisy(4000, 120.0, Some((120.0, 0.3)), &[]);
        assert!(discover(&a, &EventParams::default(), &[]).is_empty());
        let s = spectrogram(&a, &SpecParams::default());
        let mask = crate::audio::lines::persistent_lines(&s);
        let b = ((120.0 - s.freqs[0]) / s.bin_hz()).round() as usize;
        assert!(mask[b], "120 Hz hum not masked");
        assert!(mask.iter().filter(|m| **m).count() < 20, "mask too wide");
    }

    #[test]
    fn slow_pass_not_masked() {
        // A slow, distant pass sweeps little per minute; the line mask must not take it for hum.
        let (a, truth) = sim("one_pass.toml", |s| {
            s.replace("speed_mps = 140.0", "speed_mps = 40.0")
                .replace("cpa_offset_m = 1500.0", "cpa_offset_m = 2500.0")
                .replace("half_span_s = 120.0", "half_span_s = 200.0")
        });
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 1, "{ev:#?}");
        assert!((ev[0].fit.tc - truth.events[0].cpa_arrival_audio_s).abs() < 2.0);
    }

    #[test]
    fn high_tone_pass_found_on_its_own_track() {
        // A single 1800 Hz engine tone at 80 m/s, 1.5 km away, is heard from 2347 Hz down to
        // 1460 Hz: all but the end of its fall lies above 1600 Hz. The recording is shorter
        // than the template stage accepts, so only the tone's own track can give the pass.
        let (a, truth) = sim("one_pass.toml", |s| {
            s.replace("f0 = 87.0", "f0 = 1800.0").replace(
                "speed_mps = 140.0\nalt_agl_m = 1800.0\ncpa_offset_m = 1500.0",
                &pass_with(80.0, 1000.0, 1100.0),
            )
        });
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 1, "{ev:#?}");
        close(&ev[0], &truth.events[0]);
        assert!((ev[0].fit.f0 - 1800.0).abs() < 5.0, "{:?}", ev[0].fit);
    }

    #[test]
    fn high_tone_pass_is_one_event_on_a_long_recording() {
        // The discovery bench's high-tone cell at 5 km, 30 dB: the tone crosses 1600 Hz 40 s
        // after its closest approach. One aircraft, one event at its closest approach: not a
        // second one on an image of the tone's last stretch at a fraction of its frequency,
        // and not a coarse template in place of the tone's own track.
        let (a, truth) = sim("faint_jet.toml", |s| {
            s.replace("snr_db = 16.0", "snr_db = 30.0")
                .replace("f0 = 650.0", "f0 = 1800.0")
                .replace(
                    "alt_agl_m = 1000.0\ncpa_offset_m = 1100.0",
                    "alt_agl_m = 2000.0\ncpa_offset_m = 4600.0",
                )
        });
        let ev = discover(&a, &EventParams::default(), &[]);
        let brief: Vec<String> = ev
            .iter()
            .map(|e| {
                format!(
                    "v {:.0} d {:.0} tc {:.1} f0 {:.1} ridges {}",
                    e.fit.v, e.fit.d, e.fit.tc, e.fit.f0, e.ridges
                )
            })
            .collect();
        assert_eq!(ev.len(), 1, "{brief:?}");
        close(&ev[0], &truth.events[0]);
        // The tone's own track: pursuit's event (a template's has no ridges), at the tone.
        assert!(ev[0].ridges > 0, "{brief:?}");
        assert!((ev[0].fit.f0 - 1800.0).abs() < 5.0, "{brief:?}");
    }

    #[test]
    fn recording_sampled_below_the_high_band_keeps_its_pass() {
        // Sampled at 3 kHz, a recording ends below the bottom of the band above 1600 Hz: that
        // band is empty, and the pass in the first band is found as before.
        let (a, truth) = sim("one_pass.toml", |s| {
            s.replace("sample_rate = 8000", "sample_rate = 3000")
        });
        assert_eq!(a.sample_rate, 3000);
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 1, "{ev:#?}");
        close(&ev[0], &truth.events[0]);
    }

    /// Mean whitened level along a truth pass's exact received-frequency curve (±40 s of CPA).
    fn level_along(s: &crate::audio::Spectrogram, truth: &gm_sim::Truth, i: usize) -> f32 {
        let m = crate::physics::Medium::new(truth.c, truth.wind_ecef).unwrap();
        // gm-sim links its own copy of this crate: rebuild the track with our types.
        let knots = truth.tracks[i]
            .track
            .knots
            .iter()
            .map(|k| crate::physics::Knot {
                t: k.t,
                pos: k.pos,
                vel: k.vel,
            })
            .collect();
        let tr = &crate::physics::Track::new("truth", knots).unwrap();
        let (f0, tc) = (truth.events[i].f0, truth.events[i].cpa_arrival_audio_s);
        let (mut sum, mut n) = (0.0f32, 0usize);
        for (fi, &t) in s.times.iter().enumerate() {
            if (t - tc).abs() > 40.0 {
                continue;
            }
            let Some(k) = crate::physics::doppler_factor(tr, truth.mic_ecef, t, m) else {
                continue;
            };
            let b = ((f0 * k - s.freqs[0]) / s.bin_hz()).round();
            if b < 1.0 || b as usize + 1 >= s.freqs.len() {
                continue;
            }
            let b = b as usize;
            sum += s.whitened[fi][b - 1..=b + 1]
                .iter()
                .copied()
                .fold(f32::MIN, f32::max);
            n += 1;
        }
        sum / n.max(1) as f32
    }

    #[test]
    fn multi_window_gains_on_far_pass() {
        use crate::audio::spectrogram::{SpecParams, multi_spectrogram};
        let (a, truth) = sim("suite/quiet.toml", |s| s);
        let d = crate::audio::decimate(&a, crate::audio::ANALYSIS_RATE);
        let one = multi_spectrogram(&d, &[1.0], &SpecParams::default());
        let multi = multi_spectrogram(
            &d,
            &EventParams::default().windows_s,
            &SpecParams::default(),
        );
        let (l1, lm) = (level_along(&one, &truth, 0), level_along(&multi, &truth, 0));
        assert!(
            lm > l1 + 0.5,
            "far pass: 1 s window {l1:.1} dB, multi {lm:.1} dB"
        );
    }

    #[test]
    fn missing_fundamental_found() {
        // A codec high-pass removed the 87 Hz fundamental: the fit has no fixed f0, so the
        // surviving harmonics alone give the pass (f0 a harmonic multiple of 87 Hz).
        let (a, truth) = sim("one_pass.toml", |s| {
            s.replace("harmonics = [1.0]", "harmonics = [0.0, 1.0, 1.0]")
        });
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 1, "{ev:#?}");
        let r = ev[0].fit.f0 / 87.0;
        assert!(
            (r - r.round()).abs() < 0.05 && r.round() >= 2.0,
            "f0 {} not a harmonic of 87",
            ev[0].fit.f0
        );
        assert!((ev[0].fit.tc - truth.events[0].cpa_arrival_audio_s).abs() < 1.0);
    }

    #[test]
    fn faint_far_pass_found() {
        // suite/quiet: 18 dB, the PC12 pass at 2.3 km slant was never discovered in v1–v2b.
        let (a, truth) = sim("suite/quiet.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 2, "{ev:#?}");
        for t in &truth.events {
            assert!(
                ev.iter()
                    .any(|e| (e.fit.tc - t.cpa_arrival_audio_s).abs() < 3.0),
                "{t:?} not found in {ev:#?}"
            );
        }
    }

    #[test]
    fn slowing_pass_found_on_its_tracks() {
        // decel: the B350 slows by 0.6 m/s² and descends while it is heard, so that no one
        // constant-velocity curve runs along the whole pass. Its tracks still give it.
        let (a, truth) = sim("decel.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        let t = truth
            .events
            .iter()
            .map(|e| e.cpa_arrival_audio_s)
            .fold(f64::MIN, f64::max);
        assert!(
            ev.iter()
                .any(|e| e.ridges > 0 && (e.fit.tc - t).abs() < 3.0),
            "no tracked event at {t}: {:?}",
            ev.iter().map(|e| (e.fit.tc, e.ridges)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn two_props_two_events() {
        let (a, truth) = sim("two_props.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 2, "{ev:#?}");
        close(&ev[0], &truth.events[0]);
        close(&ev[1], &truth.events[1]);
        // Harmonics [1, 0.6, 0.3] collapse into one event each, with the fundamental found.
        assert!(ev[0].ridges >= 2, "{:?}", ev[0]);
        assert!((ev[0].f0_fundamental - 87.0).abs() < 1.5, "{:?}", ev[0]);
        assert!((ev[1].f0_fundamental - 103.0).abs() < 1.5, "{:?}", ev[1]);
        assert_eq!(ev[0].id, 1);
    }

    #[test]
    fn overlap_two_events() {
        let (a, truth) = sim("overlap.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 2, "{ev:#?}");
        let mut by_tc = truth.events.clone();
        by_tc.sort_by(|x, y| x.cpa_arrival_audio_s.total_cmp(&y.cpa_arrival_audio_s));
        close(&ev[0], &by_tc[0]);
        close(&ev[1], &by_tc[1]);
    }

    #[test]
    fn noise_only_zero_events() {
        let mut s = 0x1234_5678u64;
        let samples = (0..8000 * 60)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
            })
            .collect();
        assert!(
            discover(
                &Audio {
                    sample_rate: 8000,
                    samples,
                    warnings: Vec::new()
                },
                &EventParams::default(),
                &[]
            )
            .is_empty()
        );
        assert!(
            discover(
                &Audio {
                    sample_rate: 8000,
                    samples: vec![0.0; 8000],
                    warnings: Vec::new()
                },
                &EventParams::default(),
                &[]
            )
            .is_empty()
        );
        assert!(
            discover(
                &Audio {
                    sample_rate: 8000,
                    samples: vec![],
                    warnings: Vec::new()
                },
                &EventParams::default(),
                &[]
            )
            .is_empty()
        );
    }

    #[test]
    fn window_restricts() {
        let (a, truth) = sim("two_props.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[(900.0, 1000.0)]);
        assert_eq!(ev.len(), 1, "{ev:#?}");
        close(&ev[0], &truth.events[1]);
    }

    #[test]
    fn intermittent_hum_one_event() {
        let (mut a, truth) = sim("one_pass.toml", |s| s);
        let rate = f64::from(a.sample_rate);
        for (i, v) in a.samples.iter_mut().enumerate() {
            let t = i as f64 / rate;
            if (20.0..56.0).contains(&t) {
                let hum: f64 = (1..=5)
                    .map(|k| 0.08 * (std::f64::consts::TAU * 60.0 * f64::from(k) * t).sin())
                    .sum();
                *v += hum as f32;
            }
        }
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 1, "{ev:#?}");
        close(&ev[0], &truth.events[0]);
    }

    #[test]
    fn lingering_tone_not_biased() {
        let (a, truth) = sim("overlap.toml", |s| {
            s.replace("duration_s = 150.0", "duration_s = 400.0")
        });
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 2, "{ev:#?}");
        let mut by_tc = truth.events.clone();
        by_tc.sort_by(|x, y| x.cpa_arrival_audio_s.total_cmp(&y.cpa_arrival_audio_s));
        for (e, t) in ev.iter().zip(&by_tc) {
            assert!(
                (e.fit.d - t.slant_m).abs() / t.slant_m < 0.04,
                "{e:?} {t:?}"
            );
            assert!(
                (e.fit.tc - t.cpa_arrival_audio_s).abs() < 0.2,
                "{e:?} {t:?}"
            );
        }
    }

    #[test]
    fn sigma_covers_actual_error() {
        let (a, truth) = sim("two_props.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        for (e, t) in ev.iter().zip(&truth.events) {
            assert!(
                (e.fit.v - t.speed_mps).abs() <= 3.0 * e.fit.sigma_v,
                "v {e:?} {t:?}"
            );
            assert!(
                (e.fit.tc - t.cpa_arrival_audio_s).abs() <= 3.0 * e.fit.sigma_tc,
                "tc {e:?} {t:?}"
            );
            assert!(
                (e.fit.d - t.slant_m).abs() <= 3.0 * e.fit.sigma_d,
                "d {e:?} {t:?}"
            );
        }
    }

    #[test]
    fn wavering_tone_is_not_an_event() {
        // An engine tone that drifts down while wavering (spool-down, machinery) is not a
        // Doppler pass. (A steady *linear* decline over a short window is genuinely ambiguous
        // with a slow, distant pass seen near CPA; ADS-B matching rejects those later.)
        let rate = 8000u32;
        let mut phase = 0.0f64;
        let samples = (0..rate as usize * 120)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let f = 180.0 - 0.1 * t + 4.0 * (0.25 * t).sin();
                phase += std::f64::consts::TAU * f / f64::from(rate);
                (0.3 * phase.sin()) as f32
            })
            .collect();
        let ev = discover(&Audio::new(rate, samples), &EventParams::default(), &[]);
        assert!(ev.is_empty(), "{ev:#?}");
    }

    #[test]
    fn event_points_follow_fit() {
        let (a, _) = sim("two_props.toml", |s| s);
        let ev = discover(&a, &EventParams::default(), &[]);
        assert_eq!(ev.len(), 2);
        for e in &ev {
            assert!(e.points.len() >= 50, "{}", e.points.len());
            assert!(e.points.windows(2).all(|w| w[0].0 <= w[1].0));
            let f = &e.fit;
            let rms = (e
                .points
                .iter()
                .map(|&(t, y)| (y - f.f0 * shape(t, f.v, f.d, f.tc, 343.0)).powi(2))
                .sum::<f64>()
                / e.points.len() as f64)
                .sqrt();
            assert!(rms <= 2.0 * f.rms_hz + 0.2, "rms {rms} vs fit {}", f.rms_hz);
        }
    }

    #[test]
    fn far_ridges_never_attach() {
        // A ridge that lies exactly on the seed's curve, but 10 minutes from its CPA (where the
        // curve is a flat asymptote), must not join that pass.
        use crate::audio::ridges::{Ridge, RidgePoint};
        let seed = CvFit {
            v: 120.0,
            d: 2000.0,
            tc: 100.0,
            f0: 90.0,
            rms_hz: 0.1,
            sigma_v: 1.0,
            sigma_d: 20.0,
            sigma_tc: 0.1,
            n: 400,
        };
        let on_curve = |t0: f64| Ridge {
            points: (0..80)
                .map(|i| {
                    let t = t0 + f64::from(i) * 0.25;
                    RidgePoint {
                        t,
                        f: 2.0 * 90.0 * shape(t, 120.0, 2000.0, 100.0, 343.0),
                        snr_db: 15.0,
                    }
                })
                .collect(),
        };
        assert!(
            fits_model(&on_curve(95.0), &seed, 343.0),
            "near CPA it attaches"
        );
        assert!(
            !fits_model(&on_curve(700.0), &seed, 343.0),
            "10 min later it must not"
        );
        assert!(
            !fits_model(&on_curve(-500.0), &seed, 343.0),
            "10 min earlier it must not"
        );
    }

    /// White noise with Doppler tones on one curve `geom` = [v, d, tc]: each voice (rest
    /// frequency, amplitude, from s, to s) sounds only in its stretch, with 1 s ramps.
    fn voices(secs: f64, geom: [f64; 3], voices: &[(f64, f64, f64, f64)]) -> Audio {
        use crate::audio::spectrogram::tests::noisy;
        let mut a = noisy(4000, secs, None, &[]);
        let rate = f64::from(a.sample_rate);
        for &(f0, amp, on, off) in voices {
            let mut phase = 0.0f64;
            for (i, y) in a.samples.iter_mut().enumerate() {
                let t = i as f64 / rate;
                phase +=
                    std::f64::consts::TAU * f0 * shape(t, geom[0], geom[1], geom[2], 343.0) / rate;
                let gate = ((t - on).min(off - t)).clamp(0.0, 1.0);
                // `noisy` scales its samples by 0.05.
                *y += (0.05 * amp * gate * phase.sin()) as f32;
            }
        }
        a
    }

    #[test]
    fn faded_pass_on_unrelated_partials_is_one_event() {
        // As on real recordings: the approach is heard on one partial, the closest approach
        // on another, the departure on a third, none a small-integer multiple of another.
        // No piece alone says where the aircraft was; together they are one pass.
        let geom = [86.0, 907.0, 60.0];
        let a = voices(
            120.0,
            geom,
            &[
                (108.0, 0.35, 22.0, 52.0),
                (85.0, 0.35, 54.0, 66.0),
                (147.9, 0.35, 64.0, 104.0),
            ],
        );
        let ev = discover(&a, &EventParams::default(), &[]);
        let brief: Vec<String> = ev
            .iter()
            .map(|e| {
                format!(
                    "v {:.0} d {:.0} tc {:.1} f0 {:.1} ridges {}",
                    e.fit.v, e.fit.d, e.fit.tc, e.fit.f0, e.ridges
                )
            })
            .collect();
        assert_eq!(ev.len(), 1, "{brief:?}");
        let f = &ev[0].fit;
        assert!((f.v / geom[0] - 1.0).abs() < 0.06, "{brief:?}");
        assert!((f.d / geom[1] - 1.0).abs() < 0.10, "{brief:?}");
        assert!((f.tc - geom[2]).abs() < 1.0, "{brief:?}");
        assert!(ev[0].ridges >= 3, "{brief:?}");
    }

    /// The log tells of the search the same windows run: a pass the windows cut off is no
    /// event in the log either.
    #[test]
    fn the_log_for_windows_is_the_search_in_those_windows() {
        let geom = [86.0, 907.0, 60.0];
        let a = voices(
            120.0,
            geom,
            &[
                (108.0, 0.35, 22.0, 52.0),
                (85.0, 0.35, 54.0, 66.0),
                (147.9, 0.35, 64.0, 104.0),
            ],
        );
        let p = EventParams::default();
        let events_in = |log: &[String]| log.iter().filter(|l| l.ends_with(": event")).count();
        let whole = explain(&a, &p, &[], (0.0, 120.0));
        assert_eq!(events_in(&whole), 1, "{whole:#?}");
        // The first 27 s hold 5 s of one partial: nothing to start from.
        let w = (0.0, 27.0);
        assert!(discover(&a, &p, &[w]).is_empty());
        let log = explain(&a, &p, &[w], w);
        assert_eq!(events_in(&log), 0, "{log:#?}");
    }
}
