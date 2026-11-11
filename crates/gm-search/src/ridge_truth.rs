//! Ridge truth: detection measured against annotated frequency–time curves (Livermore: one
//! harmonic per pass, UTC-stamped). The WAV has no published start time, so each curve is tied
//! to the recording by one clock offset found by sliding it along the recording's own guide
//! image — no discovery output and no location are used, so the score can't be biased toward
//! what discovery found.

use std::path::Path;

use gm_doppler::audio::Event;
use gm_doppler::audio::guide::GuideImage;

/// A truth curve on the recording's clock: (t s, f Hz), sorted by t.
#[derive(Debug, Clone)]
pub struct TruthRidge {
    pub id: String,
    pub points: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RidgeDetection {
    pub expected: usize,
    pub found: usize,
    pub unmatched_events: usize,
    pub per_ridge: Vec<(String, bool)>,
}

/// Harmonic ratios at which an event may sit on a truth ridge.
const RATIOS: [f64; 5] = [1.0, 2.0, 3.0, 0.5, 1.0 / 3.0];

/// `time_utc,frequency_hz` → (seconds from the first point, Hz).
pub fn load_truth_csv(path: &Path) -> Result<Vec<(f64, f64)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut t0: Option<gm_core::Epoch> = None;
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let mut it = line.split(',');
        let (Some(t), Some(f)) = (it.next(), it.next()) else {
            continue;
        };
        let (Some(e), Ok(f)) = (
            gm_core::Epoch::from_rfc3339(t.trim()),
            f.trim().parse::<f64>(),
        ) else {
            continue;
        };
        let base = *t0.get_or_insert(e);
        out.push((e.seconds_since(base), f));
    }
    if out.len() < 8 {
        return Err(format!("{}: fewer than 8 points", path.display()));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

/// The clock offset (recording time = curve time + offset) where the curve's mean standardised
/// level on `img` peaks, and that peak's robust z over all offsets that keep at least half the
/// curve inside the recording. `None` when the curve can't fit or the level has no spread.
pub fn align(img: &GuideImage, pts: &[(f64, f64)]) -> Option<(f64, f64)> {
    let (first, last) = (pts.first()?.0, pts.last()?.0);
    let (t0, hop) = (*img.times.first()?, img.hop_s);
    let t_end = *img.times.last()?;
    if hop <= 0.0 {
        return None;
    }
    let nb = img.n_bins() as i64;
    let bins: Vec<i64> = pts
        .iter()
        .map(|&(_, f)| img.bin_of(f).round() as i64)
        .collect();
    // Slide across the whole recording (offsets keeping at least half the curve inside), so the
    // z has offsets far from the true one to compare against.
    let (lo, hi) = (t0 - last, t_end - first);
    let n = ((hi - lo) / hop).floor() as usize;
    let mut scores: Vec<(f64, f64)> = Vec::with_capacity(n + 1);
    for k in 0..=n {
        let off = lo + k as f64 * hop;
        let (mut sum, mut used) = (0.0f64, 0usize);
        for (&(t, _), &q) in pts.iter().zip(&bins) {
            let j = ((t + off - t0) / hop).round() as i64;
            if j < 0 || j as usize >= img.level.len() || q < 0 || q >= nb {
                continue;
            }
            sum += f64::from(img.level[j as usize][q as usize]);
            used += 1;
        }
        if used * 2 >= pts.len() {
            scores.push((off, sum / used as f64));
        }
    }
    if scores.len() < 10 {
        return None;
    }
    let mut v: Vec<f64> = scores.iter().map(|s| s.1).collect();
    v.sort_by(f64::total_cmp);
    let med = v[v.len() / 2];
    let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    dev.sort_by(f64::total_cmp);
    let mad = 1.4826 * dev[dev.len() / 2];
    let &(off, best) = scores.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    (mad > 0.0).then(|| (off, (best - med) / mad))
}

/// The ridge's frequency at `t` (linear between its points), or `None` outside its span.
fn freq_at(r: &TruthRidge, t: f64) -> Option<f64> {
    let p = &r.points;
    if p.is_empty() || t < p[0].0 || t > p[p.len() - 1].0 {
        return None;
    }
    let i = p.partition_point(|q| q.0 < t);
    if i == 0 {
        return Some(p[0].1);
    }
    let (a, b) = (p[i - 1], p[i.min(p.len() - 1)]);
    if b.0 <= a.0 {
        return Some(a.1);
    }
    Some(a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0))
}

/// Does `e` follow `r`: at least half of its points inside the ridge's span and within 1 % of
/// r × the ridge's frequency there, for one ratio r?
fn follows(e: &Event, r: &TruthRidge) -> bool {
    follows_ratio(e, r).is_some()
}

/// The fraction of `e`'s points on `r` at harmonic `ratio` (within 1 %).
pub fn on_fraction(e: &Event, r: &TruthRidge, ratio: f64) -> f64 {
    if e.points.is_empty() {
        return 0.0;
    }
    let on = e
        .points
        .iter()
        .filter(|&&(t, f)| freq_at(r, t).is_some_and(|fr| (f / (ratio * fr) - 1.0).abs() <= 0.01))
        .count();
    on as f64 / e.points.len() as f64
}

/// The harmonic ratio (event / truth) at which `e` follows `r` — at least half its points
/// within 1 % — first in RATIOS order; `None` if it follows at none.
pub fn follows_ratio(e: &Event, r: &TruthRidge) -> Option<f64> {
    RATIOS
        .iter()
        .copied()
        .find(|&ratio| on_fraction(e, r, ratio) >= 0.5)
}

/// For a diagnostic: the best (fraction of `e`'s points on `r`, ratio) over RATIOS.
pub fn best_fraction(e: &Event, r: &TruthRidge) -> (f64, f64) {
    RATIOS
        .iter()
        .map(|&ratio| (on_fraction(e, r, ratio), ratio))
        .fold((0.0, 1.0), |b, x| if x.0 > b.0 { x } else { b })
}

/// For a diagnostic: over the points of `e` inside `r`'s span, the best (fraction within 1 % of
/// `r`, ratio) over RATIOS — `None` unless those points cover at least half of `r`'s span. It
/// separates an event that follows `r` but runs past a shorter truth from one that is off it.
pub fn best_overlap_fraction(e: &Event, r: &TruthRidge) -> Option<(f64, f64)> {
    let (first, last) = (r.points.first()?.0, r.points.last()?.0);
    let inside: Vec<(f64, f64, f64)> = e
        .points
        .iter()
        .filter_map(|&(t, f)| freq_at(r, t).map(|fr| (t, f, fr)))
        .collect();
    let (a, b) = (inside.first()?.0, inside.last()?.0);
    if last <= first || (b - a) < 0.5 * (last - first) {
        return None;
    }
    RATIOS
        .iter()
        .map(|&ratio| {
            let on = inside
                .iter()
                .filter(|&&(_, f, fr)| (f / (ratio * fr) - 1.0).abs() <= 0.01)
                .count();
            (on as f64 / inside.len() as f64, ratio)
        })
        .reduce(|b, x| if x.0 > b.0 { x } else { b })
}

/// Truth ridges found by `events` (each event matches at most one ridge).
pub fn ridge_detection(events: &[Event], truth: &[TruthRidge]) -> RidgeDetection {
    let mut used = vec![false; events.len()];
    let mut per_ridge = Vec::new();
    for r in truth {
        let hit = (0..events.len()).find(|&i| !used[i] && follows(&events[i], r));
        if let Some(i) = hit {
            used[i] = true;
        }
        per_ridge.push((r.id.clone(), hit.is_some()));
    }
    RidgeDetection {
        expected: truth.len(),
        found: per_ridge.iter().filter(|x| x.1).count(),
        unmatched_events: used.iter().filter(|u| !**u).count(),
        per_ridge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ridge(id: &str, t0: f64, t1: f64, f: impl Fn(f64) -> f64) -> TruthRidge {
        let n = ((t1 - t0) / 0.25) as usize;
        TruthRidge {
            id: id.into(),
            points: (0..=n)
                .map(|k| {
                    let t = t0 + k as f64 * 0.25;
                    (t, f(t))
                })
                .collect(),
        }
    }

    fn event_on(t0: f64, t1: f64, f: impl Fn(f64) -> f64) -> Event {
        let mut e = crate::testutil::fixture("one_pass.toml", |s| s).events[0].clone();
        e.points = ridge("e", t0, t1, f).points;
        e
    }

    #[test]
    fn ridge_truth_matches_at_harmonic() {
        let truth = ridge("p1", 100.0, 160.0, |t| 120.0 - 0.2 * (t - 100.0));
        let e = event_on(100.0, 160.0, |t| 2.0 * (120.0 - 0.2 * (t - 100.0)));
        assert_eq!(ridge_detection(&[e], &[truth]).found, 1);
    }

    #[test]
    fn ridge_truth_needs_time_overlap() {
        let truth = ridge("p1", 100.0, 160.0, |_| 120.0);
        let e = event_on(400.0, 460.0, |_| 120.0);
        let d = ridge_detection(&[e], &[truth]);
        assert_eq!((d.found, d.unmatched_events), (0, 1));
    }

    #[test]
    fn align_recovers_the_clock_offset() {
        // An image lit along the curve 30 s into the recording: the curve (from t = 0) aligns at
        // offset ≈ 30 s with a high z.
        let n = 400;
        let mut img = GuideImage {
            times: (0..n).map(|j| 0.5 + j as f64 * 0.25).collect(),
            hop_s: 0.25,
            f_lo: gm_doppler::audio::guide::GUIDE_F_LO,
            bpo: gm_doppler::audio::guide::GUIDE_BPO,
            score: Vec::new(),
            level: vec![vec![0.0; 900]; n],
        };
        let f = |t: f64| 300.0 - 1.0 * t;
        let pts: Vec<(f64, f64)> = (0..160)
            .map(|k| (k as f64 * 0.25, f(k as f64 * 0.25)))
            .collect();
        for &(t, fr) in &pts {
            let j = ((t + 30.0 - 0.5) / 0.25).round() as usize;
            let q = img.bin_of(fr).round() as usize;
            img.level[j][q] = 5.0;
        }
        // Standardised noise everywhere, as a real guide image has.
        let mut x: u64 = 3;
        for row in img.level.iter_mut() {
            for c in row.iter_mut() {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *c += ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0;
            }
        }
        let (off, z) = align(&img, &pts).unwrap();
        assert!((off - 30.0).abs() <= 0.25, "{off}");
        assert!(z > 10.0, "{z}");
    }
}
