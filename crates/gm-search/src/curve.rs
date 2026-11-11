//! Exact-curve check: the emission-time Doppler curve the matched aircraft would produce at the
//! cell, against the event's measured ridge points, with f₀ solved in closed form.

use gm_doppler::audio::Event;
use gm_doppler::physics::Track;

use crate::atmos::Atmosphere;

use crate::grid::{Candidate, Cell, Inputs};
use crate::matching::Match;
use crate::params::CURVE_HALF_WINDOW_S;
use crate::score::{CellScore, score_cell};

/// Spectrogram frames overlap: one 1 s window per 0.25 s hop → points are ~4× correlated.
pub const OVERLAP: f64 = 4.0;
/// CPA-time nuisance search: ±SHIFT_STEPS × SHIFT_STEP_S.
pub const SHIFT_STEPS: i32 = 6;
pub const SHIFT_STEP_S: f64 = 0.1;
/// Model error of the still-air curve, as a fraction of frequency (wind, curvature).
pub const MODEL_F_FRAC: f64 = 0.005;
/// Most effective independent points one event's curve can count for.
pub const CURVE_NEFF_MAX: f64 = 50.0;

/// The capped curve log-likelihood of ridge `points` (audio time, Hz) received at `times` (track
/// clock) from `track` at `cell`, with f₀ solved in closed form: `(ll, f0, χ²/n)`. The ll is
/// capped at 0 (flat wherever χ²/n ≤ 1), so alignment searches minimise χ²/n. `None` when fewer
/// than half the points (or fewer than 8) have a defined received frequency.
pub(crate) fn curve_fit(
    points: &[(f64, f64)],
    times: &[f64],
    track: &Track,
    cell: &Cell,
    rms_hz: f64,
    atmos: &Atmosphere,
) -> Option<(f64, f64, f64)> {
    let factors = atmos.factors(track, cell.ecef, times);
    let pairs: Vec<(f64, f64)> = points
        .iter()
        .zip(&factors)
        .filter_map(|(&(_, f), fac)| fac.map(|k| (f, k)))
        .collect();
    if pairs.len() * 2 < points.len() || pairs.len() < 8 {
        return None;
    }
    let (num, den) = pairs
        .iter()
        .fold((0.0, 0.0), |(n, d), &(f, k)| (n + f * k, d + k * k));
    if den <= 0.0 {
        return None;
    }
    let f0 = num / den;
    let chi2: f64 = pairs
        .iter()
        .map(|&(f, k)| {
            let sigma2 = rms_hz.max(0.2).powi(2) + (MODEL_F_FRAC * f).powi(2);
            (f - f0 * k).powi(2) / sigma2
        })
        .sum();
    // Excess over the event's own fit quality; never a bonus (a cell can't earn more than a
    // perfect explanation), so it composes with the match and outlier terms. Systematic model
    // error (wind aloft, temperature, ADS-B sampling) makes residuals correlated far beyond frame
    // overlap: cap the effective count so thousands of points can't drown the match evidence.
    let n = pairs.len() as f64;
    let n_eff = (n / OVERLAP).min(CURVE_NEFF_MAX);
    Some((-0.5 * n_eff * (chi2 / n - 1.0).max(0.0), f0, chi2 / n))
}

/// The v1 refinement's curve term for a gate match at clock `delta_s` (best over a ±0.6 s CPA-time
/// nuisance). Since v2a it uses only ridge points within CURVE_HALF_WINDOW_S of the event's CV
/// tc; an event with no points there gets `None` (and the refinement charges OUTLIER_LL).
pub fn curve_ll(
    event: &Event,
    m: &Match,
    cell: &Cell,
    aircraft: &Candidate,
    delta_s: f64,
    atmos: &Atmosphere,
) -> Option<f64> {
    let pts: Vec<(f64, f64)> = event
        .points
        .iter()
        .filter(|p| (p.0 - event.fit.tc).abs() <= CURVE_HALF_WINDOW_S)
        .copied()
        .collect();
    let (t_lo, t_hi) = (pts.first()?.0 + delta_s, pts.last()?.0 + delta_s);
    let mut best: Option<f64> = None;
    for tr in &aircraft.tracks {
        let Some(w) = tr.window(t_lo - 120.0, t_hi + 5.0) else {
            continue;
        };
        // Nuisance: a small per-event CPA-time correction (measurement error, wind) — colsto's
        // "corrected for measurement error on CPA time & f0"; f0 is solved in closed form.
        for k in -SHIFT_STEPS..=SHIFT_STEPS {
            let shift = f64::from(k) * SHIFT_STEP_S;
            let times: Vec<f64> = pts.iter().map(|p| p.0 + delta_s + shift).collect();
            if let Some((ll, _, _)) = curve_fit(&pts, &times, &w, cell, event.fit.rms_hz, atmos) {
                best = Some(best.map_or(ll, |b: f64| b.max(ll)));
            }
        }
    }
    let _ = m;
    best
}

/// `score_cell` plus the curve terms for each matched pair.
pub fn score_cell_refined(inp: &Inputs, cell: &Cell) -> CellScore {
    use crate::params::OUTLIER_LL;
    let mut s = score_cell(inp, cell);
    let mut extra = 0.0;
    for (ei, m) in &s.matched {
        // Curve-first matches already carry their curve term.
        if m.via == crate::matching::MatchVia::Curve {
            continue;
        }
        let (Some(e), Some(a)) = (inp.events.get(*ei), inp.aircraft.get(m.aircraft)) else {
            continue;
        };
        // The curve can subtract at most as much as an unexplained event costs: in strong wind
        // the still-air curve is off everywhere, and the match evidence must survive it.
        let curve = curve_ll(e, m, cell, a, s.delta_s, inp.atmos)
            .unwrap_or(OUTLIER_LL)
            .max(OUTLIER_LL);
        extra += curve;
    }
    s.ll += extra;
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::cell_at;
    use crate::matching::matches;
    use crate::testutil::fixture;
    use gm_core::LocalFrame;

    #[test]
    fn curve_prefers_truth() {
        let f = fixture("two_props.toml", |s| s);
        let o = f.truth.observer;
        let frame = LocalFrame::new(o);
        let h = o.h_m;
        let at = cell_at(&frame, 0.0, 0.0, &move |_, _| h);
        let off = cell_at(&frame, 300.0, 0.0, &move |_, _| h);
        for e in &f.events {
            let m = matches(
                e,
                &at,
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still(),
            )
            .into_iter()
            .max_by(|a, b| a.ll.total_cmp(&b.ll))
            .unwrap();
            let a = &f.aircraft[m.aircraft];
            let good =
                curve_ll(e, &m, &at, a, m.delta_s, &crate::atmos::Atmosphere::still()).unwrap();
            let bad = curve_ll(
                e,
                &m,
                &off,
                a,
                m.delta_s,
                &crate::atmos::Atmosphere::still(),
            )
            .unwrap();
            assert!(good > bad + 1.0, "{good} vs {bad}");
            assert!(good > -20.0, "{good}");
        }
    }

    #[test]
    fn curve_none_without_overlap() {
        let f = fixture("two_props.toml", |s| s);
        let o = f.truth.observer;
        let h = o.h_m;
        let at = cell_at(&LocalFrame::new(o), 0.0, 0.0, &move |_, _| h);
        let e = &f.events[0];
        let m = matches(
            e,
            &at,
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        )
        .into_iter()
        .next()
        .unwrap();
        assert!(
            curve_ll(
                e,
                &m,
                &at,
                &f.aircraft[m.aircraft],
                5000.0,
                &crate::atmos::Atmosphere::still()
            )
            .is_none()
        );
    }
}
