//! Curve-first matching: align an event's measured ridge points with an aircraft's exact
//! emission-time Doppler curve and take the clock offset δ from that alignment, not from the
//! constant-velocity fit's CPA time (which accelerating or descending passes bias by seconds).

use gm_doppler::audio::Event;
use gm_doppler::physics::Track;

use crate::atmos::Atmosphere;

use crate::curve::curve_fit;
use crate::grid::{Candidate, Cell};
use crate::matching::{Match, MatchVia};
use crate::params::{
    CURVE_DECIMATE, CURVE_DELTA_STEP_S, CURVE_FINE_STEP_S, CURVE_GEOM_RATIO, CURVE_HALF_WINDOW_S,
    CURVE_MAX_SLANT_M, CURVE_SOFT_CAP, CURVE_TC_BIAS_S, CURVE_TC_SIGMA_S, MODEL_D_FRAC,
    MODEL_V_FRAC, OUTLIER_LL,
};

/// δ values to scan: CURVE_DELTA_STEP_S steps within CURVE_TC_BIAS_S of the CPA-implied δ₀,
/// clipped to the prior window widened by 1 s.
pub(crate) fn delta_grid(delta0: f64, range: (f64, f64)) -> Vec<f64> {
    let lo = (delta0 - CURVE_TC_BIAS_S).max(range.0 - 1.0);
    let hi = (delta0 + CURVE_TC_BIAS_S).min(range.1 + 1.0);
    if lo.is_nan() || hi.is_nan() || lo > hi {
        return Vec::new();
    }
    let n = ((hi - lo) / CURVE_DELTA_STEP_S + 1e-9).floor() as usize;
    let mut g: Vec<f64> = (0..=n)
        .map(|i| lo + i as f64 * CURVE_DELTA_STEP_S)
        .collect();
    if hi - g[g.len() - 1] > 1e-9 {
        g.push(hi);
    }
    g
}

/// Curve-first match proposals for `event` heard at `cell`: for each audible aircraft whose
/// CPA lies within CURVE_MAX_SLANT_M, the δ (near δ₀ and inside the prior window) whose exact
/// curve best fits the ridge points, kept when it beats calling the event an outlier.
pub fn curve_matches(
    event: &Event,
    cell: &Cell,
    aircraft: &[Candidate],
    delta_range: (f64, f64),
    atmos: &Atmosphere,
) -> Vec<Match> {
    // All points near CPA score the match; every CURVE_DECIMATE-th of them aligns δ (cheap).
    let win: Vec<(f64, f64)> = event
        .points
        .iter()
        .filter(|p| (p.0 - event.fit.tc).abs() <= CURVE_HALF_WINDOW_S)
        .copied()
        .collect();
    let pts: Vec<(f64, f64)> = win.iter().step_by(CURVE_DECIMATE).copied().collect();
    let (Some(first), Some(last)) = (pts.first(), pts.last()) else {
        return Vec::new();
    };
    let f = &event.fit;
    let tc = f.tc;
    let sd = (f.sigma_d.powi(2) + (MODEL_D_FRAC * f.d).powi(2)).sqrt();
    let sv = (f.sigma_v.powi(2) + (MODEL_V_FRAC * f.v).powi(2)).sqrt();
    let (lo, hi) = (
        delta_range.0 - 1.0 - CURVE_TC_BIAS_S,
        delta_range.1 + 1.0 + CURVE_TC_BIAS_S,
    );
    // (χ²/n, ll): align on χ²/n — the capped ll is flat near the optimum.
    let fit_at = |tr: &Track, d: f64| -> Option<(f64, f64)> {
        let times: Vec<f64> = pts.iter().map(|p| p.0 + d).collect();
        curve_fit(&pts, &times, tr, cell, f.rms_hz, atmos).map(|(ll, _, chi)| (chi, ll))
    };
    let mut out: Vec<Match> = Vec::new();
    for (ai, a) in aircraft.iter().enumerate() {
        if !crate::matching::may_match(event, a) {
            continue;
        }
        let mut best: Option<Match> = None;
        for tr in &a.tracks {
            let Some(w) = tr.window(first.0.min(tc) + lo - 120.0, last.0.max(tc) + hi + 60.0)
            else {
                continue;
            };
            let Some(c) = atmos.cpa(&w, cell.ecef) else {
                continue;
            };
            let within =
                |a: f64, b: f64| a > 0.0 && b > 0.0 && (a / b).ln().abs() <= CURVE_GEOM_RATIO.ln();
            if c.slant_m > CURVE_MAX_SLANT_M.min(a.class.tone_range_m())
                || !within(c.slant_m, f.d)
                || !within(c.ground_speed_mps, f.v)
            {
                continue;
            }
            let delta0 = c.t_arrive - tc;
            let mut top: Option<(f64, f64, f64)> = None; // (χ²/n, ll, δ)
            let consider = |top: &mut Option<(f64, f64, f64)>, d: f64| {
                if let Some((chi, ll)) = fit_at(&w, d)
                    && top.is_none_or(|(b, _, _)| chi < b)
                {
                    *top = Some((chi, ll, d));
                }
            };
            for d in delta_grid(delta0, delta_range) {
                consider(&mut top, d);
            }
            let Some((_, _, coarse)) = top else { continue };
            let steps = (CURVE_DELTA_STEP_S / CURVE_FINE_STEP_S).round() as i32;
            for k in -steps..=steps {
                let d = coarse + f64::from(k) * CURVE_FINE_STEP_S;
                if d >= delta_range.0 - 1.0 && d <= delta_range.1 + 1.0 {
                    consider(&mut top, d);
                }
            }
            let Some((_, _, d)) = top else { continue };
            // Score on all windowed points: the decimated fit already dropped the frame overlap
            // that curve_fit's n_eff divides out, so its ll would be ~4× too lenient.
            let times: Vec<f64> = win.iter().map(|p| p.0 + d).collect();
            let Some((curve, _, _)) = curve_fit(&win, &times, &w, cell, f.rms_hz, atmos) else {
                continue;
            };
            // Geometry as the gate scores it (slant, speed), soft rather than gated, plus a weak
            // tie to the CV fit's CPA time: the capped curve term is flat near its optimum.
            let (dd, dv) = ((c.slant_m - f.d) / sd, (c.ground_speed_mps - f.v) / sv);
            let dt = (d - delta0) / CURVE_TC_SIGMA_S;
            let soft = |x: f64| (0.5 * x * x).min(CURVE_SOFT_CAP);
            let ll = curve - soft(dd) - soft(dv) - soft(dt);
            if ll <= OUTLIER_LL || best.as_ref().is_some_and(|b| b.ll >= ll) {
                continue;
            }
            best = Some(Match {
                aircraft: ai,
                icao: a.icao.clone(),
                delta_s: d,
                ll,
                slant_m: c.slant_m,
                speed_mps: c.ground_speed_mps,
                via: MatchVia::Curve,
            });
        }
        out.extend(best);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{AircraftClass, cell_at};
    use crate::matching::matches;
    use crate::params::OUTLIER_LL;
    use crate::testutil::{Fixture, fixture};
    use gm_core::LocalFrame;

    fn cell(f: &Fixture, e: f64, n: f64) -> Cell {
        let o = f.truth.observer;
        let h = o.h_m;
        cell_at(&LocalFrame::new(o), e, n, &move |_, _| h)
    }

    /// The discovered event closest to the B350's true CPA arrival.
    fn b350_event(f: &Fixture) -> Event {
        let t = f.truth.events[1].cpa_arrival_audio_s;
        f.events
            .iter()
            .min_by(|a, b| (a.fit.tc - t).abs().total_cmp(&(b.fit.tc - t).abs()))
            .unwrap()
            .clone()
    }

    #[test]
    fn gate_misses_decelerating_pass() {
        let f = fixture("decel.toml", |s| s);
        let e = b350_event(&f);
        // The CV fit's tc is biased, so the gate's δ lands outside the 1 s shared-clock window
        // around the true clock (0): v1 can never count this pass alongside the PC12.
        let m = matches(
            &e,
            &cell(&f, 0.0, 0.0),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        assert!(
            m.iter()
                .all(|m| m.icao != "aaaa02" || m.delta_s.abs() > 1.0 || m.ll <= OUTLIER_LL),
            "gate gives a usable match for the decelerating pass (tc {} vs truth {}): {m:?}",
            e.fit.tc,
            f.truth.events[1].cpa_arrival_audio_s
        );
    }

    #[test]
    fn curve_finds_decelerating_pass() {
        let f = fixture("decel.toml", |s| s);
        let e = b350_event(&f);
        let m = curve_matches(
            &e,
            &cell(&f, 0.0, 0.0),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        let b = m
            .iter()
            .find(|m| m.icao == "aaaa02")
            .unwrap_or_else(|| panic!("{m:?}"));
        assert!(b.delta_s.abs() < 0.5, "{b:?}");
        assert!(b.ll > -3.0, "{b:?}");
        assert_eq!(b.via, crate::matching::MatchVia::Curve);
    }

    #[test]
    fn curve_match_respects_class_and_prefilter() {
        let mut f = fixture("two_props.toml", |s| s);
        let e = f.events[0].clone();
        let far = curve_matches(
            &e,
            &cell(&f, 9000.0, 0.0),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        assert!(far.is_empty(), "{far:?}");
        assert!(
            !curve_matches(
                &e,
                &cell(&f, 0.0, 0.0),
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still()
            )
            .is_empty()
        );
        // As jets, only passes within their tone range (3 km) may match.
        for a in &mut f.aircraft {
            a.class = AircraftClass::Jet;
        }
        let none = curve_matches(
            &e,
            &cell(&f, 0.0, 0.0),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        assert!(
            none.is_empty(),
            "a discovered event never matches a jet: {none:?}"
        );
        let mut g = e.clone();
        g.guided = true;
        g.guide_icao = Some(f.truth.events[0].icao24.clone());
        let m = curve_matches(
            &g,
            &cell(&f, 0.0, 0.0),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        assert!(
            m.iter()
                .all(|m| m.slant_m <= crate::params::JET_MAX_SLANT_M),
            "{m:?}"
        );
        assert!(!m.is_empty(), "a guided 1.1 km jet pass must be matchable");
    }

    #[test]
    fn curve_tolerates_slow_f0_drift() {
        // Propeller RPM changes on approach: f₀ drifts by ±2 % over a ±100 s ridge. A constant-f₀
        // curve can only hold near CPA, so the fit must not use the whole ridge. The ridge is
        // built from the truth physics (independent of discovery), the CV fit kept from discovery.
        let f = fixture("decel.toml", |s| s);
        let mut e = b350_event(&f);
        let tr = &f.truth.tracks[1].track;
        let medium = gm_doppler::physics::Medium::new(f.truth.c, f.truth.wind_ecef).unwrap();
        let tcpa = f.truth.events[1].cpa_arrival_audio_s;
        let times: Vec<f64> = (0..800)
            .map(|i| tcpa - 100.0 + 0.25 * f64::from(i))
            .collect();
        let freqs =
            gm_doppler::physics::received_frequencies(tr, f.truth.mic_ecef, &times, 103.0, medium);
        e.points = times
            .iter()
            .zip(&freqs)
            .filter_map(|(&t, k)| k.map(|k| (t, k * (1.0 + 2.0e-4 * (t - tcpa)))))
            .collect();
        assert!(e.points.len() > 700, "{}", e.points.len());
        let m = curve_matches(
            &e,
            &cell(&f, 0.0, 0.0),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        let b = m
            .iter()
            .find(|m| m.icao == "aaaa02")
            .unwrap_or_else(|| panic!("{m:?}"));
        assert!(b.delta_s.abs() < 0.5, "{b:?}");
    }

    #[test]
    fn curve_match_survives_biased_cv_geometry() {
        // A long, decelerating pass can bias the CV fit's slant and speed by many σ; the exact
        // curve is the evidence, and the CV geometry may only break ties, never veto.
        let f = fixture("decel.toml", |s| s);
        let mut e = b350_event(&f);
        // The real Oyster Bay B350's CV fit was 2.5× short in slant and 1.4× slow, relative to
        // the truth (not to whatever this front end happens to fit).
        e.fit.d = f.truth.events[1].slant_m / 2.5;
        e.fit.v = f.truth.events[1].speed_mps / 1.4;
        let m = curve_matches(
            &e,
            &cell(&f, 0.0, 0.0),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        let b = m
            .iter()
            .find(|m| m.icao == "aaaa02")
            .unwrap_or_else(|| panic!("{m:?}"));
        assert!(b.delta_s.abs() < 0.5, "{b:?}");
        assert!(b.ll > OUTLIER_LL, "{b:?}");
    }

    #[test]
    fn curve_match_rejects_inconsistent_geometry() {
        // A short, close ridge (a road vehicle: CV fit ~300 m, ~40 m/s) can be fitted by a
        // distant aircraft's curve with f₀ free; the CV geometry must still rule that out.
        let f = fixture("two_props.toml", |s| s);
        let mut e = f.events[0].clone();
        let at = cell(&f, 0.0, 0.0);
        assert!(
            curve_matches(
                &e,
                &at,
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still()
            )
            .iter()
            .any(|m| m.icao == "aaaa01")
        );
        e.fit.d /= 24.0;
        assert!(
            curve_matches(
                &e,
                &at,
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still()
            )
            .iter()
            .all(|m| m.icao != "aaaa01"),
            "slant 24× the CV fit's d must not match"
        );
        let mut e = f.events[0].clone();
        e.fit.v /= 4.0;
        assert!(
            curve_matches(
                &e,
                &at,
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still()
            )
            .iter()
            .all(|m| m.icao != "aaaa01"),
            "speed 4× the CV fit's v must not match"
        );
    }

    #[test]
    fn curve_match_scores_full_window() {
        // δ may be aligned on decimated points, but the reported curve term must be the same
        // misfit the v1 refinement charges on all windowed points (decimating then dividing by
        // the frame overlap again would make curve-first matches ~4× too lenient).
        let f = fixture("two_props.toml", |s| s);
        let e = &f.events[1];
        let pts: Vec<(f64, f64)> = e
            .points
            .iter()
            .filter(|p| (p.0 - e.fit.tc).abs() <= crate::params::CURVE_HALF_WINDOW_S)
            .copied()
            .collect();
        let mut misfit_checked = false;
        for north in [150.0, 160.0, 170.0, 180.0, 185.0, 190.0, 195.0, 200.0] {
            let c = cell(&f, 0.0, north);
            for m in curve_matches(
                e,
                &c,
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still(),
            ) {
                let times: Vec<f64> = pts.iter().map(|p| p.0 + m.delta_s).collect();
                let tr = &f.aircraft[m.aircraft].tracks[0];
                let full = crate::curve::curve_fit(
                    &pts,
                    &times,
                    tr,
                    &c,
                    e.fit.rms_hz,
                    &crate::atmos::Atmosphere::still(),
                )
                .unwrap()
                .0;
                assert!(
                    m.ll <= full + 1e-9,
                    "{north} m: curve-first ll {} vs full-window curve {full}",
                    m.ll
                );
                misfit_checked |= full < -1.0;
            }
        }
        assert!(
            misfit_checked,
            "no accepted match with a real misfit was checked"
        );
    }

    #[test]
    fn short_event_no_curve_match() {
        let f = fixture("two_props.toml", |s| s);
        let mut e = f.events[0].clone();
        e.points.truncate(20);
        assert!(
            curve_matches(
                &e,
                &cell(&f, 0.0, 0.0),
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still()
            )
            .is_empty()
        );
    }

    #[test]
    fn delta_grid_is_bounded() {
        let g = delta_grid(3.0, (-600.0, 600.0));
        assert!(!g.is_empty() && g.len() <= 61, "{}", g.len());
        assert!(delta_grid(100.0, (-5.0, 5.0)).is_empty());
        let g = delta_grid(0.0, (-5.0, 5.0));
        assert!(
            (g[0] + 6.0).abs() < 1e-9 && (g[g.len() - 1] - 6.0).abs() < 1e-9,
            "{g:?}"
        );
    }
}
