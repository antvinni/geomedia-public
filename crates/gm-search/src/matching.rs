//! Which aircraft could have produced an event, heard at a given cell, and what recording-clock
//! offset each would imply.

use crate::atmos::Atmosphere;
use gm_doppler::audio::Event;
use serde::Serialize;

use crate::grid::{Candidate, Cell};
use crate::params::{GATE_SIGMA, MODEL_D_FRAC, MODEL_V_FRAC};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Match {
    /// Index into `Inputs::aircraft`.
    pub aircraft: usize,
    pub icao: String,
    /// Implied clock offset: CPA arrival (track time) − event tc (audio time).
    pub delta_s: f64,
    pub ll: f64,
    pub slant_m: f64,
    pub speed_mps: f64,
    /// How the pairing was found: CV-fit gates, or curve-first alignment.
    pub via: MatchVia,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchVia {
    Gate,
    Curve,
}

/// Whether `event` may be explained by aircraft `a`. A guided event only by the aircraft whose
/// curve proposed it (re-attributing it to another — say one of a busy sky's jets near a wrong
/// cell — would satisfy the two-aircraft rule falsely). Jets only through their own guided
/// events: letting a freely discovered event match a jet moved a real answer 4 km (Oyster Bay,
/// a junk event on jet a613e7).
pub fn may_match(event: &Event, a: &Candidate) -> bool {
    if event.guided {
        return event.guide_icao.as_deref() == Some(a.icao.as_str());
    }
    a.class != crate::grid::AircraftClass::Jet
}

/// All plausible matches of `event` heard at `cell`; δ must lie in `delta_range` (± 1 s).
pub fn matches(
    event: &Event,
    cell: &Cell,
    aircraft: &[Candidate],
    delta_range: (f64, f64),
    atmos: &Atmosphere,
) -> Vec<Match> {
    let f = &event.fit;
    let (lo, hi) = (delta_range.0 - 1.0, delta_range.1 + 1.0);
    let sd = (f.sigma_d.powi(2) + (MODEL_D_FRAC * f.d).powi(2)).sqrt();
    let sv = (f.sigma_v.powi(2) + (MODEL_V_FRAC * f.v).powi(2)).sqrt();
    let mut out = Vec::new();
    for (ai, a) in aircraft.iter().enumerate() {
        if !may_match(event, a) {
            continue;
        }
        for tr in &a.tracks {
            // CPA near this event only: the emission happened shortly before tc + δ.
            let Some(w) = tr.window(f.tc + lo - 120.0, f.tc + hi + 60.0) else {
                continue;
            };
            let Some(c) = atmos.cpa(&w, cell.ecef) else {
                continue;
            };
            let delta = c.t_arrive - f.tc;
            if delta < lo || delta > hi || c.slant_m > a.class.tone_range_m() {
                continue;
            }
            let (dd, dv) = ((c.slant_m - f.d) / sd, (c.ground_speed_mps - f.v) / sv);
            if dd.abs() > GATE_SIGMA || dv.abs() > GATE_SIGMA {
                continue;
            }
            out.push(Match {
                aircraft: ai,
                icao: a.icao.clone(),
                delta_s: delta,
                ll: -0.5 * (dd * dd + dv * dv),
                slant_m: c.slant_m,
                speed_mps: c.ground_speed_mps,
                via: MatchVia::Gate,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{AircraftClass, cell_at};
    use crate::testutil::fixture;
    use gm_core::LocalFrame;

    fn truth_cell(f: &crate::testutil::Fixture) -> Cell {
        let o = f.truth.observer;
        let frame = LocalFrame::new(o);
        let h = o.h_m;
        cell_at(&frame, 0.0, 0.0, &move |_, _| h)
    }

    #[test]
    fn truth_cell_matches_right_aircraft() {
        let f = fixture("two_props.toml", |s| s);
        let cell = truth_cell(&f);
        assert_eq!(f.events.len(), 2);
        for (e, want) in f.events.iter().zip(["aaaa01", "aaaa02"]) {
            let m = matches(
                e,
                &cell,
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still(),
            );
            let best = m
                .iter()
                .max_by(|a, b| a.ll.total_cmp(&b.ll))
                .unwrap_or_else(|| panic!("no match for {want}"));
            assert_eq!(best.icao, want);
            assert!(best.delta_s.abs() < 0.5, "{best:?}");
        }
    }

    #[test]
    fn far_cell_no_match() {
        let f = fixture("two_props.toml", |s| s);
        let o = f.truth.observer;
        let frame = LocalFrame::new(o);
        let h = o.h_m;
        let far = cell_at(&frame, 5000.0, -3000.0, &move |_, _| h);
        let near = matches(
            &f.events[0],
            &truth_cell(&f),
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        let away = matches(
            &f.events[0],
            &far,
            &f.aircraft,
            (-5.0, 5.0),
            &crate::atmos::Atmosphere::still(),
        );
        let best = |v: &[Match]| v.iter().map(|m| m.ll).fold(f64::NEG_INFINITY, f64::max);
        assert!(
            away.is_empty() || best(&away) < best(&near) - 5.0,
            "{away:?}"
        );
    }

    #[test]
    fn jets_match_only_within_tone_range() {
        // A jet's tones carry ~3 km: as jets, the same aircraft match only passes that close.
        let mut f = fixture("two_props.toml", |s| s);
        for a in &mut f.aircraft {
            a.class = AircraftClass::Jet;
        }
        let limit = crate::params::JET_MAX_SLANT_M;
        for (e, t) in f.events.iter().zip(&f.truth.events) {
            // A discovered event never matches a jet: on real street audio a junk event matched
            // a jet and moved the answer 4 km. Jets are matched by guided jet events only.
            let none = matches(
                e,
                &truth_cell(&f),
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still(),
            );
            assert!(none.is_empty(), "{none:?}");
            // Guided by some other aircraft: never re-attributed.
            let mut other = e.clone();
            other.guided = true;
            other.guide_icao = Some("zzzz99".into());
            let none = matches(
                &other,
                &truth_cell(&f),
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still(),
            );
            assert!(none.is_empty(), "{none:?}");
            let mut e = e.clone();
            e.guided = true;
            e.guide_icao = Some(t.icao24.clone());
            let m = matches(
                &e,
                &truth_cell(&f),
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still(),
            );
            assert!(m.iter().all(|m| m.slant_m <= limit), "{m:?}");
            assert_eq!(!m.is_empty(), t.slant_m <= limit, "{t:?}: {m:?}");
        }
        // Beyond the range: a 4.5 km pass (move the cell 3 km off) matches no jet.
        let o = f.truth.observer;
        let h = o.h_m;
        let far = crate::grid::cell_at(&gm_core::LocalFrame::new(o), 0.0, 4500.0, &move |_, _| h);
        for (e, t) in f.events.iter().zip(&f.truth.events) {
            let mut e = e.clone();
            e.guided = true;
            e.guide_icao = Some(t.icao24.clone());
            let m = matches(
                &e,
                &far,
                &f.aircraft,
                (-5.0, 5.0),
                &crate::atmos::Atmosphere::still(),
            );
            assert!(m.iter().all(|m| m.slant_m <= limit), "{m:?}");
        }
    }
}
