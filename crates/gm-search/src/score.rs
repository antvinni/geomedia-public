//! A cell's log-likelihood: every event explained (or floored) under one shared clock offset,
//! minus a penalty for close propeller passes that should have been heard but weren't.

use serde::Serialize;

use crate::curvematch::curve_matches;
use crate::grid::{AircraftClass, Cell, Inputs};
use crate::matching::{Match, matches};
use crate::params::{MODEL_TC_S, OUTLIER_LL, SILENCE_LL, SILENCE_SLANT_M};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CellScore {
    pub ll: f64,
    pub delta_s: f64,
    /// (event index, its match)
    pub matched: Vec<(usize, Match)>,
    /// ICAO of close propeller passes with no event.
    pub silent: Vec<String>,
}

pub fn score_cell(inp: &Inputs, cell: &Cell) -> CellScore {
    let range = (-inp.when.minus_s, inp.when.plus_s);
    // Gate matches (CV fit) and curve-first matches side by side: the two can imply different
    // clocks for the same pair, and the shared-clock assignment below (one event per aircraft)
    // picks whichever agrees with δ*.
    let per_event: Vec<Vec<Match>> = inp
        .events
        .iter()
        .map(|e| {
            let mut all = matches(e, cell, inp.aircraft, range, inp.atmos);
            all.extend(curve_matches(e, cell, inp.aircraft, range, inp.atmos));
            all
        })
        .collect();
    // A match at offset δ scores its own ll plus how far δ sits from the shared clock δ*.
    let score_at = |m: &Match, d: f64| m.ll - 0.5 * ((m.delta_s - d) / MODEL_TC_S).powi(2);
    // Each aircraft explains at most one event (split events of one pass can't count twice):
    // assign greedily, strongest pairing first.
    let total_at = |d: f64| -> (f64, Vec<(usize, Match)>) {
        let mut pairs: Vec<(f64, usize, &Match)> = per_event
            .iter()
            .enumerate()
            .flat_map(|(ei, ms)| {
                ms.iter()
                    .filter(|m| (m.delta_s - d).abs() <= 1.0)
                    .map(move |m| (score_at(m, d), ei, m))
            })
            .filter(|(s, _, _)| *s > OUTLIER_LL)
            .collect();
        pairs.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.aircraft.cmp(&b.2.aircraft))
        });
        let mut used: Vec<(usize, Match)> = Vec::new();
        for (_, ei, m) in pairs {
            if used
                .iter()
                .any(|(e, u)| *e == ei || u.aircraft == m.aircraft)
            {
                continue;
            }
            used.push((ei, m.clone()));
        }
        used.sort_by_key(|(e, _)| *e);
        let explained: f64 = used.iter().map(|(_, m)| score_at(m, d)).sum();
        let total = explained + (per_event.len() - used.len()) as f64 * OUTLIER_LL;
        (total, used)
    };
    // Candidate clocks: every match's own δ (clamped into the prior window).
    let mut best: (f64, f64, Vec<(usize, Match)>) =
        (inp.events.len() as f64 * OUTLIER_LL, 0.0, Vec::new());
    let mut cands: Vec<f64> = per_event
        .iter()
        .flatten()
        .map(|m| m.delta_s.clamp(range.0, range.1))
        .collect();
    cands.sort_by(f64::total_cmp);
    cands.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    for d in cands {
        let (t, used) = total_at(d);
        if t > best.0 {
            best = (t, d, used);
        }
    }
    let (mut ll, delta_s, matched) = best;
    let silent = silent_passes(inp, cell, delta_s, &matched);
    ll += SILENCE_LL * silent.len() as f64;
    CellScore {
        ll,
        delta_s,
        matched,
        silent,
    }
}

/// Propeller/rotor aircraft passing within SILENCE_SLANT_M of the cell during the recording with
/// no matched event, arriving while the recording was quiet: they should have been heard.
fn silent_passes(inp: &Inputs, cell: &Cell, delta: f64, matched: &[(usize, Match)]) -> Vec<String> {
    let mut out = Vec::new();
    for (ai, a) in inp.aircraft.iter().enumerate() {
        if !matches!(a.class, AircraftClass::Prop | AircraftClass::Rotor)
            || matched.iter().any(|(_, m)| m.aircraft == ai)
        {
            continue;
        }
        let (t0, t1) = (delta, delta + inp.duration_s);
        let heard = a.tracks.iter().any(|tr| {
            let Some(w) = tr.window(t0 - 120.0, t1) else {
                return false;
            };
            let Some(c) = inp.atmos.cpa(&w, cell.ecef) else {
                return false;
            };
            let (s0, s1) = w.span();
            // Interior CPA only: a pass that is still approaching or receding at the window edge isn't a full pass.
            c.slant_m <= SILENCE_SLANT_M
                && c.t_emit > s0
                && c.t_emit < s1
                && c.t_arrive >= t0
                && c.t_arrive <= t1
                // Silence is evidence only where the recording was quiet when the pass arrived.
                && inp.noise.is_none_or(|n| n.quiet_at(c.t_arrive - delta))
        });
        if heard {
            out.push(a.icao.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::cell_at;
    use crate::testutil::{Fixture, fixture};
    use gm_core::LocalFrame;
    use gm_sources::area::Area;

    fn inputs<'a>(
        f: &'a Fixture,
        area: &'a Area,
        ground: &'a (dyn Fn(f64, f64) -> f64 + Sync),
    ) -> Inputs<'a> {
        Inputs {
            events: &f.events,
            aircraft: &f.aircraft,
            area,
            when: f.when,
            duration_s: f.duration_s,
            ground,
            atmos: &crate::atmos::STILL_AIR,
            noise: None,
            decoys: &[],
        }
    }

    fn cell(f: &Fixture, e: f64, n: f64) -> Cell {
        let o = f.truth.observer;
        let h = o.h_m;
        cell_at(&LocalFrame::new(o), e, n, &move |_, _| h)
    }

    #[test]
    fn truth_beats_neighbours_and_finds_clock() {
        let f = fixture("two_props.toml", |s| s);
        let area = Area::Circle {
            lat: f.truth.observer.lat_deg,
            lon: f.truth.observer.lon_deg,
            radius_m: 5000.0,
        };
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        let inp = inputs(&f, &area, &g);
        let at = score_cell(&inp, &cell(&f, 0.0, 0.0));
        assert_eq!(at.matched.len(), 2, "{at:?}");
        assert!(at.delta_s.abs() < 0.5, "{at:?}");
        for k in 0..8 {
            let a = f64::from(k) * std::f64::consts::FRAC_PI_4;
            let other = score_cell(&inp, &cell(&f, 1000.0 * a.cos(), 1000.0 * a.sin()));
            assert!(other.ll < at.ll, "dir {k}: {} vs {}", other.ll, at.ll);
        }
    }

    #[test]
    fn decel_truth_matches_both() {
        let f = fixture("decel.toml", |s| s);
        let area = Area::Circle {
            lat: f.truth.observer.lat_deg,
            lon: f.truth.observer.lon_deg,
            radius_m: 5000.0,
        };
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        let at = score_cell(&inputs(&f, &area, &g), &cell(&f, 0.0, 0.0));
        assert_eq!(at.matched.len(), 2, "{at:?}");
        assert!(at.delta_s.abs() < 0.5, "{at:?}");
    }

    #[test]
    fn split_event_counts_once_with_curve() {
        let mut f = fixture("two_props.toml", |s| s);
        let area = Area::Circle {
            lat: f.truth.observer.lat_deg,
            lon: f.truth.observer.lon_deg,
            radius_m: 5000.0,
        };
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        let base = score_cell(&inputs(&f, &area, &g), &cell(&f, 0.0, 0.0));
        let dup = f.events[0].clone();
        f.events.push(dup);
        let with = score_cell(&inputs(&f, &area, &g), &cell(&f, 0.0, 0.0));
        let pc12 = with
            .matched
            .iter()
            .filter(|(_, m)| m.icao == "aaaa01")
            .count();
        assert_eq!(pc12, 1, "{with:?}");
        assert_eq!(with.matched.len(), 2, "{with:?}");
        assert!(
            (base.ll - with.ll - OUTLIER_LL.abs()).abs() < 1e-9,
            "{} {}",
            base.ll,
            with.ll
        );
    }

    #[test]
    fn outlier_floor() {
        let mut f = fixture("two_props.toml", |s| s);
        let area = Area::Circle {
            lat: f.truth.observer.lat_deg,
            lon: f.truth.observer.lon_deg,
            radius_m: 5000.0,
        };
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        let base = score_cell(&inputs(&f, &area, &g), &cell(&f, 0.0, 0.0)).ll;
        let mut junk = f.events[0].clone();
        junk.fit.tc = 500.0;
        junk.fit.v = 25.0;
        junk.fit.d = 80.0;
        f.events.push(junk);
        let with = score_cell(&inputs(&f, &area, &g), &cell(&f, 0.0, 0.0)).ll;
        assert!(
            (base - with - OUTLIER_LL.abs()).abs() < 1e-9,
            "{base} {with}"
        );
    }

    #[test]
    fn silence_unknown_when_masked() {
        // two_props with the second pass's event removed: silent in quiet audio, but not when
        // the recording was loud around its arrival.
        let mut f = fixture("two_props.toml", |s| s);
        let area = Area::Circle {
            lat: f.truth.observer.lat_deg,
            lon: f.truth.observer.lon_deg,
            radius_m: 5000.0,
        };
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        f.events.truncate(1);
        let arrive = f
            .truth
            .events
            .iter()
            .find(|e| e.icao24 == "aaaa02")
            .unwrap()
            .cpa_arrival_audio_s;
        let n = f.duration_s.ceil() as usize;
        let quiet = crate::noise::NoiseProfile::new(1.0, vec![-40.0; n]);
        let loud = crate::noise::NoiseProfile::new(
            1.0,
            (0..n)
                .map(|i| {
                    if (i as f64 - arrive).abs() <= 15.0 {
                        -20.0
                    } else {
                        -40.0
                    }
                })
                .collect(),
        );
        let mut inp = inputs(&f, &area, &g);
        inp.noise = Some(&quiet);
        assert_eq!(
            score_cell(&inp, &cell(&f, 0.0, 0.0)).silent,
            vec!["aaaa02".to_string()]
        );
        inp.noise = Some(&loud);
        assert!(score_cell(&inp, &cell(&f, 0.0, 0.0)).silent.is_empty());
    }

    #[test]
    fn silence_penalty() {
        let mut f = fixture("two_props.toml", |s| s);
        let area = Area::Circle {
            lat: f.truth.observer.lat_deg,
            lon: f.truth.observer.lon_deg,
            radius_m: 5000.0,
        };
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        let both = score_cell(&inputs(&f, &area, &g), &cell(&f, 0.0, 0.0));
        f.events.truncate(1);
        let one = score_cell(&inputs(&f, &area, &g), &cell(&f, 0.0, 0.0));
        assert_eq!(one.silent, vec!["aaaa02".to_string()], "{one:?}");
        let drop = both.ll - one.ll;
        assert!((2.5..4.5).contains(&drop), "{drop}");
    }
}
