//! Detection benchmark: which aircraft passes a recording at a known position should contain
//! (from the tracks), and which of them the events found. Measures the detector, not the
//! locator — it needs the truth position, so it is for evaluation only.

use gm_core::Vec3;
use gm_doppler::audio::Event;
use serde::Serialize;

use crate::atmos::Atmosphere;
use crate::grid::{AircraftClass, Candidate};

/// One pass the recording should contain.
#[derive(Debug, Clone, Serialize)]
pub struct ExpectedPass {
    pub icao: String,
    pub class: AircraftClass,
    pub slant_m: f64,
    /// Ground speed at CPA.
    pub speed_mps: f64,
    /// CPA arrival, seconds from the recording's start (clock offset 0).
    pub t_audio: f64,
}

/// Every track's CPA at `obs` within `max_slant_m` whose arrival falls inside the recording.
pub fn expected_passes(
    aircraft: &[Candidate],
    obs: Vec3,
    duration_s: f64,
    atmos: &Atmosphere,
    max_slant_m: f64,
) -> Vec<ExpectedPass> {
    let mut v: Vec<ExpectedPass> = aircraft
        .iter()
        .flat_map(|a| {
            a.tracks.iter().filter_map(move |tr| {
                let c = atmos.cpa(tr, obs)?;
                (c.slant_m <= max_slant_m && c.t_arrive >= 0.0 && c.t_arrive <= duration_s).then(
                    || ExpectedPass {
                        icao: a.icao.clone(),
                        class: a.class,
                        slant_m: c.slant_m,
                        speed_mps: c.ground_speed_mps,
                        t_audio: c.t_arrive,
                    },
                )
            })
        })
        .collect();
    v.sort_by(|a, b| a.t_audio.total_cmp(&b.t_audio));
    v
}

/// A find's CPA time must be within this of the pass (front-end analysis: at ±10 s most real
/// "finds" were chance; the fits that matter land well within 3 s).
pub const TOL_S: f64 = 3.0;

/// Slant-range bins (upper edges, m) the recall is reported in.
pub const SLANT_BINS_M: [f64; 5] = [1000.0, 2000.0, 3000.0, 5000.0, 8000.0];

/// A missed pass's nearest discovered event within NEAR_MISS_REACH_S must be this close in time
/// to be reported (beyond it: "no event").
pub const NEAR_MISS_REACH_S: f64 = 60.0;

/// Why the nearest event didn't count as finding a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MissReason {
    Time,
    Geometry,
    TimeAndGeometry,
    /// It would have matched, but was assigned to another pass.
    Taken,
}

#[derive(Debug, Clone, Serialize)]
pub struct NearMiss {
    /// Event CPA time − pass arrival.
    pub dt_s: f64,
    /// Event / pass.
    pub d_ratio: f64,
    pub v_ratio: f64,
    pub reason: MissReason,
}

/// A pass discovery didn't find, with its nearest event (if any within reach).
#[derive(Debug, Clone, Serialize)]
pub struct Missed {
    pub icao: String,
    pub slant_m: f64,
    pub t_audio: f64,
    pub nearest: Option<NearMiss>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Detection {
    /// Per slant bin: (expected, found by discovery, found counting guided events too).
    pub by_slant: Vec<(f64, usize, usize, usize)>,
    /// Per class name: (expected, found by discovery).
    pub by_class: Vec<(String, usize, usize)>,
    /// Discovered events matching no expected pass (junk, or aircraft beyond the bins / not in
    /// the tracks).
    pub unmatched_discovered: usize,
    pub unmatched_guided: usize,
    pub discovered: usize,
    pub guided: usize,
    /// (icao, slant m, |tc error| s) for each pass found by discovery.
    pub found: Vec<(String, f64, f64)>,
    /// Every pass inside the slant bins that discovery missed, with its nearest event.
    pub missed: Vec<Missed>,
    /// Fraction of the recording within ±tol of some expected pass: an unmatched event lands
    /// there by chance this often (what time-only matching would have credited).
    pub chance_coverage: f64,
}

/// An event's fitted geometry must agree with the pass within this factor (slant and speed):
/// time alone matches by chance — a ±10 s window around each pass covers ~11 % of a busy hour.
pub const GEOM_FACTOR: f64 = 2.0;

fn geometry_agrees(e: &Event, p: &ExpectedPass) -> bool {
    let within = |a: f64, b: f64| a > 0.0 && b > 0.0 && (a / b).ln().abs() <= GEOM_FACTOR.ln();
    within(e.fit.d, p.slant_m) && within(e.fit.v, p.speed_mps)
}

/// Match events to expected passes: each pass takes the nearest unused event whose CPA time is
/// within `tol_s` and whose geometry agrees (nearest pairs first).
fn assign(events: &[&Event], passes: &[ExpectedPass], tol_s: f64) -> Vec<Option<usize>> {
    let mut pairs: Vec<(f64, usize, usize)> = passes
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| {
            events
                .iter()
                .enumerate()
                .map(move |(ei, e)| ((e.fit.tc - p.t_audio).abs(), pi, ei))
        })
        // ±max(tol, 2σ_tc): a fit's own CPA-time uncertainty widens its window.
        .filter(|&(d, pi, ei)| {
            d <= tol_s.max(2.0 * events[ei].fit.sigma_tc)
                && geometry_agrees(events[ei], &passes[pi])
        })
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut by_pass = vec![None; passes.len()];
    let mut used = vec![false; events.len()];
    for (_, pi, ei) in pairs {
        if by_pass[pi].is_none() && !used[ei] {
            by_pass[pi] = Some(ei);
            used[ei] = true;
        }
    }
    by_pass
}

pub fn detection(
    events: &[Event],
    passes: &[ExpectedPass],
    tol_s: f64,
    duration_s: f64,
) -> Detection {
    let disc: Vec<&Event> = events.iter().filter(|e| !e.guided).collect();
    let all: Vec<&Event> = events.iter().collect();
    let d_hit = assign(&disc, passes, tol_s);
    let a_hit = assign(&all, passes, tol_s);
    let bin_of = |s: f64| SLANT_BINS_M.iter().position(|&b| s <= b);
    let mut by_slant: Vec<(f64, usize, usize, usize)> =
        SLANT_BINS_M.iter().map(|&b| (b, 0, 0, 0)).collect();
    let mut classes: Vec<(String, usize, usize)> = Vec::new();
    let mut found = Vec::new();
    for (i, p) in passes.iter().enumerate() {
        if let Some(b) = bin_of(p.slant_m) {
            by_slant[b].1 += 1;
            by_slant[b].2 += usize::from(d_hit[i].is_some());
            by_slant[b].3 += usize::from(a_hit[i].is_some());
        }
        let name = format!("{:?}", p.class);
        let k = match classes.iter().position(|c| c.0 == name) {
            Some(k) => k,
            None => {
                classes.push((name, 0, 0));
                classes.len() - 1
            }
        };
        classes[k].1 += 1;
        if let Some(ei) = d_hit[i] {
            classes[k].2 += 1;
            found.push((
                p.icao.clone(),
                p.slant_m,
                (disc[ei].fit.tc - p.t_audio).abs(),
            ));
        }
    }
    let missed: Vec<Missed> = passes
        .iter()
        .enumerate()
        .filter(|(i, p)| d_hit[*i].is_none() && bin_of(p.slant_m).is_some())
        .map(|(_, p)| {
            let nearest = disc
                .iter()
                .filter(|e| (e.fit.tc - p.t_audio).abs() <= NEAR_MISS_REACH_S)
                .min_by(|a, b| {
                    (a.fit.tc - p.t_audio)
                        .abs()
                        .total_cmp(&(b.fit.tc - p.t_audio).abs())
                })
                .map(|e| {
                    let dt = e.fit.tc - p.t_audio;
                    let time_ok = dt.abs() <= tol_s.max(2.0 * e.fit.sigma_tc);
                    let geom_ok = geometry_agrees(e, p);
                    NearMiss {
                        dt_s: dt,
                        d_ratio: e.fit.d / p.slant_m,
                        v_ratio: e.fit.v / p.speed_mps,
                        reason: match (time_ok, geom_ok) {
                            (true, true) => MissReason::Taken,
                            (false, true) => MissReason::Time,
                            (true, false) => MissReason::Geometry,
                            (false, false) => MissReason::TimeAndGeometry,
                        },
                    }
                });
            Missed {
                icao: p.icao.clone(),
                slant_m: p.slant_m,
                t_audio: p.t_audio,
                nearest,
            }
        })
        .collect();
    let used_a: Vec<usize> = a_hit.iter().flatten().copied().collect();
    let used_d: Vec<usize> = d_hit.iter().flatten().copied().collect();
    let unmatched_guided = all
        .iter()
        .enumerate()
        .filter(|(i, e)| e.guided && !used_a.contains(i))
        .count();
    let mut iv: Vec<(f64, f64)> = passes
        .iter()
        .map(|p| {
            (
                (p.t_audio - tol_s).max(0.0),
                (p.t_audio + tol_s).min(duration_s),
            )
        })
        .collect();
    iv.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut covered, mut end) = (0.0, f64::NEG_INFINITY);
    for (a, b) in iv {
        let a = a.max(end);
        if b > a {
            covered += b - a;
        }
        end = end.max(b);
    }
    Detection {
        missed,
        chance_coverage: if duration_s > 0.0 {
            covered / duration_s
        } else {
            0.0
        },
        by_slant,
        by_class: classes,
        unmatched_discovered: disc.len() - used_d.len(),
        unmatched_guided,
        discovered: disc.len(),
        guided: all.len() - disc.len(),
        found,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::fixture;

    #[test]
    fn geometry_must_agree() {
        // At the right time but the wrong geometry (a road vehicle, 76 m at 28 m/s, near a jet
        // at 1355 m, 80 m/s): not a find. The same time with matching geometry is.
        let proto = fixture("one_pass.toml", |s| s).events[0].clone();
        let pass = ExpectedPass {
            icao: "a09590".into(),
            class: AircraftClass::Jet,
            slant_m: 1355.0,
            speed_mps: 80.0,
            t_audio: 500.0,
        };
        let ev = |d: f64, v: f64| {
            let mut e = proto.clone();
            e.fit.tc = 509.8;
            e.fit.d = d;
            e.fit.v = v;
            e
        };
        let found = |e: Event| {
            detection(&[e], std::slice::from_ref(&pass), 10.0, 3600.0)
                .by_slant
                .iter()
                .map(|b| b.2)
                .sum::<usize>()
        };
        assert_eq!(found(ev(76.0, 28.0)), 0);
        assert_eq!(found(ev(1300.0, 85.0)), 1);
    }

    #[test]
    fn tolerance_widens_with_the_fits_own_sigma() {
        // 5 s off: outside ±3 s, but inside 2σ when the fit's own σ_tc is 3 s.
        let proto = fixture("one_pass.toml", |s| s).events[0].clone();
        let pass = ExpectedPass {
            icao: "x".into(),
            class: AircraftClass::Prop,
            slant_m: 1000.0,
            speed_mps: 100.0,
            t_audio: 500.0,
        };
        let ev = |sigma: f64| {
            let mut e = proto.clone();
            e.fit.tc = 505.0;
            e.fit.d = 1000.0;
            e.fit.v = 100.0;
            e.fit.sigma_tc = sigma;
            e
        };
        let found = |e: Event| {
            detection(&[e], std::slice::from_ref(&pass), TOL_S, 3600.0)
                .by_slant
                .iter()
                .map(|b| b.2)
                .sum::<usize>()
        };
        assert_eq!(found(ev(0.5)), 0);
        assert_eq!(found(ev(3.0)), 1);
        assert_eq!(SLANT_BINS_M.last(), Some(&8000.0));
    }

    fn ev(tc: f64, d: f64, v: f64) -> Event {
        let mut e = fixture("one_pass.toml", |s| s).events[0].clone();
        e.fit.tc = tc;
        e.fit.d = d;
        e.fit.v = v;
        e.fit.sigma_tc = 0.1;
        e
    }

    fn pass(t: f64, slant: f64, v: f64) -> ExpectedPass {
        ExpectedPass {
            icao: "p1".into(),
            class: AircraftClass::Prop,
            slant_m: slant,
            speed_mps: v,
            t_audio: t,
        }
    }

    #[test]
    fn missed_pass_reports_nearest_event() {
        // Pass at 100 s, 1000 m, 60 m/s; event 7 s late with matching geometry: missed on time.
        let d = detection(
            &[ev(107.0, 1000.0, 60.0)],
            &[pass(100.0, 1000.0, 60.0)],
            TOL_S,
            600.0,
        );
        assert_eq!(d.missed.len(), 1);
        let n = d.missed[0].nearest.as_ref().unwrap();
        assert!((n.dt_s - 7.0).abs() < 1e-9);
        assert!(matches!(n.reason, MissReason::Time));
        // Right time, wrong geometry: missed on geometry.
        let d = detection(
            &[ev(100.5, 80.0, 25.0)],
            &[pass(100.0, 1000.0, 60.0)],
            TOL_S,
            600.0,
        );
        assert!(matches!(
            d.missed[0].nearest.as_ref().unwrap().reason,
            MissReason::Geometry
        ));
    }

    #[test]
    fn missed_pass_without_any_event_in_reach() {
        let d = detection(
            &[ev(400.0, 1000.0, 60.0)],
            &[pass(100.0, 1000.0, 60.0)],
            TOL_S,
            600.0,
        );
        assert!(d.missed[0].nearest.is_none());
    }

    #[test]
    fn two_props_both_passes_found() {
        let f = fixture("two_props.toml", |s| s);
        let p = expected_passes(
            &f.aircraft,
            f.truth.mic_ecef,
            f.duration_s,
            &Atmosphere::still(),
            5000.0,
        );
        assert_eq!(p.len(), 2, "{p:#?}");
        for (e, t) in p.iter().zip(&f.truth.events) {
            assert!(
                (e.t_audio - t.cpa_arrival_audio_s).abs() < 1.0,
                "{e:?} vs {t:?}"
            );
        }
        let d = detection(&f.events, &p, 10.0, f.duration_s);
        let found: usize = d.by_slant.iter().map(|b| b.2).sum();
        assert_eq!(found, 2, "{d:#?}");
        assert_eq!(d.unmatched_discovered, 0, "{d:#?}");
    }

    #[test]
    fn quiet_both_passes_found_by_discovery() {
        // suite/quiet: the ridge stage finds the near B350; the blind template stage finds the
        // far PC12 (once only the guided target) — from the audio alone.
        let f = fixture("suite/quiet.toml", |s| s);
        let p = expected_passes(
            &f.aircraft,
            f.truth.mic_ecef,
            f.duration_s,
            &Atmosphere::still(),
            5000.0,
        );
        let d = detection(&f.events, &p, 10.0, f.duration_s);
        let found: usize = d.by_slant.iter().map(|b| b.2).sum();
        let expected: usize = d.by_slant.iter().map(|b| b.1).sum();
        assert_eq!((expected, found), (2, 2), "{d:#?}");
        assert_eq!(d.unmatched_discovered, 0, "{d:#?}");
    }
}
