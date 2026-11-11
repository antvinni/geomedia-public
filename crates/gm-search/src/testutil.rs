//! Shared test fixtures: simulator renders with their truth tracks standing in for ADS-B.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use gm_core::Epoch;
use gm_doppler::audio::{Audio, Event, EventParams, discover};
use gm_sources::area::When;

use crate::grid::{AircraftClass, Candidate};

#[derive(Clone)]
pub struct Fixture {
    pub events: Vec<Event>,
    pub aircraft: Vec<Candidate>,
    pub truth: gm_sim::Truth,
    pub when: When,
    pub duration_s: f64,
    /// The rendered recording (shared: fixtures are cloned per caller).
    pub audio: Arc<Audio>,
}

/// Rendered fixtures by (edited) scenario text: each is rendered and discovered once per test
/// binary. Every caller gets its own clone, so a test may edit its copy freely.
fn memo() -> &'static Mutex<HashMap<String, Arc<OnceLock<Fixture>>>> {
    static M: OnceLock<Mutex<HashMap<String, Arc<OnceLock<Fixture>>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

pub fn fixture(name: &str, edit: impl Fn(String) -> String) -> Fixture {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/sim")
        .join(name);
    let text = edit(std::fs::read_to_string(p).unwrap());
    // One slot per scenario: different scenarios still build in parallel, and concurrent
    // requests for the same one wait for a single render.
    let slot = memo()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(text.clone())
        .or_default()
        .clone();
    slot.get_or_init(|| build(&text)).clone()
}

fn build(text: &str) -> Fixture {
    let s = gm_sim::parse(text).unwrap();
    let r = gm_sim::render(&s).unwrap();
    let audio = Arc::new(Audio::new(s.sample_rate, r.samples));
    let events = discover(&audio, &EventParams::default(), &[]);
    let aircraft = r
        .truth
        .tracks
        .iter()
        .map(|t| Candidate {
            icao: t.icao24.clone(),
            type_code: Some(t.type_code.clone()),
            class: AircraftClass::Prop,
            tracks: vec![t.track.clone()],
        })
        .collect();
    let start = Epoch::from_rfc3339(&r.truth.start_utc).unwrap();
    Fixture {
        events,
        aircraft,
        when: When {
            center: start,
            minus_s: 5.0,
            plus_s: 5.0,
        },
        duration_s: s.duration_s,
        truth: r.truth,
        audio,
    }
}

/// `n` decoy aircraft for guided nulls: straight passes 3 h after the fixture's start, with
/// geometry unlike the scenario's (speed, altitude, CPA offset, heading cycle independently),
/// as other traffic of the day would be. Translating the real passes instead would copy their
/// speed and altitude, hence their curve asymptotes.
pub fn decoys(f: &Fixture, n: usize) -> Vec<Candidate> {
    use gm_core::{LocalFrame, Vec3};
    use gm_doppler::physics::{Knot, Track};
    const SPEEDS: [f64; 4] = [60.0, 90.0, 200.0, 250.0];
    const ALTS: [f64; 2] = [600.0, 3000.0];
    const OFFSETS: [f64; 4] = [2000.0, -4000.0, 6000.0, -8000.0];
    const HEADINGS: [f64; 4] = [0.0, 45.0, 135.0, 300.0];
    let frame = LocalFrame::new(f.truth.observer);
    (0..n)
        .map(|k| {
            let (v, alt) = (SPEEDS[k % 4], ALTS[k % 2]);
            let (off, h) = (OFFSETS[(k / 2) % 4], HEADINGS[(k / 3) % 4].to_radians());
            let dir = Vec3::new(h.sin(), h.cos(), 0.0);
            let right = Vec3::new(h.cos(), -h.sin(), 0.0);
            let at_cpa = right * -off + Vec3::new(0.0, 0.0, alt);
            let t_cpa = 3.0 * 3600.0 + 30.0 * k as f64;
            let knots = (-150..=150)
                .map(|i| {
                    let tau = f64::from(i);
                    Knot {
                        t: t_cpa + tau,
                        pos: frame.from_enu(at_cpa + dir * (v * tau)),
                        vel: frame.vec_from_enu(dir * v),
                    }
                })
                .collect();
            let icao = format!("decoy{k:02}");
            Candidate {
                icao: icao.clone(),
                type_code: None,
                class: AircraftClass::Prop,
                tracks: vec![Track { id: icao, knots }],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoys_lie_within_the_prop_slant() {
        let f = fixture("suite/quiet.toml", |s| s);
        let obs = gm_core::geodetic_to_ecef(f.truth.observer);
        let m = gm_doppler::physics::Medium::new(343.0, gm_core::Vec3::new(0.0, 0.0, 0.0)).unwrap();
        for d in decoys(&f, 24) {
            let c = gm_doppler::physics::cpa(&d.tracks[0], obs, m).unwrap();
            assert!(
                c.slant_m < crate::params::GUIDE_MAX_SLANT_M,
                "{}",
                c.slant_m
            );
        }
    }

    #[test]
    fn fixture_is_memoised_and_independent() {
        // A distinct scenario text, so no other test has rendered it yet.
        let edit = |s: String| s.replace("seed = 11", "seed = 12345");
        let t = std::time::Instant::now();
        let mut a = fixture("one_pass.toml", edit);
        let first = t.elapsed();
        let t = std::time::Instant::now();
        let b = fixture("one_pass.toml", edit);
        let second = t.elapsed();
        assert!(
            second < first / 10,
            "second call {second:?} vs first {first:?}"
        );
        assert_eq!(a.events.len(), b.events.len());
        // A test that edits its copy can't affect anyone else's.
        a.events.clear();
        a.aircraft[0].class = AircraftClass::Jet;
        let c = fixture("one_pass.toml", edit);
        assert_eq!(c.events.len(), b.events.len());
        assert_eq!(c.aircraft[0].class, AircraftClass::Prop);
    }
}
