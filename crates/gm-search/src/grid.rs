//! Candidate cells over the area and candidate aircraft.

use std::collections::HashMap;

use gm_core::{Geodetic, LocalFrame, Vec3};
use gm_doppler::audio::Event;
use gm_doppler::physics::Track;
use gm_sources::area::{Area, When};
use gm_sources::trace::AircraftTrack;
use gm_sources::types::TypeInfo;
use serde::Serialize;

#[derive(Clone, Copy)]
pub struct Inputs<'a> {
    pub events: &'a [Event],
    pub aircraft: &'a [Candidate],
    pub area: &'a Area,
    /// Clock prior: the recording started at `when.center + δ`, δ ∈ [−minus, +plus].
    pub when: When,
    pub duration_s: f64,
    /// Ellipsoidal ground height at (lat, lon).
    pub ground: &'a (dyn Fn(f64, f64) -> f64 + Sync),
    /// The atmosphere sound travels through (still air unless weather or a wind fit says so).
    pub atmos: &'a crate::atmos::Atmosphere,
    /// The recording's noise level over time: silence counts only where it was quiet. `None`:
    /// every unmatched close pass counts (no audio, e.g. events from a file).
    pub noise: Option<&'a crate::noise::NoiseProfile>,
    /// Other aircraft from around the prior window (not in `aircraft`): guided scoring's null
    /// re-times their passes onto a tested curve's arrival, so short clips can be checked
    /// against chance. Never matched to events. Empty: no decoys (e.g. simulator input).
    pub decoys: &'a [Candidate],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AircraftClass {
    Prop,
    Rotor,
    Jet,
    Unknown,
}

impl AircraftClass {
    /// Farthest CPA slant at which this class's tones can be heard: jets' airframe and engine
    /// tones carry ~3 km (high frequencies are absorbed — B §3); propeller and rotor combs
    /// have no class limit here (each stage applies its own).
    pub fn tone_range_m(self) -> f64 {
        match self {
            AircraftClass::Jet => crate::params::JET_MAX_SLANT_M,
            _ => f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub icao: String,
    pub type_code: Option<String>,
    pub class: AircraftClass,
    /// Knot times relative to `Inputs::when.center`.
    pub tracks: Vec<Track>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Cell {
    pub east: f64,
    pub north: f64,
    /// Microphone position (ground + MIC_H).
    pub geo: Geodetic,
    #[serde(skip)]
    pub ecef: Vec3,
}

/// Cells on `frame`'s tangent plane, `step_m` apart, whose centre lies inside `area`.
pub fn grid(
    area: &Area,
    frame: &LocalFrame,
    step_m: f64,
    ground: &dyn Fn(f64, f64) -> f64,
) -> Vec<Cell> {
    let (s, w, n, e) = area.bbox();
    // Extent of the bbox corners on the tangent plane.
    let corners = [
        (s, w),
        (s, e),
        (n, w),
        (n, e),
        (s, 0.5 * (w + e)),
        (n, 0.5 * (w + e)),
        (0.5 * (s + n), w),
        (0.5 * (s + n), e),
    ];
    let (mut x0, mut x1, mut y0, mut y1) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (la, lo) in corners {
        let p = frame.to_enu(gm_core::geodetic_to_ecef(Geodetic {
            lat_deg: la,
            lon_deg: lo,
            h_m: 0.0,
        }));
        x0 = x0.min(p.x);
        x1 = x1.max(p.x);
        y0 = y0.min(p.y);
        y1 = y1.max(p.y);
    }
    let (i0, i1) = ((x0 / step_m).floor() as i64, (x1 / step_m).ceil() as i64);
    let (j0, j1) = ((y0 / step_m).floor() as i64, (y1 / step_m).ceil() as i64);
    let mut out = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            let (east, north) = (i as f64 * step_m, j as f64 * step_m);
            let g = frame.ground_point(east, north, 0.0);
            if area.contains(g.lat_deg, g.lon_deg) {
                out.push(cell_at(frame, east, north, ground));
            }
        }
    }
    out
}

/// One cell at plane coordinates (east, north).
pub fn cell_at(
    frame: &LocalFrame,
    east: f64,
    north: f64,
    ground: &dyn Fn(f64, f64) -> f64,
) -> Cell {
    let g = frame.ground_point(east, north, 0.0);
    let geo = Geodetic {
        h_m: ground(g.lat_deg, g.lon_deg) + crate::params::MIC_H,
        ..g
    };
    Cell {
        east,
        north,
        geo,
        ecef: gm_core::geodetic_to_ecef(geo),
    }
}

pub fn candidates_from(
    tracks: Vec<AircraftTrack>,
    types: &HashMap<String, TypeInfo>,
) -> Vec<Candidate> {
    use gm_sources::types::EngineClass;
    tracks
        .into_iter()
        .map(|a| {
            let class = match a.type_code.as_deref().and_then(|t| types.get(t)) {
                Some(t) if t.rotorcraft => AircraftClass::Rotor,
                Some(t) => match t.class {
                    EngineClass::Jet => AircraftClass::Jet,
                    EngineClass::Piston | EngineClass::Turboprop | EngineClass::Electric => {
                        AircraftClass::Prop
                    }
                    EngineClass::Unknown => AircraftClass::Unknown,
                },
                None => AircraftClass::Unknown,
            };
            Candidate {
                icao: a.icao,
                type_code: a.type_code,
                class,
                tracks: a.tracks,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::MIC_H;
    use gm_sources::trace::AltBasis;
    use gm_sources::types::EngineClass;

    #[test]
    fn grid_covers_circle() {
        let area = Area::Circle {
            lat: 40.8765,
            lon: -73.525,
            radius_m: 1000.0,
        };
        let frame = LocalFrame::new(Geodetic {
            lat_deg: 40.8765,
            lon_deg: -73.525,
            h_m: 0.0,
        });
        let cells = grid(&area, &frame, 200.0, &|_, _| 12.0);
        assert!((65..=90).contains(&cells.len()), "{}", cells.len()); // 81 lattice points, 12 exactly on the rim
        for c in &cells {
            assert!(area.contains(c.geo.lat_deg, c.geo.lon_deg));
            assert!((c.geo.h_m - (12.0 + MIC_H)).abs() < 1e-9);
            assert!((gm_core::ecef_to_geodetic(c.ecef).h_m - c.geo.h_m).abs() < 1e-6);
        }
    }

    #[test]
    fn class_mapping() {
        let info = |class, rotorcraft| TypeInfo {
            class,
            rotorcraft,
            desc: String::new(),
        };
        let mut types = HashMap::new();
        types.insert("PC12".into(), info(EngineClass::Turboprop, false));
        types.insert("A320".into(), info(EngineClass::Jet, false));
        types.insert("R44".into(), info(EngineClass::Piston, true));
        let mk = |icao: &str, t: Option<&str>| AircraftTrack {
            icao: icao.into(),
            type_code: t.map(str::to_string),
            registration: None,
            alt_basis: AltBasis::Geometric,
            tracks: vec![],
        };
        let c = candidates_from(
            vec![
                mk("a", Some("PC12")),
                mk("b", Some("A320")),
                mk("c", Some("R44")),
                mk("d", None),
                mk("e", Some("ZZZZ")),
            ],
            &types,
        );
        let classes: Vec<AircraftClass> = c.iter().map(|x| x.class).collect();
        assert_eq!(
            classes,
            [
                AircraftClass::Prop,
                AircraftClass::Jet,
                AircraftClass::Rotor,
                AircraftClass::Unknown,
                AircraftClass::Unknown
            ]
        );
    }
}
