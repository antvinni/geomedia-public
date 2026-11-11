//! Scenario files (TOML): observer, atmosphere, noise and straight-line aircraft.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("cannot read scenario: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid scenario: {0}")]
    Parse(String),
    #[error("invalid scenario: {field}: {reason}")]
    Invalid { field: String, reason: String },
}

/// Limits that keep a hostile or mistaken scenario from exhausting memory or time.
pub const MAX_SAMPLE_RATE: u32 = 384_000;
pub const MAX_DURATION_S: f64 = 4.0 * 3600.0;
pub const MAX_SAMPLES: f64 = 2.0e8;
pub const MIN_ADSB_DT_S: f64 = 0.01;
pub const MAX_KNOTS: f64 = 1.0e6;

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub seed: u64,
    pub sample_rate: u32,
    pub duration_s: f64,
    pub start_utc: String,
    #[serde(default = "one")]
    pub adsb_dt_s: f64,
    pub observer: ObserverCfg,
    pub atmosphere: AtmosphereCfg,
    pub noise: NoiseCfg,
    pub aircraft: Vec<AircraftCfg>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObserverCfg {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub h_ell_m: f64,
    pub mic_height_m: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtmosphereCfg {
    pub c: f64,
    pub wind_enu: [f64; 3],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseCfg {
    /// Amplitude-1 tone at 1 km over the noise in a 1 Hz band.
    pub snr_db: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AircraftCfg {
    pub id: String,
    pub icao24: String,
    pub type_code: String,
    pub f0: f64,
    pub harmonics: Vec<f64>,
    pub amplitude: f64,
    #[serde(default)]
    pub drift_per_s: f64,
    /// A broadband (jet-like) source in addition to the tones; `harmonics = [0.0]` makes the
    /// aircraft broadband only.
    #[serde(default)]
    pub broadband: Option<BroadbandCfg>,
    /// `false`: the aircraft is ADS-B traffic only (a truth track, no sound) — background
    /// traffic for location nulls.
    #[serde(default = "yes")]
    pub audible: bool,
    pub straight: StraightCfg,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BroadbandCfg {
    /// Spectral density at 1 km over the noise floor's, dB, averaged over [low_hz, high_hz].
    pub level_db: f64,
    pub low_hz: f64,
    pub high_hz: f64,
    /// ISO 9613-1 air absorption at 15 °C, 70 % RH (per octave band).
    #[serde(default = "yes")]
    pub absorption: bool,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StraightCfg {
    pub heading_deg: f64,
    pub speed_mps: f64,
    pub alt_agl_m: f64,
    pub cpa_offset_m: f64,
    /// Audio time (s) of the geometric CPA.
    pub cpa_time_s: f64,
    pub half_span_s: f64,
    /// Along-track acceleration: speed(τ) = speed_mps + accel_mps2·τ, τ = t − cpa_time_s.
    #[serde(default)]
    pub accel_mps2: f64,
    /// Vertical rate: height(τ) = alt_agl_m + climb_mps·τ.
    #[serde(default)]
    pub climb_mps: f64,
}

pub fn load(path: &Path) -> Result<Scenario, SimError> {
    parse(&std::fs::read_to_string(path)?)
}

/// Parse and validate scenario TOML.
pub fn parse(text: &str) -> Result<Scenario, SimError> {
    let s: Scenario = toml::from_str(text).map_err(|e| SimError::Parse(e.to_string()))?;
    let bad = |field: &str, reason: &str| {
        Err(SimError::Invalid {
            field: field.into(),
            reason: reason.into(),
        })
    };
    let finite_pos = |x: f64| x.is_finite() && x > 0.0;
    if s.sample_rate == 0 || s.sample_rate > MAX_SAMPLE_RATE {
        return bad(
            "sample_rate",
            &format!("must be between 1 and {MAX_SAMPLE_RATE} Hz"),
        );
    }
    if !finite_pos(s.duration_s) || s.duration_s > MAX_DURATION_S {
        return bad(
            "duration_s",
            &format!("must be between 0 and {MAX_DURATION_S} s"),
        );
    }
    if s.duration_s * f64::from(s.sample_rate) > MAX_SAMPLES {
        return bad(
            "duration_s",
            &format!("duration × sample_rate exceeds {MAX_SAMPLES} samples"),
        );
    }
    if !(s.adsb_dt_s.is_finite() && s.adsb_dt_s >= MIN_ADSB_DT_S) {
        return bad("adsb_dt_s", &format!("must be at least {MIN_ADSB_DT_S} s"));
    }
    if gm_core::Epoch::from_rfc3339(&s.start_utc).is_none() {
        return bad("start_utc", "must be an RFC 3339 timestamp");
    }
    let o = &s.observer;
    if !(o.lat_deg.is_finite()
        && o.lat_deg.abs() <= 90.0
        && o.lon_deg.is_finite()
        && o.lon_deg.abs() <= 180.0)
    {
        return bad("observer", "lat/lon out of range");
    }
    if !(o.h_ell_m.is_finite() && o.mic_height_m.is_finite()) {
        return bad("observer", "heights must be finite");
    }
    if !finite_pos(s.atmosphere.c) {
        return bad("atmosphere.c", "must be > 0");
    }
    let w = s.atmosphere.wind_enu;
    if !w.iter().all(|x| x.is_finite())
        || (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt() >= s.atmosphere.c
    {
        return bad(
            "atmosphere.wind_enu",
            "must be finite and slower than sound",
        );
    }
    if !s.noise.snr_db.is_finite() {
        return bad("noise.snr_db", "must be finite");
    }
    for (i, a) in s.aircraft.iter().enumerate() {
        let f = |name: &str| format!("aircraft[{i}].{name}");
        if !finite_pos(a.f0) {
            return bad(&f("f0"), "must be > 0");
        }
        if a.harmonics.is_empty() || !a.harmonics.iter().all(|h| h.is_finite()) {
            return bad(&f("harmonics"), "must be a non-empty list of numbers");
        }
        if !a.amplitude.is_finite() {
            return bad(&f("amplitude"), "must be finite");
        }
        if let Some(b) = &a.broadband {
            if !b.level_db.is_finite() {
                return bad(&f("broadband.level_db"), "must be finite");
            }
            if !(finite_pos(b.low_hz) && b.high_hz.is_finite() && b.high_hz > b.low_hz) {
                return bad(&f("broadband"), "needs 0 < low_hz < high_hz");
            }
        }
        if !a.drift_per_s.is_finite() {
            return bad(&f("drift_per_s"), "must be finite");
        }
        let st = &a.straight;
        if !finite_pos(st.speed_mps) || st.speed_mps >= s.atmosphere.c {
            return bad(&f("straight.speed_mps"), "must be > 0 and subsonic");
        }
        if !finite_pos(st.half_span_s) || st.half_span_s > MAX_DURATION_S {
            return bad(
                &f("straight.half_span_s"),
                &format!("must be between 0 and {MAX_DURATION_S} s"),
            );
        }
        if 2.0 * st.half_span_s / s.adsb_dt_s > MAX_KNOTS {
            return bad(
                &f("straight.half_span_s"),
                &format!("half_span_s / adsb_dt_s gives more than {MAX_KNOTS} knots"),
            );
        }
        if ![st.heading_deg, st.alt_agl_m, st.cpa_offset_m, st.cpa_time_s]
            .iter()
            .all(|x| x.is_finite())
        {
            return bad(&f("straight"), "values must be finite");
        }
        if !st.accel_mps2.is_finite() || !st.climb_mps.is_finite() {
            return bad(&f("straight"), "values must be finite");
        }
        // Manoeuvre limits apply only to a manoeuvre: plain straight passes keep v1's limits.
        let h = st.half_span_s;
        if st.accel_mps2 != 0.0 {
            let v_lo = st.speed_mps - st.accel_mps2.abs() * h;
            let v_hi = st.speed_mps + st.accel_mps2.abs() * h;
            if v_lo < 20.0 || v_hi.hypot(st.climb_mps) >= s.atmosphere.c {
                return bad(
                    &f("straight.accel_mps2"),
                    "speed must stay between 20 m/s and the sound speed over the track",
                );
            }
        }
        if st.climb_mps != 0.0 {
            let v_hi = st.speed_mps + st.accel_mps2.abs() * h;
            if st.alt_agl_m - st.climb_mps.abs() * h < 50.0
                || v_hi.hypot(st.climb_mps) >= s.atmosphere.c
            {
                return bad(
                    &f("straight.climb_mps"),
                    "height must stay at least 50 m above ground, and the 3-D speed subsonic, over the track",
                );
            }
        }
    }
    Ok(s)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const SMALL: &str = r#"
seed = 7
sample_rate = 4000
duration_s = 60.0
start_utc = "2026-09-08T20:50:00Z"
[observer]
lat_deg = 40.8765
lon_deg = -73.525
h_ell_m = -20.0
mic_height_m = 1.5
[atmosphere]
c = 343.0
wind_enu = [8.0, -9.0, 0.0]
[noise]
snr_db = 40.0
[[aircraft]]
id = "a"
icao24 = "a1dfeb"
type_code = "PC12"
f0 = 87.0
harmonics = [1.0, 0.5]
amplitude = 1.0
[aircraft.straight]
heading_deg = 70.0
speed_mps = 130.0
alt_agl_m = 1500.0
cpa_offset_m = 900.0
cpa_time_s = 25.0
half_span_s = 80.0
"#;

    #[test]
    fn parses_small() {
        let s = parse(SMALL).unwrap();
        assert_eq!(s.sample_rate, 4000);
        assert_eq!(s.adsb_dt_s, 1.0);
        assert_eq!(s.aircraft[0].drift_per_s, 0.0);
    }

    #[test]
    fn rejects_bad_scenarios() {
        let unknown = SMALL.replace("seed = 7", "seed = 7\nbogus = 1");
        let e = parse(&unknown).unwrap_err().to_string();
        assert!(e.contains("bogus"), "{e}");
        let e = parse(&SMALL.replace("sample_rate = 4000", "sample_rate = 0"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("sample_rate"), "{e}");
        let e = parse(&SMALL.replace("duration_s = 60.0", "duration_s = -1.0"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("duration_s"), "{e}");
        let e = parse(&SMALL.replace("f0 = 87.0", "f0 = 0.0"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("f0"), "{e}");
        let e = parse(&SMALL.replace("harmonics = [1.0, 0.5]", "harmonics = []"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("harmonics"), "{e}");
        let e = parse(&SMALL.replace("c = 343.0", "c = -5.0"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("atmosphere.c"), "{e}");
        let e = parse(&SMALL.replace("2026-09-08T20:50:00Z", "yesterday"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("start_utc"), "{e}");
    }

    #[test]
    fn no_manoeuvre_keeps_old_limits() {
        // Scenarios valid before manoeuvres existed stay valid: a slow, low helicopter.
        let slow_low = SMALL
            .replace("speed_mps = 130.0", "speed_mps = 15.0")
            .replace("alt_agl_m = 1500.0", "alt_agl_m = 30.0");
        parse(&slow_low).unwrap();
        // A manoeuvre's 3-D speed must stay subsonic.
        let steep = SMALL.replace("half_span_s = ", "climb_mps = 340.0\nhalf_span_s = ");
        let e = parse(&steep).unwrap_err().to_string();
        assert!(e.contains("straight.climb_mps"), "{e}");
    }

    #[test]
    fn rejects_stall_or_ground() {
        let base = SMALL.replace("speed_mps = 130.0", "speed_mps = 95.0");
        let stall = base.replace("half_span_s = ", "accel_mps2 = -1.0\nhalf_span_s = ");
        let e = parse(&stall).unwrap_err().to_string();
        assert!(e.contains("straight.accel_mps2"), "{e}");
        let ground = base.replace("half_span_s = ", "climb_mps = -20.0\nhalf_span_s = ");
        let e = parse(&ground).unwrap_err().to_string();
        assert!(e.contains("straight.climb_mps"), "{e}");
        let ok = base.replace(
            "half_span_s = ",
            "accel_mps2 = -0.2\nclimb_mps = -3.0\nhalf_span_s = ",
        );
        parse(&ok).unwrap();
    }

    #[test]
    fn rejects_resource_bombs() {
        for (from, to, field) in [
            ("duration_s = 60.0", "duration_s = 1e300", "duration_s"),
            ("duration_s = 60.0", "duration_s = 1e7", "duration_s"),
            ("seed = 7", "seed = 7\nadsb_dt_s = 1e-300", "adsb_dt_s"),
            ("half_span_s = 80.0", "half_span_s = 1e300", "half_span_s"),
            (
                "sample_rate = 4000",
                "sample_rate = 3000000000",
                "sample_rate",
            ),
        ] {
            let e = parse(&SMALL.replace(from, to)).unwrap_err().to_string();
            assert!(e.contains(field), "{to}: {e}");
        }
    }
}
