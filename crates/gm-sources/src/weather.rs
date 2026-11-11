//! Wind and temperature by height from Open-Meteo's historical-forecast API (pressure levels),
//! cached per site and day, turned into a layered sound-speed and wind column.

use std::path::{Path, PathBuf};

use gm_core::{Epoch, LocalFrame};
use gm_doppler::physics::{Column, ColumnNode};

use crate::SourceError;

/// Pressure levels requested (hPa): the lowest ~3 km, where light aircraft fly.
pub const LEVELS_HPA: [u32; 8] = [1000, 975, 950, 925, 900, 850, 800, 700];

#[derive(Debug, Clone, PartialEq)]
pub struct Level {
    /// Height above mean sea level (geopotential height for pressure levels).
    pub h_msl_m: f64,
    pub temp_c: f64,
    /// Wind *towards* east and north, m/s.
    pub wind_east: f64,
    pub wind_north: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    /// Sorted by height.
    pub levels: Vec<Level>,
    /// The hour used, UTC.
    pub time_utc: String,
    pub source: String,
}

/// A weather reply is a few KB: give up after this and fall back to still air.
const WEATHER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Kilometres per hour to metres per second.
const KMH: f64 = 1.0 / 3.6;

fn hourly_vars() -> String {
    let mut v: Vec<String> = Vec::new();
    for p in LEVELS_HPA {
        for k in [
            "geopotential_height",
            "temperature",
            "wind_speed",
            "wind_direction",
        ] {
            v.push(format!("{k}_{p}hPa"));
        }
    }
    v.extend(["temperature_2m", "wind_speed_10m", "wind_direction_10m"].map(String::from));
    v.join(",")
}

/// The request for one UTC day at (lat, lon).
pub fn url(lat: f64, lon: f64, date: &str) -> String {
    format!(
        "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude={lat:.2}&longitude={lon:.2}&start_date={date}&end_date={date}&hourly={}&wind_speed_unit=kmh&timezone=GMT",
        hourly_vars()
    )
}

/// Meteorological convention: `from_deg` is where the wind blows *from*.
fn towards(speed_kmh: f64, from_deg: f64) -> (f64, f64) {
    let (s, d) = (speed_kmh * KMH, from_deg.to_radians());
    (-s * d.sin(), -s * d.cos())
}

/// The profile for the hour in `json` nearest `hour`. Levels with a missing value, or below the
/// model surface, are dropped; fewer than 2 levels is an error.
pub fn parse_openmeteo(json: &[u8], hour: Epoch) -> Result<Profile, SourceError> {
    let bad = |m: String| SourceError::Invalid(format!("Open-Meteo: {m}"));
    let v: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| bad(format!("not JSON ({e})")))?;
    if v["error"].as_bool() == Some(true) {
        return Err(bad(v["reason"].as_str().unwrap_or("error").to_string()));
    }
    let h = &v["hourly"];
    let times = h["time"]
        .as_array()
        .ok_or_else(|| bad("no hourly times".into()))?;
    let (idx, t) = times
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let e = Epoch::from_rfc3339(&format!("{}:00Z", t.as_str()?))?;
            Some((i, e))
        })
        .min_by(|a, b| {
            a.1.seconds_since(hour)
                .abs()
                .total_cmp(&b.1.seconds_since(hour).abs())
        })
        .ok_or_else(|| bad("no usable hourly times".into()))?;
    let at = |k: &str| {
        h[k].get(idx)
            .and_then(serde_json::Value::as_f64)
            .filter(|x| x.is_finite())
    };
    let surface = v["elevation"].as_f64().unwrap_or(0.0);
    let mut levels = Vec::new();
    if let (Some(tc), Some(sp), Some(dir)) = (
        at("temperature_2m"),
        at("wind_speed_10m"),
        at("wind_direction_10m"),
    ) {
        let (e, n) = towards(sp, dir);
        levels.push(Level {
            h_msl_m: surface + 10.0,
            temp_c: tc,
            wind_east: e,
            wind_north: n,
        });
    }
    for p in LEVELS_HPA {
        let get = |k: &str| at(&format!("{k}_{p}hPa"));
        if let (Some(z), Some(tc), Some(sp), Some(dir)) = (
            get("geopotential_height"),
            get("temperature"),
            get("wind_speed"),
            get("wind_direction"),
        ) && z > surface + 10.0
        {
            let (e, n) = towards(sp, dir);
            levels.push(Level {
                h_msl_m: z,
                temp_c: tc,
                wind_east: e,
                wind_north: n,
            });
        }
    }
    levels.sort_by(|a, b| a.h_msl_m.total_cmp(&b.h_msl_m));
    levels.dedup_by(|b, a| b.h_msl_m <= a.h_msl_m);
    if levels.len() < 2 {
        return Err(bad(format!(
            "only {} usable levels at {}",
            levels.len(),
            t.to_rfc3339()
        )));
    }
    Ok(Profile {
        levels,
        time_utc: t.to_rfc3339(),
        source: "Open-Meteo historical forecast (pressure levels), CC BY 4.0".into(),
    })
}

fn cache_path(dir: &Path, lat: f64, lon: f64, date: &str) -> PathBuf {
    dir.join("weather/openmeteo")
        .join(format!("{lat:.2}_{lon:.2}_{date}.json"))
}

/// The profile near (lat, lon) for `hour`, from `dir`'s cache or fetched with `fetch(url, dest)`.
/// A cached reply that doesn't parse is removed so the next run fetches again.
pub fn profile_in(
    dir: &Path,
    lat: f64,
    lon: f64,
    hour: Epoch,
    fetch: &dyn Fn(&str, &Path) -> Result<(), SourceError>,
) -> Result<Profile, SourceError> {
    let date = hour.to_rfc3339().get(..10).unwrap_or_default().to_string();
    let dest = cache_path(dir, lat, lon, &date);
    if !dest.exists() {
        fetch(&url(lat, lon, &date), &dest)?;
    }
    let io = |source| SourceError::Io {
        path: dest.display().to_string(),
        source,
    };
    let body = std::fs::read(&dest).map_err(io)?;
    if let Some(name) = dest.file_name().and_then(|n| n.to_str()) {
        crate::paths::touch_used_in(dir, "weather", name);
    }
    parse_openmeteo(&body, hour).inspect_err(|_| {
        let _ = std::fs::remove_file(&dest);
    })
}

/// `profile_in` the user cache, fetching over HTTP.
pub fn profile(lat: f64, lon: f64, hour: Epoch) -> Result<Profile, SourceError> {
    profile_in(&crate::cache_dir(), lat, lon, hour, &|url, dest| {
        crate::cache::fetch(
            &crate::cache::Fetch {
                url: url.to_string(),
                dest: dest.to_path_buf(),
                expected_len: None,
                timeout: Some(WEATHER_TIMEOUT),
            },
            &mut |_, _| {},
        )
        .map(|_| ())
    })
}

/// The column for `p`: ellipsoidal heights (MSL + geoid `geoid_n_m`), sound speed from
/// temperature, wind as given.
pub fn column(p: &Profile, frame: LocalFrame, geoid_n_m: f64) -> Option<Column> {
    let nodes = p
        .levels
        .iter()
        .map(|l| ColumnNode {
            h_m: l.h_msl_m + geoid_n_m,
            c: Column::sound_speed(l.temp_c),
            wind_east: l.wind_east,
            wind_north: l.wind_north,
        })
        .collect();
    Column::new(frame, nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/weather/openmeteo-pl.json"
        ))
        .unwrap()
    }

    fn hour() -> Epoch {
        Epoch::from_rfc3339("2026-09-08T20:17:29Z").unwrap()
    }

    #[test]
    fn parse_fixture_profile() {
        let p = parse_openmeteo(&fixture(), hour()).unwrap();
        assert_eq!(p.time_utc, "2026-09-08T20:00:00Z");
        assert_eq!(p.levels.len(), 9, "{p:?}");
        assert!(p.levels.windows(2).all(|w| w[0].h_msl_m < w[1].h_msl_m));
        let l850 = p.levels.iter().find(|l| l.h_msl_m == 1500.0).unwrap();
        assert!((l850.temp_c - 15.2).abs() < 1e-9);
        assert!((l850.wind_east - 10.0).abs() < 1e-9, "{l850:?}");
        assert!(l850.wind_north.abs() < 1e-9, "{l850:?}");
        // Surface: model elevation 12 m + 10 m wind height.
        assert_eq!(p.levels[0].h_msl_m, 22.0);
        // The 21:00 hour is used for a time nearer to it.
        let late = parse_openmeteo(
            &fixture(),
            Epoch::from_rfc3339("2026-09-08T20:45:00Z").unwrap(),
        )
        .unwrap();
        assert_eq!(late.time_utc, "2026-09-08T21:00:00Z");
    }

    #[test]
    fn missing_levels_dropped() {
        let mut v: serde_json::Value = serde_json::from_slice(&fixture()).unwrap();
        v["hourly"]["temperature_700hPa"][0] = serde_json::Value::Null;
        let p = parse_openmeteo(&serde_json::to_vec(&v).unwrap(), hour()).unwrap();
        assert_eq!(p.levels.len(), 8);
        for (k, x) in v["hourly"].as_object_mut().unwrap() {
            if k != "time" {
                x[0] = serde_json::Value::Null;
            }
        }
        assert!(parse_openmeteo(&serde_json::to_vec(&v).unwrap(), hour()).is_err());
        let api_error = br#"{"error":true,"reason":"Parameter 'hourly' is invalid"}"#;
        let e = parse_openmeteo(api_error, hour()).unwrap_err().to_string();
        assert!(e.contains("invalid"), "{e}");
    }

    #[test]
    fn cache_hit_skips_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let body = fixture();
        let fetched = std::cell::Cell::new(0);
        let fetch = |_: &str, dest: &Path| {
            fetched.set(fetched.get() + 1);
            std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
            std::fs::write(dest, &body).unwrap();
            Ok(())
        };
        let a = profile_in(dir.path(), 40.8765, -73.525, hour(), &fetch).unwrap();
        let b = profile_in(dir.path(), 40.8765, -73.525, hour(), &|_, _| {
            panic!("refetched")
        })
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(fetched.get(), 1);
        let failing = |_: &str, _: &Path| -> Result<(), SourceError> {
            Err(SourceError::Http {
                url: "u".into(),
                reason: "offline".into(),
            })
        };
        assert!(profile_in(dir.path(), 10.0, 10.0, hour(), &failing).is_err());
    }

    #[test]
    fn column_from_profile() {
        let p = parse_openmeteo(&fixture(), hour()).unwrap();
        let frame = LocalFrame::new(gm_core::Geodetic {
            lat_deg: 40.88,
            lon_deg: -73.52,
            h_m: 0.0,
        });
        let col = column(&p, frame, -32.5).unwrap();
        let n = col.nodes();
        assert_eq!(n.len(), 9);
        assert!((n[0].h_m - (22.0 - 32.5)).abs() < 1e-9);
        let c850 = n
            .iter()
            .find(|x| (x.h_m - (1500.0 - 32.5)).abs() < 1e-9)
            .unwrap();
        assert!((c850.c - Column::sound_speed(15.2)).abs() < 1e-9);
        assert!((c850.wind_east - 10.0).abs() < 1e-9);
    }
}
