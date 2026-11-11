//! `--area` and `--when`: what the analyst already knows about where and when, as precisely as
//! they can say it. Nothing here is inferred silently.

use gm_core::{Epoch, Geodetic, geodesic_distance_m};
use serde::{Deserialize, Serialize};

use crate::SourceError;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Area {
    Circle {
        lat: f64,
        lon: f64,
        radius_m: f64,
    },
    BBox {
        south: f64,
        west: f64,
        north: f64,
        east: f64,
    },
    /// (lat, lon) ring.
    Polygon(Vec<(f64, f64)>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Geocoded {
    pub display_name: String,
    pub lat: f64,
    pub lon: f64,
    /// (south, west, north, east)
    pub bbox: (f64, f64, f64, f64),
}

const M_PER_DEG: f64 = 111_320.0;
const R_EARTH: f64 = 6_371_008.8;

impl Area {
    /// Parse `lat,lon~5km`, `S,W,N,E`, `*.geojson`, `country:GB` or a place name (via `geocode`).
    pub fn parse(
        s: &str,
        geocode: &mut dyn FnMut(&str) -> Result<Geocoded, SourceError>,
    ) -> Result<Area, SourceError> {
        let s = s.trim();
        let bad = |why: &str| SourceError::Invalid(format!("--area {s:?}: {why}"));
        let lat_ok = |v: f64| v.is_finite() && v.abs() <= 90.0;
        let lon_ok = |v: f64| v.is_finite() && v.abs() <= 180.0;
        if let Some((pt, r)) = s.split_once('~') {
            let (lat, lon) = pt
                .split_once(',')
                .ok_or_else(|| bad("expected lat,lon~radius"))?;
            let (lat, lon): (f64, f64) = (
                lat.trim().parse().map_err(|_| bad("bad latitude"))?,
                lon.trim().parse().map_err(|_| bad("bad longitude"))?,
            );
            let r = r.trim().to_ascii_lowercase();
            let radius_m = if let Some(km) = r.strip_suffix("km") {
                km.trim().parse::<f64>().map(|v| v * 1000.0)
            } else if let Some(m) = r.strip_suffix('m') {
                m.trim().parse::<f64>()
            } else {
                r.parse::<f64>()
            }
            .map_err(|_| bad("radius must look like 5km or 800m"))?;
            if !(lat_ok(lat) && lon_ok(lon)) {
                return Err(bad("coordinates out of range"));
            }
            if !(radius_m.is_finite() && radius_m > 0.0 && radius_m <= 2.0e6) {
                return Err(bad("radius must be between 0 and 2000 km"));
            }
            return Ok(Area::Circle { lat, lon, radius_m });
        }
        let nums: Vec<Option<f64>> = s.split(',').map(|x| x.trim().parse::<f64>().ok()).collect();
        if nums.len() == 4 && nums.iter().all(Option::is_some) {
            let v: Vec<f64> = nums.into_iter().flatten().collect();
            let (south, west, north, east) = (v[0], v[1], v[2], v[3]);
            if !(lat_ok(south) && lat_ok(north) && lon_ok(west) && lon_ok(east)) || south >= north {
                return Err(bad("expected S,W,N,E with S < N"));
            }
            return Ok(Area::BBox {
                south,
                west,
                north,
                east,
            });
        }
        let lower = s.to_ascii_lowercase();
        if lower.ends_with(".geojson")
            || (lower.ends_with(".json") && std::path::Path::new(s).exists())
        {
            let text = std::fs::read_to_string(s).map_err(|e| bad(&e.to_string()))?;
            return polygon_from_geojson(&text)
                .map(Area::Polygon)
                .ok_or_else(|| bad("no Polygon/MultiPolygon in the GeoJSON"));
        }
        let g = geocode(s)?;
        if lower.starts_with("country:") {
            let (south, west, north, east) = g.bbox;
            return Ok(Area::BBox {
                south,
                west,
                north,
                east,
            });
        }
        let (south, west, north, east) = g.bbox;
        let half_diag = 0.5
            * geodesic_distance_m(
                Geodetic {
                    lat_deg: south,
                    lon_deg: west,
                    h_m: 0.0,
                },
                Geodetic {
                    lat_deg: north,
                    lon_deg: east,
                    h_m: 0.0,
                },
            );
        Ok(Area::Circle {
            lat: g.lat,
            lon: g.lon,
            radius_m: half_diag.max(2000.0),
        })
    }

    /// (south, west, north, east)
    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        match self {
            Area::Circle { lat, lon, radius_m } => {
                let dlat = radius_m / M_PER_DEG;
                let dlon = radius_m / (M_PER_DEG * lat.to_radians().cos().max(1e-6));
                (
                    (lat - dlat).max(-90.0),
                    lon - dlon,
                    (lat + dlat).min(90.0),
                    lon + dlon,
                )
            }
            Area::BBox {
                south,
                west,
                north,
                east,
            } => (*south, *west, *north, *east),
            Area::Polygon(r) => r
                .iter()
                .fold((90.0, 180.0, -90.0, -180.0), |(s, w, n, e), &(la, lo)| {
                    (s.min(la), w.min(lo), n.max(la), e.max(lo))
                }),
        }
    }

    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        match self {
            Area::Circle {
                lat: c_lat,
                lon: c_lon,
                radius_m,
            } => {
                geodesic_distance_m(
                    Geodetic {
                        lat_deg: *c_lat,
                        lon_deg: *c_lon,
                        h_m: 0.0,
                    },
                    Geodetic {
                        lat_deg: lat,
                        lon_deg: lon,
                        h_m: 0.0,
                    },
                ) <= *radius_m
            }
            Area::BBox {
                south,
                west,
                north,
                east,
            } => {
                lat >= *south
                    && lat <= *north
                    && if west <= east {
                        lon >= *west && lon <= *east
                    } else {
                        lon >= *west || lon <= *east
                    }
            }
            Area::Polygon(r) => {
                // Ray casting in (lon, lat).
                let mut inside = false;
                let n = r.len();
                for i in 0..n {
                    let (yi, xi) = r[i];
                    let (yj, xj) = r[(i + n - 1) % n];
                    if (yi > lat) != (yj > lat) && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi {
                        inside = !inside;
                    }
                }
                inside
            }
        }
    }

    pub fn area_km2(&self) -> f64 {
        match self {
            Area::Circle { radius_m, .. } => std::f64::consts::PI * radius_m * radius_m / 1e6,
            Area::BBox {
                south,
                west,
                north,
                east,
            } => {
                let dlon = if east >= west {
                    east - west
                } else {
                    east + 360.0 - west
                };
                R_EARTH
                    * R_EARTH
                    * dlon.to_radians()
                    * (north.to_radians().sin() - south.to_radians().sin())
                    / 1e6
            }
            Area::Polygon(r) => {
                let (lat0, _) = self.centre();
                let k = lat0.to_radians().cos();
                let xy: Vec<(f64, f64)> = r
                    .iter()
                    .map(|&(la, lo)| (lo.to_radians() * R_EARTH * k, la.to_radians() * R_EARTH))
                    .collect();
                let twice: f64 = (0..xy.len())
                    .map(|i| {
                        let (a, b) = (xy[i], xy[(i + 1) % xy.len()]);
                        a.0 * b.1 - b.0 * a.1
                    })
                    .sum();
                twice.abs() / 2.0 / 1e6
            }
        }
    }

    /// Whether (lat, lon) is inside the area or within `pad_m` of it (approximate for boxes/polygons).
    pub fn near(&self, lat: f64, lon: f64, pad_m: f64) -> bool {
        match self {
            Area::Circle {
                lat: c_lat,
                lon: c_lon,
                radius_m,
            } => {
                geodesic_distance_m(
                    Geodetic {
                        lat_deg: *c_lat,
                        lon_deg: *c_lon,
                        h_m: 0.0,
                    },
                    Geodetic {
                        lat_deg: lat,
                        lon_deg: lon,
                        h_m: 0.0,
                    },
                ) <= radius_m + pad_m
            }
            _ => {
                let (s, w, n, e) = self.bbox();
                let dlat = pad_m / M_PER_DEG;
                let dlon = pad_m / (M_PER_DEG * (0.5 * (s + n)).to_radians().cos().max(1e-3));
                Area::BBox {
                    south: s - dlat,
                    west: w - dlon,
                    north: n + dlat,
                    east: e + dlon,
                }
                .contains(lat, lon)
            }
        }
    }

    /// A representative centre point.
    pub fn centre(&self) -> (f64, f64) {
        match self {
            Area::Circle { lat, lon, .. } => (*lat, *lon),
            _ => {
                let (s, w, n, e) = self.bbox();
                (
                    0.5 * (s + n),
                    if w <= e {
                        0.5 * (w + e)
                    } else {
                        (0.5 * (w + e + 360.0) + 180.0) % 360.0 - 180.0
                    },
                )
            }
        }
    }
}

/// The first ring of the first Polygon/MultiPolygon, as (lat, lon).
fn polygon_from_geojson(text: &str) -> Option<Vec<(f64, f64)>> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let geom = match v.get("type")?.as_str()? {
        "FeatureCollection" => v.get("features")?.get(0)?.get("geometry")?.clone(),
        "Feature" => v.get("geometry")?.clone(),
        _ => v,
    };
    let ring = match geom.get("type")?.as_str()? {
        "Polygon" => geom.get("coordinates")?.get(0)?,
        "MultiPolygon" => geom.get("coordinates")?.get(0)?.get(0)?,
        _ => return None,
    };
    let pts: Vec<(f64, f64)> = ring
        .as_array()?
        .iter()
        .filter_map(|p| Some((p.get(1)?.as_f64()?, p.get(0)?.as_f64()?)))
        .collect();
    (pts.len() >= 4).then_some(pts)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct When {
    pub center: Epoch,
    pub minus_s: f64,
    pub plus_s: f64,
}

impl When {
    /// `2026-09-08T20:17:29Z`, `…±5s` / `±10m` / `±2h`, or `A/B`. A time without a zone is an error.
    pub fn parse(s: &str) -> Result<When, SourceError> {
        let s = s.trim();
        if let Some((a, b)) = s.split_once('/') {
            let (a, b) = (instant(a)?, instant(b)?);
            if b <= a {
                return Err(SourceError::Invalid(format!(
                    "--when {s:?}: the end must be after the start"
                )));
            }
            let half = 0.5 * b.seconds_since(a);
            return Ok(When {
                center: a.plus_seconds(half),
                minus_s: half,
                plus_s: half,
            });
        }
        let (t, pm) = match s.split_once('±').or_else(|| s.split_once("+-")) {
            Some((t, pm)) => (t, Some(pm)),
            None => (s, None),
        };
        let center = instant(t)?;
        let u = match pm {
            None => 0.0,
            Some(pm) => duration_s(pm).ok_or_else(|| {
                SourceError::Invalid(format!(
                    "--when {s:?}: uncertainty must look like ±5s, ±10m or ±2h"
                ))
            })?,
        };
        Ok(When {
            center,
            minus_s: u,
            plus_s: u,
        })
    }

    pub fn start(&self) -> Epoch {
        self.center.plus_seconds(-self.minus_s)
    }

    pub fn end(&self) -> Epoch {
        self.center.plus_seconds(self.plus_s)
    }
}

/// An instant that must carry a zone (`Z` or an offset); `HH:MM` without seconds is accepted.
fn instant(raw: &str) -> Result<Epoch, SourceError> {
    let t = raw.trim();
    if !t.is_ascii() {
        return Err(SourceError::Invalid(format!(
            "--when {t:?}: expected an RFC 3339 time like 2026-09-08T20:17:29Z"
        )));
    }
    let mut s: String = t
        .chars()
        .enumerate()
        .map(|(i, c)| if i == 10 && c == ' ' { 'T' } else { c })
        .collect();
    let has_zone =
        s.ends_with(['Z', 'z']) || s.get(16..).is_some_and(|tail| tail.contains(['+', '-']));
    if !has_zone {
        return Err(SourceError::Invalid(format!(
            "--when {t:?} has no time zone: append Z for UTC or an offset such as +01:00 (local times are never guessed)"
        )));
    }
    // Insert seconds into "YYYY-MM-DDTHH:MM<zone>".
    if s.len() >= 16
        && s.as_bytes()
            .get(16)
            .is_some_and(|b| !b.is_ascii_digit() && *b != b':' && *b != b'.')
    {
        s.insert_str(16, ":00");
    }
    Epoch::from_rfc3339(&s).ok_or_else(|| {
        SourceError::Invalid(format!(
            "--when {t:?}: expected an RFC 3339 time like 2026-09-08T20:17:29Z"
        ))
    })
}

/// "5s", "10m", "2h", "1d", or plain seconds.
fn duration_s(s: &str) -> Option<f64> {
    let s = s.trim();
    if !s.is_ascii() {
        return None;
    }
    let (num, mult) = match s.chars().last()? {
        's' => (&s[..s.len() - 1], 1.0),
        'm' => (&s[..s.len() - 1], 60.0),
        'h' => (&s[..s.len() - 1], 3600.0),
        'd' => (&s[..s.len() - 1], 86_400.0),
        _ => (s, 1.0),
    };
    let v: f64 = num.trim().parse().ok()?;
    (v.is_finite() && v >= 0.0).then_some(v * mult)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_geocode(q: &str) -> Result<Geocoded, SourceError> {
        Err(SourceError::Invalid(format!("unexpected geocode {q}")))
    }

    #[test]
    fn area_forms() {
        let c = Area::parse("40.8765,-73.525~5km", &mut no_geocode).unwrap();
        assert_eq!(
            c,
            Area::Circle {
                lat: 40.8765,
                lon: -73.525,
                radius_m: 5000.0
            }
        );
        assert!(
            c.contains(40.8765, -73.525)
                && c.contains(40.90, -73.525)
                && !c.contains(40.95, -73.525)
        );
        assert!((c.area_km2() - std::f64::consts::PI * 25.0).abs() < 0.01);
        let (s, w, n, e) = c.bbox();
        assert!(s < 40.8765 && n > 40.8765 && w < -73.525 && e > -73.525);
        assert_eq!(
            Area::parse("51.5,-0.12~800m", &mut no_geocode).unwrap(),
            Area::Circle {
                lat: 51.5,
                lon: -0.12,
                radius_m: 800.0
            }
        );
        let b = Area::parse("40.8,-73.6,40.95,-73.4", &mut no_geocode).unwrap();
        assert_eq!(
            b,
            Area::BBox {
                south: 40.8,
                west: -73.6,
                north: 40.95,
                east: -73.4
            }
        );
        assert!(b.contains(40.9, -73.5) && !b.contains(41.0, -73.5));
        assert!(Area::parse("95,0~1km", &mut no_geocode).is_err());
        assert!(Area::parse("40,-73~-5km", &mut no_geocode).is_err());
        assert!(
            Area::parse("41,-73,40,-72", &mut no_geocode).is_err(),
            "south > north"
        );
    }

    #[test]
    fn area_geojson() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("valley.geojson");
        std::fs::write(&p, r#"{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-73.6,40.8],[-73.4,40.8],[-73.4,40.95],[-73.6,40.95],[-73.6,40.8]]]}}"#).unwrap();
        let a = Area::parse(p.to_str().unwrap(), &mut no_geocode).unwrap();
        assert!(matches!(a, Area::Polygon(ref r) if r.len() == 5));
        assert!(a.contains(40.9, -73.5) && !a.contains(40.9, -73.7));
        assert!(
            a.area_km2() > 200.0 && a.area_km2() < 300.0,
            "{}",
            a.area_km2()
        );
    }

    #[test]
    fn place_name_uses_geocoder() {
        let mut asked = Vec::new();
        let mut geo = |q: &str| {
            asked.push(q.to_string());
            Ok(Geocoded {
                display_name: "Oyster Bay, Nassau County, New York, United States".into(),
                lat: 40.8657,
                lon: -73.5318,
                bbox: (40.84, -73.56, 40.89, -73.50),
            })
        };
        let a = Area::parse("Oyster Bay, NY", &mut geo).unwrap();
        let Area::Circle { lat, radius_m, .. } = a else {
            panic!("{a:?}")
        };
        assert!((lat - 40.8657).abs() < 1e-9 && radius_m >= 2000.0);
        let c = Area::parse("country:GB", &mut geo).unwrap();
        assert!(matches!(c, Area::BBox { .. }));
        assert_eq!(
            asked,
            vec!["Oyster Bay, NY".to_string(), "country:GB".to_string()]
        );
    }

    #[test]
    fn when_forms() {
        let w = When::parse("2026-09-08T20:17:29.662Z").unwrap();
        assert_eq!(w.center.to_rfc3339(), "2026-09-08T20:17:29.662Z");
        assert_eq!((w.minus_s, w.plus_s), (0.0, 0.0));
        let w = When::parse("2026-09-08T20:17:29Z±5s").unwrap();
        assert_eq!((w.minus_s, w.plus_s), (5.0, 5.0));
        assert_eq!(When::parse("2026-09-08T20:17Z±10m").unwrap().minus_s, 600.0);
        assert_eq!(
            When::parse("2026-09-08T20:17:29-04:00±2h").unwrap().plus_s,
            7200.0
        );
        let r = When::parse("2026-09-08T20:00Z/2026-09-08T21:00Z").unwrap();
        assert_eq!(r.start().to_rfc3339(), "2026-09-08T20:00:00Z");
        assert_eq!(r.end().to_rfc3339(), "2026-09-08T21:00:00Z");
        assert!(When::parse("2026-09-08T21:00Z/2026-09-08T20:00Z").is_err());
    }

    #[test]
    fn when_non_ascii_no_panic() {
        assert!(When::parse("2026-09-08T20:1€Z").is_err());
        assert!(When::parse("2026-09-08T20:17:29Z±5€").is_err());
        assert!(When::parse("€€€€€€€€€€€€€€€€€€").is_err());
    }

    #[test]
    fn when_requires_zone() {
        let e = When::parse("2026-09-08 20:17").unwrap_err().to_string();
        assert!(e.contains("Z") && e.contains("offset"), "{e}");
    }
}
