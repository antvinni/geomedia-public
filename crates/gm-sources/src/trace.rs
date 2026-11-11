//! readsb trace JSON → aircraft tracks in ECEF with velocities.
//!
//! Point layout: `[dt, lat, lon, alt_baro_ft | "ground" | null, gs_kt, track_deg, flags,
//! baro_rate_fpm, details | null, source, alt_geom_ft, geom_rate_fpm, …]`.

use gm_core::{Epoch, Vec3};
use gm_doppler::physics::Track;
use serde::Serialize;

use crate::SourceError;

/// Points further apart than this start a new track piece.
pub const MAX_GAP_S: f64 = 60.0;
/// A geom−baro offset sample is used for baro-only points within this time.
pub const OFFSET_WINDOW_S: f64 = 600.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AltBasis {
    /// GNSS height (treated as WGS84 HAE).
    Geometric,
    /// Barometric altitude + the aircraft's own interpolated geom−baro offset.
    BaroWithOffset,
    /// Barometric altitude only (weather-dependent error of tens of metres).
    BaroOnly,
}

#[derive(Debug, Clone)]
pub struct AircraftTrack {
    pub icao: String,
    pub type_code: Option<String>,
    pub registration: Option<String>,
    /// The weakest altitude basis any knot needed.
    pub alt_basis: AltBasis,
    /// Knot times are seconds relative to `origin`.
    pub tracks: Vec<Track>,
}

/// Parse one trace (gzip or plain JSON). Points outside `window` (if given) are dropped.
pub fn parse_trace(
    data: &[u8],
    origin: Epoch,
    window: Option<(Epoch, Epoch)>,
) -> Result<AircraftTrack, SourceError> {
    let json = crate::adsblol::gunzip_bounded(data)
        .ok_or_else(|| SourceError::Invalid("trace is neither gzip nor JSON".into()))?;
    let v: serde_json::Value = serde_json::from_slice(&json)
        .map_err(|e| SourceError::Invalid(format!("trace JSON: {e}")))?;
    let icao = v
        .get("icao")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let text = |k: &str| {
        v.get(k)
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let base = v
        .get("timestamp")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| SourceError::Invalid("trace has no timestamp".into()))?;
    let empty = Vec::new();
    let raw = v.get("trace").and_then(|x| x.as_array()).unwrap_or(&empty);

    struct P {
        t: f64,
        lat: f64,
        lon: f64,
        baro_ft: Option<f64>,
        geom_ft: Option<f64>,
        gs_kt: Option<f64>,
        track_deg: Option<f64>,
        rate_fpm: Option<f64>,
    }
    let num = |a: &[serde_json::Value], i: usize| {
        a.get(i)
            .and_then(serde_json::Value::as_f64)
            .filter(|x| x.is_finite())
    };
    let (w0, w1) = window.map_or((f64::NEG_INFINITY, f64::INFINITY), |(a, b)| {
        (a.seconds_since(origin), b.seconds_since(origin))
    });
    let mut pts: Vec<P> = Vec::new();
    for p in raw {
        let Some(a) = p.as_array() else { continue };
        let (Some(dt), Some(lat), Some(lon)) = (num(a, 0), num(a, 1), num(a, 2)) else {
            continue;
        };
        if lat.abs() > 90.0 || lon.abs() > 180.0 {
            continue;
        }
        // "ground" (or anything non-numeric) in the baro slot means on the ground or unknown.
        let on_ground = a.get(3).and_then(|x| x.as_str()) == Some("ground");
        if on_ground {
            continue;
        }
        let t = Epoch::from_unix_f64(base + dt).seconds_since(origin);
        // Keep ±OFFSET_WINDOW_S beyond the window so baro points can borrow geom−baro offsets.
        if t < w0 - OFFSET_WINDOW_S || t > w1 + OFFSET_WINDOW_S {
            continue;
        }
        pts.push(P {
            t,
            lat,
            lon,
            baro_ft: num(a, 3),
            geom_ft: num(a, 10),
            gs_kt: num(a, 4),
            track_deg: num(a, 5),
            rate_fpm: num(a, 11).or_else(|| num(a, 7)),
        });
    }
    pts.sort_by(|a, b| a.t.total_cmp(&b.t));
    pts.dedup_by(|b, a| b.t <= a.t);
    // geom − baro offsets, for baro-only points.
    let offsets: Vec<(f64, f64)> = pts
        .iter()
        .filter_map(|p| Some((p.t, p.geom_ft? - p.baro_ft?)))
        .collect();
    let offset_at = |t: f64| -> Option<f64> {
        let near: Vec<&(f64, f64)> = offsets
            .iter()
            .filter(|(ot, _)| (ot - t).abs() <= OFFSET_WINDOW_S)
            .collect();
        let before = near
            .iter()
            .filter(|(ot, _)| *ot <= t)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let after = near
            .iter()
            .filter(|(ot, _)| *ot >= t)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match (before, after) {
            (Some(a), Some(b)) if b.0 > a.0 => Some(a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0)),
            (Some(a), _) => Some(a.1),
            (_, Some(b)) => Some(b.1),
            _ => None,
        }
    };
    let mut basis = AltBasis::Geometric;
    let mut knots = Vec::new();
    for p in pts.iter().filter(|p| p.t >= w0 && p.t <= w1) {
        let h_ft = match (p.geom_ft, p.baro_ft) {
            (Some(g), _) => g,
            (None, Some(b)) => match offset_at(p.t) {
                Some(o) => {
                    if basis == AltBasis::Geometric {
                        basis = AltBasis::BaroWithOffset;
                    }
                    b + o
                }
                None => {
                    basis = AltBasis::BaroOnly;
                    b
                }
            },
            (None, None) => continue,
        };
        let g = gm_core::Geodetic {
            lat_deg: p.lat,
            lon_deg: p.lon,
            h_m: h_ft * FT,
        };
        let pos = gm_core::geodetic_to_ecef(g);
        let (gs, trk) = (
            p.gs_kt.unwrap_or(0.0) * KT,
            p.track_deg.unwrap_or(0.0).to_radians(),
        );
        let enu = Vec3::new(
            gs * trk.sin(),
            gs * trk.cos(),
            p.rate_fpm.unwrap_or(0.0) * FT / 60.0,
        );
        let (la, lo) = (p.lat.to_radians(), p.lon.to_radians());
        let east = Vec3::new(-lo.sin(), lo.cos(), 0.0);
        let north = Vec3::new(-la.sin() * lo.cos(), -la.sin() * lo.sin(), la.cos());
        let up = Vec3::new(la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin());
        let vel = east * enu.x + north * enu.y + up * enu.z;
        knots.push(gm_doppler::physics::Knot { t: p.t, pos, vel });
    }
    let mut tracks = Vec::new();
    let mut piece: Vec<gm_doppler::physics::Knot> = Vec::new();
    let mut push = |mut piece: Vec<gm_doppler::physics::Knot>| {
        smooth_heights(&mut piece);
        if let Ok(t) = Track::new(icao.clone(), piece) {
            tracks.push(t);
        }
    };
    for k in knots {
        if piece.last().is_some_and(|l| k.t - l.t > MAX_GAP_S) {
            push(std::mem::take(&mut piece));
        }
        piece.push(k);
    }
    push(piece);
    Ok(AircraftTrack {
        icao,
        type_code: text("t"),
        registration: text("r"),
        alt_basis: basis,
        tracks,
    })
}

/// Gaussian σ (s) for smoothing ADS-B altitude, which is reported in 25 ft steps and
/// ping-pongs between neighbouring steps in level flight.
pub const ALT_SMOOTH_SIGMA_S: f64 = 8.0;

/// Replace each knot's ellipsoidal height by a Gaussian-weighted mean over ±3σ, the window
/// shrunk symmetrically near the piece ends so linear climbs pass through exactly. Latitude,
/// longitude, times and velocities are kept.
fn smooth_heights(knots: &mut [gm_doppler::physics::Knot]) {
    if knots.len() < 3 {
        return;
    }
    let geo: Vec<gm_core::Geodetic> = knots
        .iter()
        .map(|k| gm_core::ecef_to_geodetic(k.pos))
        .collect();
    let (t0, t1) = (knots[0].t, knots[knots.len() - 1].t);
    // Knots are time-sorted: each window is a contiguous slice found by binary search.
    let times: Vec<f64> = knots.iter().map(|k| k.t).collect();
    let smoothed: Vec<f64> = times
        .iter()
        .map(|&t| {
            let half = (3.0 * ALT_SMOOTH_SIGMA_S).min(t - t0).min(t1 - t) + 1e-9;
            let lo = times.partition_point(|&x| x < t - half);
            let hi = times.partition_point(|&x| x <= t + half);
            let (mut sw, mut sh) = (0.0, 0.0);
            for (x, g) in times[lo..hi].iter().zip(&geo[lo..hi]) {
                let w = (-0.5 * ((x - t) / ALT_SMOOTH_SIGMA_S).powi(2)).exp();
                sw += w;
                sh += w * g.h_m;
            }
            sh / sw
        })
        .collect();
    for ((k, g), h) in knots.iter_mut().zip(&geo).zip(smoothed) {
        k.pos = gm_core::geodetic_to_ecef(gm_core::Geodetic { h_m: h, ..*g });
    }
}

const FT: f64 = 0.3048;
const KT: f64 = 0.514_444;

#[cfg(test)]
mod tests {
    use super::*;
    use gm_core::{LocalFrame, ecef_to_geodetic};

    const BASE: f64 = 1_788_825_600.0; // 2026-09-08T00:00:00Z

    fn point(dt: f64, lat: f64, lon: f64, baro: &str, geom: &str) -> String {
        format!("[{dt},{lat},{lon},{baro},120.0,90.0,0,0,null,\"adsb_icao\",{geom},0,null,null]")
    }

    fn trace(points: &[String]) -> Vec<u8> {
        format!(
            r#"{{"icao":"a1dfeb","r":"N22NY","t":"PC12","timestamp":{BASE},"trace":[{}]}}"#,
            points.join(",")
        )
        .into_bytes()
    }

    fn origin() -> Epoch {
        Epoch::from_unix_f64(BASE + 75_000.0)
    }

    #[test]
    fn parses_fixture_trace() {
        // Heading east at 120 kt, 5000 ft geometric, one point per 5 s.
        let pts: Vec<String> = (0..30)
            .map(|i| {
                point(
                    75_000.0 + 5.0 * f64::from(i),
                    40.8,
                    -73.6 + 0.0003 * f64::from(i),
                    "4800",
                    "5000",
                )
            })
            .collect();
        let a = parse_trace(&trace(&pts), origin(), None).unwrap();
        assert_eq!(a.icao, "a1dfeb");
        assert_eq!(a.type_code.as_deref(), Some("PC12"));
        assert_eq!(a.registration.as_deref(), Some("N22NY"));
        assert_eq!(a.alt_basis, AltBasis::Geometric);
        assert_eq!(a.tracks.len(), 1);
        let tr = &a.tracks[0];
        assert_eq!(tr.knots.len(), 30);
        assert!((tr.knots[0].t - 0.0).abs() < 1e-9 && (tr.knots[29].t - 145.0).abs() < 1e-9);
        let g = ecef_to_geodetic(tr.knots[0].pos);
        assert!((g.h_m - 5000.0 * 0.3048).abs() < 1e-6 && (g.lat_deg - 40.8).abs() < 1e-9);
        // Velocity: 120 kt due east.
        let f = LocalFrame::new(g);
        let v = f.vec_to_enu(tr.knots[0].vel);
        assert!(
            (v.x - 120.0 * 0.514_444).abs() < 1e-3 && v.y.abs() < 1e-3 && v.z.abs() < 1e-6,
            "{v:?}"
        );
    }

    fn heights_ft(a: &AircraftTrack) -> Vec<f64> {
        a.tracks[0]
            .knots
            .iter()
            .map(|k| ecef_to_geodetic(k.pos).h_m / FT)
            .collect()
    }

    #[test]
    fn smoothing_removes_quantisation() {
        // Level flight reported at 25 ft resolution, ping-ponging between 3225 and 3250 ft.
        let pts: Vec<String> = (0..120)
            .map(|i| {
                let alt = if i % 2 == 0 { "3225" } else { "3250" };
                point(
                    75_000.0 + f64::from(i),
                    40.8,
                    -73.6 + 0.0003 * f64::from(i),
                    alt,
                    alt,
                )
            })
            .collect();
        let h = heights_ft(&parse_trace(&trace(&pts), origin(), None).unwrap());
        let n = h.len();
        // Within 3σ of a piece end the window shrinks symmetrically, so smoothing is partial.
        let edge = (3.0 * ALT_SMOOTH_SIGMA_S) as usize;
        for (i, x) in h.iter().enumerate() {
            let tol = if i < edge || i + edge >= n { 12.6 } else { 2.0 };
            assert!((x - 3237.5).abs() <= tol, "knot {i}: {x}");
        }
    }

    #[test]
    fn smoothing_preserves_climb() {
        let pts: Vec<String> = (0..90)
            .map(|i| {
                let alt = format!("{}", 3000.0 + 10.0 * f64::from(i));
                point(75_000.0 + f64::from(i), 40.8, -73.6, &alt, &alt)
            })
            .collect();
        let h = heights_ft(&parse_trace(&trace(&pts), origin(), None).unwrap());
        for (i, x) in h.iter().enumerate() {
            assert!(
                (x - (3000.0 + 10.0 * i as f64)).abs() < 0.01,
                "knot {i}: {x}"
            );
        }
    }

    #[test]
    fn smoothing_keeps_short_pieces() {
        for dts in [vec![0.0], vec![0.0, 7.0], vec![0.0, 3.0, 11.0]] {
            let pts: Vec<String> = dts
                .iter()
                .enumerate()
                .map(|(i, dt)| {
                    let alt = format!("{}", 3000 + 25 * i);
                    point(75_000.0 + dt, 40.8 + 0.001 * i as f64, -73.6, &alt, &alt)
                })
                .collect();
            let a = parse_trace(&trace(&pts), origin(), None).unwrap();
            let ks: Vec<_> = a.tracks.iter().flat_map(|t| t.knots.iter()).collect();
            // A single knot never makes a track (Track needs two); longer pieces keep every knot.
            assert_eq!(ks.len(), if dts.len() < 2 { 0 } else { dts.len() });
            for (i, (k, dt)) in ks.iter().zip(&dts).enumerate() {
                assert!((k.t - dt).abs() < 1e-9);
                let g = ecef_to_geodetic(k.pos);
                assert!(
                    (g.lat_deg - (40.8 + 0.001 * i as f64)).abs() < 1e-9,
                    "{g:?}"
                );
                assert!((g.lon_deg + 73.6).abs() < 1e-9, "{g:?}");
            }
        }
    }

    #[test]
    fn geometric_preferred() {
        let pts = vec![
            point(75_000.0, 40.8, -73.6, "4800", "5000"),
            point(75_005.0, 40.8, -73.599, "4800", "5000"),
        ];
        let a = parse_trace(&trace(&pts), origin(), None).unwrap();
        assert!((ecef_to_geodetic(a.tracks[0].knots[1].pos).h_m - 1524.0).abs() < 1e-6);
    }

    #[test]
    fn baro_offset_interpolated() {
        // geom−baro = +200 ft from the first points; later points have baro only.
        let mut pts = vec![
            point(75_000.0, 40.8, -73.6, "4800", "5000"),
            point(75_005.0, 40.8, -73.599, "4800", "5000"),
        ];
        pts.extend((2..6).map(|i| {
            point(
                75_000.0 + 5.0 * f64::from(i),
                40.8,
                -73.6 + 0.001 * f64::from(i),
                "4900",
                "null",
            )
        }));
        let a = parse_trace(&trace(&pts), origin(), None).unwrap();
        assert_eq!(a.alt_basis, AltBasis::BaroWithOffset);
        let h = ecef_to_geodetic(a.tracks[0].knots[4].pos).h_m;
        assert!((h - 5100.0 * 0.3048).abs() < 1e-6, "{h}");
        let only: Vec<String> = (0..4)
            .map(|i| {
                point(
                    75_000.0 + 5.0 * f64::from(i),
                    40.8,
                    -73.6 + 0.001 * f64::from(i),
                    "4900",
                    "null",
                )
            })
            .collect();
        assert_eq!(
            parse_trace(&trace(&only), origin(), None)
                .unwrap()
                .alt_basis,
            AltBasis::BaroOnly
        );
    }

    #[test]
    fn ground_points_dropped() {
        let mut pts = vec![point(75_000.0, 40.8, -73.6, "\"ground\"", "null")];
        pts.extend((1..5).map(|i| {
            point(
                75_000.0 + 5.0 * f64::from(i),
                40.8,
                -73.6 + 0.001 * f64::from(i),
                "4800",
                "5000",
            )
        }));
        let a = parse_trace(&trace(&pts), origin(), None).unwrap();
        assert_eq!(a.tracks[0].knots.len(), 4);
    }

    #[test]
    fn gap_split_60s() {
        let mut pts: Vec<String> = (0..4)
            .map(|i| {
                point(
                    75_000.0 + 5.0 * f64::from(i),
                    40.8,
                    -73.6 + 0.001 * f64::from(i),
                    "4800",
                    "5000",
                )
            })
            .collect();
        pts.extend((0..4).map(|i| {
            point(
                75_200.0 + 5.0 * f64::from(i),
                40.8,
                -73.5 + 0.001 * f64::from(i),
                "4800",
                "5000",
            )
        }));
        let a = parse_trace(&trace(&pts), origin(), None).unwrap();
        assert_eq!(a.tracks.len(), 2);
        let w = Some((
            Epoch::from_unix_f64(BASE + 75_190.0),
            Epoch::from_unix_f64(BASE + 75_300.0),
        ));
        assert_eq!(
            parse_trace(&trace(&pts), origin(), w).unwrap().tracks.len(),
            1
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_trace(b"not json", origin(), None).is_err());
        assert!(
            parse_trace(
                br#"{"icao":"x","timestamp":1,"trace":[[1,"a"]]}"#,
                origin(),
                None
            )
            .unwrap()
            .tracks
            .is_empty()
        );
    }

    #[test]
    fn antimeridian_bbox() {
        assert_eq!(
            crate::adsblol::lon_extent(&[179.5, 179.9, -179.8, -179.2]),
            (179.5, -179.2)
        );
        assert_eq!(crate::adsblol::lon_extent(&[-73.6, -73.4]), (-73.6, -73.4));
    }
}
