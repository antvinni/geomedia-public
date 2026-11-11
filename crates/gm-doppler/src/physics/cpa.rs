//! Closest point of approach between a track and a fixed observer.

use gm_core::{Vec3, ecef_to_geodetic};

use super::emission::Medium;
use super::track::Track;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cpa {
    /// Emission time of the geometric closest approach.
    pub t_emit: f64,
    /// When that sound reaches the observer through the moving medium.
    pub t_arrive: f64,
    pub slant_m: f64,
    /// Velocity component perpendicular to local up at the aircraft.
    pub ground_speed_mps: f64,
}

/// Geometric CPA over the whole track, with its moving-medium arrival time.
pub fn cpa(track: &Track, obs: Vec3, m: Medium) -> Option<Cpa> {
    let range = |t: f64| track.state_at(t).map(|(p, _)| (p - obs).norm());
    let rate = |t: f64| track.state_at(t).map(|(p, v)| (p - obs).dot(v));
    let mut best: Option<(f64, f64)> = None;
    let mut consider = |t: f64, r: f64| {
        if r.is_finite() && best.is_none_or(|(_, br)| r < br) {
            best = Some((t, r));
        }
    };
    for k in &track.knots {
        consider(k.t, (k.pos - obs).norm());
    }
    for w in track.knots.windows(2) {
        let (a, b) = (w[0].t, w[1].t);
        let (Some(ra), Some(rb)) = (rate(a), rate(b)) else {
            continue;
        };
        if ra < 0.0 && rb > 0.0 {
            // Bisect the range rate in segment-relative time.
            let (mut lo, mut hi) = (0.0, b - a);
            while hi - lo > 1e-9 {
                let mid = 0.5 * (lo + hi);
                match rate(a + mid) {
                    Some(r) if r < 0.0 => lo = mid,
                    Some(_) => hi = mid,
                    None => break,
                }
            }
            let t = a + 0.5 * (lo + hi);
            if let Some(r) = range(t) {
                consider(t, r);
            }
        }
    }
    let (t_emit, slant_m) = best?;
    let (p, v) = track.state_at(t_emit)?;
    let q = obs - p;
    let w = m.wind;
    // Positive root of |q − w s| = c s: (c² − |w|²) s² + 2 (q·w) s − |q|² = 0.
    let a = m.c * m.c - w.dot(w);
    let qw = q.dot(w);
    let s = (-qw + (qw * qw + a * q.dot(q)).sqrt()) / a;
    let g = ecef_to_geodetic(p);
    let (lat, lon) = (g.lat_deg.to_radians(), g.lon_deg.to_radians());
    let up = Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
    let horiz = v - up * v.dot(up);
    Some(Cpa {
        t_emit,
        t_arrive: t_emit + s,
        slant_m,
        ground_speed_mps: horiz.norm(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::emission::tests::{frame, straight};
    use crate::physics::track::Knot;

    /// Arrival delay s solving |q − w s| = c s by bisection (independent of the closed form).
    fn delay(q: Vec3, w: Vec3, c: f64) -> f64 {
        let (mut lo, mut hi) = (0.0, 1e4);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if (q - w * mid).norm() > c * mid {
                lo = mid
            } else {
                hi = mid
            }
        }
        0.5 * (lo + hi)
    }

    #[test]
    fn cpa_on_straight_track() {
        let f = frame();
        let obs = f.from_enu(Vec3::ZERO);
        let tr = straight(&f, 120.0, 1500.0, 800.0, 100.0, 0.0);
        for wind in [Vec3::ZERO, f.vec_from_enu(Vec3::new(9.0, -4.0, 0.0))] {
            let m = Medium::new(343.0, wind).unwrap();
            let c = cpa(&tr, obs, m).unwrap();
            let p = tr.state_at(0.0).unwrap().0;
            assert!(c.t_emit.abs() < 1e-6, "{c:?}");
            assert!(
                (c.slant_m - (800f64.powi(2) + 1500f64.powi(2)).sqrt()).abs() < 1e-3,
                "{c:?}"
            );
            assert!((c.ground_speed_mps - 120.0).abs() < 1e-3, "{c:?}");
            assert!(
                (c.t_arrive - (c.t_emit + delay(obs - p, wind, 343.0))).abs() < 1e-6,
                "{c:?}"
            );
        }
    }

    #[test]
    fn cpa_at_track_end() {
        let f = frame();
        let obs = f.from_enu(Vec3::ZERO);
        let full = straight(&f, 120.0, 1500.0, 800.0, 100.0, 0.0);
        let approaching = Track::new(
            "a",
            full.knots
                .iter()
                .copied()
                .filter(|k| k.t <= -10.0)
                .collect(),
        )
        .unwrap();
        let c = cpa(&approaching, obs, Medium::still(343.0).unwrap()).unwrap();
        assert_eq!(c.t_emit, -10.0);
    }

    #[test]
    fn stationary_source() {
        let f = frame();
        let p = f.from_enu(Vec3::new(300.0, 400.0, 100.0));
        let knots = (0..10)
            .map(|i| Knot {
                t: f64::from(i),
                pos: p,
                vel: Vec3::ZERO,
            })
            .collect();
        let tr = Track::new("hover", knots).unwrap();
        let c = cpa(&tr, f.from_enu(Vec3::ZERO), Medium::still(343.0).unwrap()).unwrap();
        assert!(c.t_emit.is_finite() && c.t_arrive.is_finite() && c.ground_speed_mps == 0.0);
        assert!(
            (c.slant_m - (300f64.powi(2) + 400f64.powi(2) + 100f64.powi(2)).sqrt()).abs() < 1e-6
        );
    }
}
