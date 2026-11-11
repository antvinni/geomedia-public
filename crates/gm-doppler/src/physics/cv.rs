//! Constant-velocity closed form: a straight, level pass in still air, plus the linear
//! least-squares rest frequency f₀ for fixed geometry.

use gm_core::{Geodetic, LocalFrame, Vec3};

use super::track::{Knot, Track};

/// f/f₀ for a straight pass at speed `v`, CPA slant `d`, CPA *arrival* time `tc`, sound speed `c`.
///
/// With τc = tc − d/c (the CPA emission time):
/// f/f₀ = c²/(c²−v²) · [1 − v²(t−τc) / √(d²(c²−v²) + c²v²(t−τc)²)]
pub fn shape(t: f64, v: f64, d: f64, tc: f64, c: f64) -> f64 {
    let (c2, v2) = (c * c, v * v);
    let x = t - (tc - d / c);
    c2 / (c2 - v2) * (1.0 - v2 * x / (d * d * (c2 - v2) + c2 * v2 * x * x).sqrt())
}

/// Least-squares f₀ for fixed (v, d, tc): f₀ = Σ fᵢ sᵢ / Σ sᵢ². None if degenerate or non-finite.
pub fn best_f0(obs: &[(f64, f64)], v: f64, d: f64, tc: f64, c: f64) -> Option<f64> {
    let (mut num, mut den) = (0.0, 0.0);
    for &(t, f) in obs {
        let s = shape(t, v, d, tc, c);
        if !(s.is_finite() && f.is_finite()) {
            return None;
        }
        num += f * s;
        den += s * s;
    }
    let f0 = num / den;
    (den > 0.0 && f0.is_finite()).then_some(f0)
}

/// A straight level pass defined in the tangent plane of `frame_origin` (the observer).
#[derive(Clone, Debug)]
pub struct StraightLine {
    pub frame_origin: Geodetic,
    /// Degrees clockwise from north.
    pub heading_deg: f64,
    pub speed_mps: f64,
    /// Height above the observer, along the observer's up.
    pub alt_m: f64,
    /// Horizontal CPA distance; positive puts the observer on the aircraft's right.
    pub cpa_offset_m: f64,
    /// Time (relative seconds) of the geometric CPA.
    pub cpa_time_s: f64,
    pub half_span_s: f64,
    pub dt_s: f64,
}

impl StraightLine {
    pub fn track(&self, id: &str) -> Track {
        let f = LocalFrame::new(self.frame_origin);
        let h = self.heading_deg.to_radians();
        let dir = Vec3::new(h.sin(), h.cos(), 0.0);
        let right = Vec3::new(h.cos(), -h.sin(), 0.0);
        let at_cpa = right * -self.cpa_offset_m + Vec3::new(0.0, 0.0, self.alt_m);
        let vel = f.vec_from_enu(dir * self.speed_mps);
        let n = (2.0 * self.half_span_s / self.dt_s).round().max(1.0) as usize;
        let knots = (0..=n)
            .map(|i| {
                let rel = -self.half_span_s + i as f64 * self.dt_s;
                Knot {
                    t: self.cpa_time_s + rel,
                    pos: f.from_enu(at_cpa + dir * (self.speed_mps * rel)),
                    vel,
                }
            })
            .collect();
        Track {
            id: id.to_string(),
            knots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{Medium, cpa, received_frequencies};

    fn line() -> StraightLine {
        StraightLine {
            frame_origin: Geodetic {
                lat_deg: 40.8765,
                lon_deg: -73.525,
                h_m: 10.0,
            },
            heading_deg: 63.0,
            speed_mps: 135.0,
            alt_m: 1700.0,
            cpa_offset_m: 1200.0,
            cpa_time_s: 50.0,
            half_span_s: 150.0,
            dt_s: 1.0,
        }
    }

    #[test]
    fn straight_line_geometry() {
        let l = line();
        let tr = l.track("pc12");
        assert_eq!(tr.id, "pc12");
        let f = LocalFrame::new(l.frame_origin);
        let (p, v) = tr.state_at(l.cpa_time_s).unwrap();
        let enu = f.to_enu(p);
        assert!(
            (enu.z - 1700.0).abs() < 1e-6
                && ((enu.x.powi(2) + enu.y.powi(2)).sqrt() - 1200.0).abs() < 1e-6
        );
        let ve = f.vec_to_enu(v);
        assert!((ve.x - 135.0 * 63f64.to_radians().sin()).abs() < 1e-9 && ve.z.abs() < 1e-9);
        // Observer on the right: (velocity × to-observer) points down.
        assert!(ve.cross(-enu).z < 0.0);
    }

    #[test]
    fn cv_matches_newton() {
        let l = line();
        let tr = l.track("a");
        let obs = LocalFrame::new(l.frame_origin).from_enu(Vec3::ZERO);
        let m = Medium::still(343.0).unwrap();
        let c = cpa(&tr, obs, m).unwrap();
        let times: Vec<f64> = (0..1200)
            .map(|i| c.t_arrive - 60.0 + f64::from(i) * 0.1)
            .collect();
        let f0 = 91.0;
        for (t, f) in times
            .iter()
            .zip(received_frequencies(&tr, obs, &times, f0, m))
        {
            let f = f.unwrap();
            let want = f0 * shape(*t, l.speed_mps, c.slant_m, c.t_arrive, 343.0);
            assert!(((f - want) / f).abs() < 1e-6, "t={t} {f} {want}");
        }
    }

    #[test]
    fn shape_at_tc_is_one() {
        assert!((shape(12.0, 120.0, 1500.0, 12.0, 343.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scale_invariance() {
        for t in [-30.0, 0.0, 4.0, 40.0] {
            let a = shape(t, 120.0, 1500.0, 2.0, 343.0);
            let b = shape(t, 240.0, 3000.0, 2.0, 686.0);
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn best_f0_recovers() {
        let (v, d, tc, c) = (120.0, 1500.0, 0.0, 343.0);
        let clean: Vec<(f64, f64)> = (0..300)
            .map(|i| {
                let t = -30.0 + f64::from(i) * 0.2;
                (t, 97.3 * shape(t, v, d, tc, c))
            })
            .collect();
        assert!((best_f0(&clean, v, d, tc, c).unwrap() - 97.3).abs() < 1e-9);
        // Deterministic N(0, 0.2 Hz) noise via a small LCG + Box–Muller.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let noisy: Vec<(f64, f64)> = clean
            .iter()
            .map(|&(t, f)| {
                let n = (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos();
                (t, f + 0.2 * n)
            })
            .collect();
        assert!((best_f0(&noisy, v, d, tc, c).unwrap() - 97.3).abs() < 0.05);
        assert!(best_f0(&[], v, d, tc, c).is_none());
        assert!(best_f0(&[(0.0, f64::NAN)], v, d, tc, c).is_none());
    }
}
