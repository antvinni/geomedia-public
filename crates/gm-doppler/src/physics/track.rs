//! Aircraft tracks as timed ECEF knots with velocities, interpolated by cubic Hermite.

use gm_core::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Knot {
    /// Seconds relative to the run epoch.
    pub t: f64,
    pub pos: Vec3,
    pub vel: Vec3,
}

/// A validated track. Deserialising goes through [`Track::new`], so files can't bypass validation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawTrack")]
pub struct Track {
    pub id: String,
    pub knots: Vec<Knot>,
}

#[derive(Deserialize)]
struct RawTrack {
    id: String,
    knots: Vec<Knot>,
}

impl TryFrom<RawTrack> for Track {
    type Error = TrackError;
    fn try_from(r: RawTrack) -> Result<Track, TrackError> {
        Track::new(r.id, r.knots)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackError {
    TooFewKnots,
    NotIncreasing { index: usize },
    NonFinite { index: usize },
}

impl std::fmt::Display for TrackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackError::TooFewKnots => write!(f, "a track needs at least two knots"),
            TrackError::NotIncreasing { index } => {
                write!(f, "knot {index} is not later than the one before it")
            }
            TrackError::NonFinite { index } => write!(f, "knot {index} has a non-finite value"),
        }
    }
}

impl std::error::Error for TrackError {}

impl Track {
    pub fn new(id: impl Into<String>, knots: Vec<Knot>) -> Result<Track, TrackError> {
        if knots.len() < 2 {
            return Err(TrackError::TooFewKnots);
        }
        for (i, k) in knots.iter().enumerate() {
            if !(k.t.is_finite() && k.pos.is_finite() && k.vel.is_finite()) {
                return Err(TrackError::NonFinite { index: i });
            }
            if i > 0 && k.t <= knots[i - 1].t {
                return Err(TrackError::NotIncreasing { index: i });
            }
        }
        Ok(Track {
            id: id.into(),
            knots,
        })
    }

    pub fn span(&self) -> (f64, f64) {
        match (self.knots.first(), self.knots.last()) {
            (Some(a), Some(b)) => (a.t, b.t),
            _ => (f64::NAN, f64::NAN),
        }
    }

    /// Cubic Hermite position and its derivative; None outside [t_start, t_end].
    pub fn state_at(&self, t: f64) -> Option<(Vec3, Vec3)> {
        let (t0, t1) = self.span();
        if !(t >= t0 && t <= t1) {
            return None;
        }
        // Index of the segment [knots[i], knots[i+1]] containing t.
        let i = self
            .knots
            .partition_point(|k| k.t <= t)
            .saturating_sub(1)
            .min(self.knots.len() - 2);
        let (a, b) = (&self.knots[i], &self.knots[i + 1]);
        let h = b.t - a.t;
        let s = (t - a.t) / h;
        let (s2, s3) = (s * s, s * s * s);
        // Interpolate the displacement from `a` (h00 + h01 = 1) so ECEF magnitudes don't cancel.
        let dp = b.pos - a.pos;
        let (h10, h01, h11) = (s3 - 2.0 * s2 + s, -2.0 * s3 + 3.0 * s2, s3 - s2);
        let (d10, d01, d11) = (
            3.0 * s2 - 4.0 * s + 1.0,
            -6.0 * s2 + 6.0 * s,
            3.0 * s2 - 2.0 * s,
        );
        let pos = a.pos + dp * h01 + a.vel * (h * h10) + b.vel * (h * h11);
        let vel = dp * (d01 / h) + a.vel * d10 + b.vel * d11;
        Some((pos, vel))
    }

    /// The knots within [t0, t1] plus one on each side; None if fewer than two remain.
    pub fn window(&self, t0: f64, t1: f64) -> Option<Track> {
        let first = self.knots.partition_point(|k| k.t < t0).saturating_sub(1);
        let last = (self.knots.partition_point(|k| k.t <= t1) + 1).min(self.knots.len());
        if last <= first + 1 || self.knots[first].t > t1 || self.knots[last - 1].t < t0 {
            return None;
        }
        Some(Track {
            id: self.id.clone(),
            knots: self.knots[first..last].to_vec(),
        })
    }

    /// Split wherever consecutive knots are more than `max_gap_s` apart; pieces with < 2 knots are dropped.
    pub fn split_gaps(self, max_gap_s: f64) -> Vec<Track> {
        let mut out = Vec::new();
        let mut cur: Vec<Knot> = Vec::new();
        let mut piece = 0usize;
        let mut flush = |cur: &mut Vec<Knot>, piece: &mut usize| {
            if cur.len() >= 2 {
                let id = if *piece == 0 {
                    self.id.clone()
                } else {
                    format!("{}#{}", self.id, piece)
                };
                out.push(Track {
                    id,
                    knots: std::mem::take(cur),
                });
                *piece += 1;
            } else {
                cur.clear();
            }
        };
        for k in &self.knots {
            if let Some(last) = cur.last()
                && k.t - last.t > max_gap_s
            {
                flush(&mut cur, &mut piece);
            }
            cur.push(*k);
        }
        flush(&mut cur, &mut piece);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn linear(p0: Vec3, v: Vec3, times: &[f64]) -> Track {
        let knots = times
            .iter()
            .map(|&t| Knot {
                t,
                pos: p0 + v * t,
                vel: v,
            })
            .collect();
        Track::new("lin", knots).unwrap()
    }

    /// Velocity bound for exact linear motion: f64 rounding of ~6e6 m positions over 1 s knot
    /// gaps gives ~1e-9 m/s; 1e-7 m/s keeps two orders of headroom and is physically nothing.
    const VEL_TOL: f64 = 1e-7;

    proptest! {
        #[test]
        fn hermite_exact_for_linear_motion(vx in -300.0f64..300.0, vy in -300.0f64..300.0, vz in -20.0f64..20.0, t in 0.0f64..30.0) {
            let p0 = Vec3::new(1.3e6, -4.6e6, 4.2e6);
            let v = Vec3::new(vx, vy, vz);
            let tr = linear(p0, v, &[0.0, 1.0, 2.5, 7.0, 13.0, 30.0]);
            let (p, vel) = tr.state_at(t).unwrap();
            prop_assert!((p - (p0 + v * t)).norm() < 1e-6);
            prop_assert!((vel - v).norm() < VEL_TOL);
        }
    }

    #[test]
    fn hermite_linear_ci_counterexample() {
        // proptest counterexample from CI (macOS): ECEF-sized positions round at ~1e-9 m, so the
        // interpolated velocity carries ~1e-9 m/s of rounding — an absolute 1e-9 bound is flaky.
        let p0 = Vec3::new(1.3e6, -4.6e6, 4.2e6);
        let v = Vec3::new(0.0, 123.57364401914252, 8.269649592687855);
        let tr = linear(p0, v, &[0.0, 1.0, 2.5, 7.0, 13.0, 30.0]);
        let t = 1.7658440287138908;
        let (p, vel) = tr.state_at(t).unwrap();
        assert!((p - (p0 + v * t)).norm() < 1e-6);
        assert!((vel - v).norm() < VEL_TOL, "{:e}", (vel - v).norm());
    }

    #[test]
    fn velocity_is_derivative() {
        // A circle of radius 2 km at 60 m/s, sampled every 2 s.
        let w = 60.0 / 2000.0;
        let knots = (0..30)
            .map(|i| {
                let t = f64::from(i) * 2.0;
                Knot {
                    t,
                    pos: Vec3::new(2000.0 * (w * t).cos(), 2000.0 * (w * t).sin(), 1000.0),
                    vel: Vec3::new(-60.0 * (w * t).sin(), 60.0 * (w * t).cos(), 0.0),
                }
            })
            .collect();
        let tr = Track::new("circle", knots).unwrap();
        for i in 1..500 {
            let t = f64::from(i) * 0.1;
            let h = 1e-4;
            let (Some((a, _)), Some((_, v)), Some((b, _))) =
                (tr.state_at(t - h), tr.state_at(t), tr.state_at(t + h))
            else {
                panic!("in span at {t}");
            };
            assert!(((b - a) * (0.5 / h) - v).norm() < 1e-6, "t={t}");
        }
    }

    #[test]
    fn outside_span_is_none() {
        let tr = linear(Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0), &[10.0, 20.0, 30.0]);
        assert_eq!(tr.span(), (10.0, 30.0));
        assert!(tr.state_at(9.999).is_none());
        assert!(tr.state_at(30.001).is_none());
        assert!(tr.state_at(f64::NAN).is_none());
        assert!(tr.state_at(10.0).is_some() && tr.state_at(30.0).is_some());
    }

    #[test]
    fn rejects_bad_knots() {
        let k = |t: f64| Knot {
            t,
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
        };
        assert_eq!(Track::new("a", vec![k(0.0)]), Err(TrackError::TooFewKnots));
        assert_eq!(
            Track::new("a", vec![k(0.0), k(0.0)]),
            Err(TrackError::NotIncreasing { index: 1 })
        );
        assert_eq!(
            Track::new("a", vec![k(0.0), k(f64::NAN)]),
            Err(TrackError::NonFinite { index: 1 })
        );
        let mut bad = k(1.0);
        bad.pos.x = f64::INFINITY;
        assert_eq!(
            Track::new("a", vec![k(0.0), bad]),
            Err(TrackError::NonFinite { index: 1 })
        );
    }

    #[test]
    fn deserialize_validates() {
        let one = r#"{"id":"x","knots":[{"t":0.0,"pos":{"x":0.0,"y":0.0,"z":0.0},"vel":{"x":0.0,"y":0.0,"z":0.0}}]}"#;
        assert!(serde_json::from_str::<Track>(one).is_err());
        let k = r#"{"t":1.0,"pos":{"x":0.0,"y":0.0,"z":0.0},"vel":{"x":0.0,"y":0.0,"z":0.0}}"#;
        let dup = format!(r#"{{"id":"x","knots":[{k},{k}]}}"#);
        assert!(serde_json::from_str::<Track>(&dup).is_err());
        let tr = linear(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), &[0.0, 1.0]);
        let back: Track = serde_json::from_str(&serde_json::to_string(&tr).unwrap()).unwrap();
        assert_eq!(back, tr);
    }

    #[test]
    fn window_keeps_neighbours() {
        let tr = linear(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        );
        let w = tr.window(15.0, 32.0).unwrap();
        assert_eq!(w.span(), (10.0, 40.0));
        assert!(w.state_at(15.0).is_some() && w.state_at(32.0).is_some());
        assert!(tr.window(100.0, 200.0).is_none());
        assert_eq!(tr.window(-5.0, 5.0).unwrap().span(), (0.0, 10.0));
    }

    #[test]
    fn split_gaps() {
        let tr = linear(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            &[0.0, 1.0, 2.0, 30.0, 31.0, 90.0],
        );
        let parts = tr.split_gaps(10.0);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].span(), (0.0, 2.0));
        assert_eq!(parts[1].span(), (30.0, 31.0));
    }
}
