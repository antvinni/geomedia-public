//! Emission (retarded) time and Doppler factor for a moving source, a fixed observer and a
//! uniformly moving medium (Ferguson, JASA 1992; moving-medium Doppler).
//!
//! In the air frame the medium is still; sound emitted at τ reaches the observer at t when
//! |D(τ)| = c (t − τ), with D(τ) = r − p(τ) − w (t − τ). g(τ) = τ + |D|/c − t is strictly
//! increasing for a subsonic source, so the root is unique and Newton converges.

use gm_core::Vec3;

use super::track::Track;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Medium {
    pub c: f64,
    /// Uniform wind, ECEF m/s.
    pub wind: Vec3,
}

impl Medium {
    pub fn still(c: f64) -> Option<Medium> {
        Medium::new(c, Vec3::ZERO)
    }

    /// Requires c > 0 and |wind| < c.
    pub fn new(c: f64, wind: Vec3) -> Option<Medium> {
        (c.is_finite() && c > 0.0 && wind.is_finite() && wind.norm() < c)
            .then_some(Medium { c, wind })
    }
}

/// Newton solve for the emission time; returns (τ, Newton updates used).
pub(crate) fn solve(
    track: &Track,
    obs: Vec3,
    t_rx: f64,
    m: Medium,
    warm: Option<f64>,
) -> Option<(f64, u32)> {
    const MAX_ITERS: u32 = 50;
    if !t_rx.is_finite() {
        return None;
    }
    let (t0, t1) = track.span();
    // g is increasing: the root lies inside the span only if g(t0) <= 0 <= g(t1).
    if g(track, obs, t_rx, m, t0)?.0 > 0.0 || g(track, obs, t_rx, m, t1)?.0 < 0.0 {
        return None;
    }
    let tol = (4.0 * f64::EPSILON * t_rx.abs()).max(1e-9);
    let mut tau = match warm.filter(|w| w.is_finite()) {
        Some(w) => w.clamp(t0, t1),
        None => {
            let (p, _) = track.state_at(t_rx.clamp(t0, t1))?;
            (t_rx - (obs - p).norm() / m.c).clamp(t0, t1)
        }
    };
    for iter in 0..=MAX_ITERS {
        let (gv, dg) = g(track, obs, t_rx, m, tau)?;
        if gv.abs() < tol {
            // Above the speed of sound in the air frame g is not monotone: no unique emission.
            let (_, u) = track.state_at(tau)?;
            return ((u - m.wind).norm() < m.c).then_some((tau, iter));
        }
        if dg <= 0.0 {
            return None;
        }
        tau = (tau - gv / dg).clamp(t0, t1);
    }
    None
}

/// g(τ) = τ + |D(τ)|/c − t and g'(τ) = 1 + k̂·(w − u)/c.
fn g(track: &Track, obs: Vec3, t_rx: f64, m: Medium, tau: f64) -> Option<(f64, f64)> {
    let (p, u) = track.state_at(tau)?;
    let d = obs - p - m.wind * (t_rx - tau);
    let dist = d.norm();
    let dg = match d.unit() {
        Some(k) => 1.0 + k.dot(m.wind - u) / m.c,
        None => 1.0,
    };
    Some((tau + dist / m.c - t_rx, dg))
}

/// The Doppler factor at a solved emission time.
pub(crate) fn factor_at(track: &Track, obs: Vec3, t_rx: f64, tau: f64, m: Medium) -> Option<f64> {
    let (p, u) = track.state_at(tau)?;
    let d = obs - p - m.wind * (t_rx - tau);
    if d.norm() < 1e-6 {
        return None;
    }
    let k = d.unit()?;
    let c_eff = m.c + m.wind.dot(k);
    let denom = c_eff - u.dot(k);
    (denom > 0.0 && c_eff > 0.0).then(|| c_eff / denom)
}

/// Emission time for receive time `t_rx`, or None if that sound left outside the track's span.
pub fn emission_time(
    track: &Track,
    obs: Vec3,
    t_rx: f64,
    m: Medium,
    warm: Option<f64>,
) -> Option<f64> {
    solve(track, obs, t_rx, m, warm).map(|(tau, _)| tau)
}

/// F = f_rx / f0 = (c + w·k̂) / (c + w·k̂ − u·k̂), with k̂ = D/|D| the air-frame wave normal.
pub fn doppler_factor(track: &Track, obs: Vec3, t_rx: f64, m: Medium) -> Option<f64> {
    let tau = emission_time(track, obs, t_rx, m, None)?;
    factor_at(track, obs, t_rx, tau, m)
}

/// Received frequency at each receive time, warm-started along the series.
pub fn received_frequencies(
    track: &Track,
    obs: Vec3,
    t_rx: &[f64],
    f0: f64,
    m: Medium,
) -> Vec<Option<f64>> {
    let mut prev: Option<(f64, f64, f64)> = None;
    t_rx.iter()
        .map(|&t| {
            let warm = prev.map(|(tp, taup, fp)| taup + (t - tp) * fp);
            let tau = emission_time(track, obs, t, m, warm);
            let f = tau.and_then(|tau| factor_at(track, obs, t, tau, m).map(|fac| (tau, fac)));
            prev = f.map(|(tau, fac)| (t, tau, fac));
            f.map(|(_, fac)| f0 * fac)
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::physics::track::Knot;
    use gm_core::{Epoch, Geodetic, LocalFrame};

    pub(crate) fn frame() -> LocalFrame {
        LocalFrame::new(Geodetic {
            lat_deg: 40.8765,
            lon_deg: -73.525,
            h_m: 10.0,
        })
    }

    /// A straight level pass along +east at `speed`, `alt` up, `offset` north of the observer at the origin.
    pub(crate) fn straight(
        f: &LocalFrame,
        speed: f64,
        alt: f64,
        offset: f64,
        half_span: f64,
        t_shift: f64,
    ) -> Track {
        let n = (2.0 * half_span) as i64;
        let knots = (0..=n)
            .map(|i| {
                let t = -half_span + i as f64;
                Knot {
                    t: t + t_shift,
                    pos: f.from_enu(Vec3::new(speed * t, offset, alt)),
                    vel: f.vec_from_enu(Vec3::new(speed, 0.0, 0.0)),
                }
            })
            .collect();
        Track::new("straight", knots).unwrap()
    }

    fn obs(f: &LocalFrame) -> Vec3 {
        f.from_enu(Vec3::ZERO)
    }

    #[test]
    fn zero_wind_matches_textbook() {
        let f = frame();
        let tr = straight(&f, 120.0, 1500.0, 800.0, 120.0, 0.0);
        let m = Medium::still(343.0).unwrap();
        // Before t ≈ -78 s the sound arriving left before the track starts (correctly None).
        for i in -70..80 {
            let t = f64::from(i);
            let tau = emission_time(&tr, obs(&f), t, m, None).unwrap();
            let (p, u) = tr.state_at(tau).unwrap();
            let n = (obs(&f) - p).unit().unwrap();
            let want = 343.0 / (343.0 - u.dot(n));
            let got = doppler_factor(&tr, obs(&f), t, m).unwrap();
            assert!((got - want).abs() < 1e-12, "t={t} {got} {want}");
        }
    }

    #[test]
    fn factor_equals_dtau_dt() {
        let f = frame();
        let tr = straight(&f, 120.0, 1500.0, 800.0, 120.0, 0.0);
        let m = Medium::new(343.0, f.vec_from_enu(Vec3::new(8.0, -9.5, 0.0))).unwrap();
        for i in -60..60 {
            let t = f64::from(i) + 0.37;
            let h = 1e-3;
            let a = emission_time(&tr, obs(&f), t - h, m, None).unwrap();
            let b = emission_time(&tr, obs(&f), t + h, m, None).unwrap();
            let fd = (b - a) / (2.0 * h);
            let got = doppler_factor(&tr, obs(&f), t, m).unwrap();
            assert!((got - fd).abs() < 1e-7, "t={t} {got} {fd}");
        }
    }

    #[test]
    fn along_track_wind_asymptotes() {
        let f = frame();
        let tr = straight(&f, 80.0, 0.0, 100.0, 3000.0, 0.0);
        let m = Medium::new(340.0, f.vec_from_enu(Vec3::new(10.0, 0.0, 0.0))).unwrap();
        // Source ~200 km before and after the observer.
        let early = doppler_factor(&tr, obs(&f), -2500.0 + 200e3 / 350.0, m).unwrap();
        let late = doppler_factor(&tr, obs(&f), 2500.0 + 200e3 / 330.0, m).unwrap();
        assert!((early - 350.0 / 270.0).abs() < 1e-4, "{early}");
        assert!((late - 330.0 / 410.0).abs() < 1e-4, "{late}");
    }

    #[test]
    fn warm_start_converges_fast() {
        let f = frame();
        let m = Medium::new(343.0, f.vec_from_enu(Vec3::new(5.0, 3.0, 0.0))).unwrap();
        let st = straight(&f, 120.0, 1500.0, 800.0, 200.0, 0.0);
        // A 3°/s turn: radius v/ω.
        let w = 3f64.to_radians();
        let r = 90.0 / w;
        let knots = (0..=240)
            .map(|i| {
                let t = f64::from(i) - 120.0;
                Knot {
                    t,
                    pos: f.from_enu(Vec3::new(
                        r * (w * t).sin(),
                        r * (1.0 - (w * t).cos()) + 600.0,
                        1200.0,
                    )),
                    vel: f.vec_from_enu(Vec3::new(90.0 * (w * t).cos(), 90.0 * (w * t).sin(), 0.0)),
                }
            })
            .collect();
        let turn = Track::new("turn", knots).unwrap();
        for tr in [&st, &turn] {
            let mut prev: Option<(f64, f64, f64)> = None;
            for i in 0..2000 {
                let t = -100.0 + f64::from(i) * 0.05;
                let warm = prev.map(|(tp, taup, fp)| taup + (t - tp) * fp);
                let (tau, iters) = solve(tr, obs(&f), t, m, warm).unwrap();
                if prev.is_some() {
                    assert!(iters <= 3, "{} t={t} iters={iters}", tr.id);
                }
                prev = Some((t, tau, doppler_factor(tr, obs(&f), t, m).unwrap()));
            }
        }
    }

    #[test]
    fn out_of_track_none() {
        let f = frame();
        let tr = straight(&f, 120.0, 1500.0, 800.0, 60.0, 0.0);
        let m = Medium::still(343.0).unwrap();
        assert!(
            emission_time(&tr, obs(&f), -60.0, m, None).is_none(),
            "sound from t=-60 has not arrived yet"
        );
        assert!(
            doppler_factor(&tr, obs(&f), 500.0, m).is_none(),
            "emission after the track ended"
        );
        assert!(doppler_factor(&tr, obs(&f), f64::NAN, m).is_none());
    }

    #[test]
    fn observer_on_track_none() {
        let f = frame();
        let tr = straight(&f, 120.0, 0.0, 0.0, 60.0, 0.0);
        let on = tr.state_at(0.0).unwrap().0;
        let m = Medium::still(343.0).unwrap();
        assert!(doppler_factor(&tr, on, 0.0, m).is_none());
    }

    #[test]
    fn supersonic_source_none() {
        let f = frame();
        let tr = straight(&f, 400.0, 1500.0, 800.0, 120.0, 0.0);
        let m = Medium::still(343.0).unwrap();
        for i in 0..300 {
            let t = -100.0 + f64::from(i) * 0.7;
            assert!(doppler_factor(&tr, obs(&f), t, m).is_none(), "t={t}");
        }
    }

    #[test]
    fn invalid_medium_rejected() {
        assert!(Medium::still(0.0).is_none());
        assert!(Medium::still(-1.0).is_none());
        assert!(Medium::still(f64::NAN).is_none());
        assert!(Medium::new(343.0, Vec3::new(400.0, 0.0, 0.0)).is_none());
        assert!(Medium::new(343.0, Vec3::new(f64::INFINITY, 0.0, 0.0)).is_none());
        assert!(Medium::new(343.0, Vec3::new(10.0, 0.0, 0.0)).is_some());
    }

    #[test]
    fn absolute_vs_relative_epoch() {
        // Solvers take seconds relative to a run epoch close to the data; any such origin gives
        // the same answer to 1e-12. Raw UNIX seconds (~1.7e9) are not supported: f64 spacing
        // there is 2.4e-7 s, so callers must go through `Epoch::seconds_since`.
        let f = frame();
        let t_first = Epoch::from_rfc3339("2026-09-08T20:50:00Z").unwrap();
        let rx = Epoch::from_rfc3339("2026-09-08T20:51:03.25Z").unwrap();
        let m = Medium::still(343.0).unwrap();
        let mut got = Vec::new();
        for origin in [
            t_first,
            t_first.plus_seconds(-3600.0),
            t_first.plus_seconds(1800.0),
        ] {
            let shift = t_first.seconds_since(origin);
            let tr = straight(&f, 120.0, 1500.0, 800.0, 120.0, shift + 120.0);
            got.push(doppler_factor(&tr, obs(&f), rx.seconds_since(origin), m).unwrap());
        }
        assert!((got[0] - got[1]).abs() < 1e-12, "{got:?}");
        assert!((got[0] - got[2]).abs() < 1e-12, "{got:?}");
    }
}
