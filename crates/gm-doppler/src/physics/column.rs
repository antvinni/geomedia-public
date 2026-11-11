//! A height-layered atmosphere: sound speed and horizontal wind as piecewise-linear profiles.
//! Each emission uses the uniform medium of its own layer between observer and aircraft:
//! c̄ = Δh / ∫dh/c (harmonic mean) and W̄ = ∫W dh / Δh (straight-chord first-order average).

use gm_core::{LocalFrame, Vec3, ecef_to_geodetic};
use serde::{Deserialize, Serialize};

use super::emission::{Medium, emission_time, factor_at};
use super::track::Track;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnNode {
    /// Ellipsoidal height.
    pub h_m: f64,
    pub c: f64,
    pub wind_east: f64,
    pub wind_north: f64,
}

#[derive(Clone, Debug)]
pub struct Column {
    frame: LocalFrame,
    nodes: Vec<ColumnNode>,
}

impl Column {
    /// None if empty, non-finite, or any c ≤ 0.
    pub fn new(frame: LocalFrame, mut nodes: Vec<ColumnNode>) -> Option<Column> {
        let ok = |n: &ColumnNode| {
            n.h_m.is_finite()
                && n.c.is_finite()
                && n.c > 0.0
                && n.wind_east.is_finite()
                && n.wind_north.is_finite()
        };
        if nodes.is_empty() || !nodes.iter().all(ok) {
            return None;
        }
        nodes.sort_by(|a, b| a.h_m.total_cmp(&b.h_m));
        Some(Column { frame, nodes })
    }

    pub fn nodes(&self) -> &[ColumnNode] {
        &self.nodes
    }

    pub fn frame(&self) -> &LocalFrame {
        &self.frame
    }

    /// (c, east, north) at height `h`, linear between nodes and constant beyond them.
    fn at(&self, h: f64) -> (f64, f64, f64) {
        let first = self.nodes[0];
        let last = self.nodes[self.nodes.len() - 1];
        if h <= first.h_m {
            return (first.c, first.wind_east, first.wind_north);
        }
        if h >= last.h_m {
            return (last.c, last.wind_east, last.wind_north);
        }
        let i = self
            .nodes
            .partition_point(|n| n.h_m <= h)
            .clamp(1, self.nodes.len() - 1);
        let (a, b) = (self.nodes[i - 1], self.nodes[i]);
        let s = (h - a.h_m) / (b.h_m - a.h_m);
        let lerp = |x: f64, y: f64| x + (y - x) * s;
        (
            lerp(a.c, b.c),
            lerp(a.wind_east, b.wind_east),
            lerp(a.wind_north, b.wind_north),
        )
    }

    /// Uniform medium for the layer between heights `h_a` and `h_b`; constant beyond the end nodes.
    pub fn layer_medium(&self, h_a: f64, h_b: f64) -> Option<Medium> {
        if !(h_a.is_finite() && h_b.is_finite()) {
            return None;
        }
        let (lo, hi) = if h_a <= h_b { (h_a, h_b) } else { (h_b, h_a) };
        let (c, e, n) = if hi - lo < 1e-6 {
            self.at(0.5 * (lo + hi))
        } else {
            // Breakpoints: the layer ends plus every node strictly inside.
            let mut zs = vec![lo];
            zs.extend(
                self.nodes
                    .iter()
                    .map(|n| n.h_m)
                    .filter(|&z| z > lo && z < hi),
            );
            zs.push(hi);
            let (mut slowness, mut ie, mut inn) = (0.0, 0.0, 0.0);
            for w in zs.windows(2) {
                let (za, zb) = (w[0], w[1]);
                let dz = zb - za;
                let (ca, ea, na) = self.at(za);
                let (cb, eb, nb) = self.at(zb);
                // ∫dz/c for c linear in z: Δz·ln(cb/ca)/(cb − ca), or Δz/c when constant.
                slowness += if (cb - ca).abs() < 1e-12 * ca {
                    dz / ca
                } else {
                    dz * (cb / ca).ln() / (cb - ca)
                };
                ie += 0.5 * (ea + eb) * dz;
                inn += 0.5 * (na + nb) * dz;
            }
            let span = hi - lo;
            (span / slowness, ie / span, inn / span)
        };
        Medium::new(c, self.frame.vec_from_enu(Vec3::new(e, n, 0.0)))
    }

    /// Speed of sound in dry air at `temp_c` °C.
    pub fn sound_speed(temp_c: f64) -> f64 {
        331.3 * (1.0 + temp_c / 273.15).sqrt()
    }
}

/// Emission time and the layer medium it was solved in (iterated until the layer settles).
pub fn layered_emission(
    track: &Track,
    obs: Vec3,
    t_rx: f64,
    col: &Column,
) -> Option<(f64, Medium)> {
    let h_obs = ecef_to_geodetic(obs).h_m;
    let (t0, t1) = track.span();
    let start = track.state_at(t_rx.clamp(t0, t1))?.0;
    let mut h_src = ecef_to_geodetic(start).h_m;
    let mut m = col.layer_medium(h_obs, h_src)?;
    let mut tau = None;
    for _ in 0..12 {
        let t = emission_time(track, obs, t_rx, m, tau)?;
        let h = ecef_to_geodetic(track.state_at(t)?.0).h_m;
        tau = Some(t);
        let settled = (h - h_src).abs() < 1e-9;
        h_src = h;
        m = col.layer_medium(h_obs, h_src)?;
        if settled {
            break;
        }
    }
    Some((tau?, m))
}

/// Doppler factor using the layer between the observer and the aircraft at emission.
pub fn doppler_factor_layered(track: &Track, obs: Vec3, t_rx: f64, col: &Column) -> Option<f64> {
    let (tau, m) = layered_emission(track, obs, t_rx, col)?;
    factor_at(track, obs, t_rx, tau, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::doppler_factor;
    use crate::physics::emission::tests::{frame, straight};

    fn node(h: f64, c: f64, e: f64, n: f64) -> ColumnNode {
        ColumnNode {
            h_m: h,
            c,
            wind_east: e,
            wind_north: n,
        }
    }

    #[test]
    fn uniform_column_equals_uniform_medium() {
        let f = frame();
        let col = Column::new(
            f.clone(),
            vec![node(0.0, 340.0, 5.0, -3.0), node(3000.0, 340.0, 5.0, -3.0)],
        )
        .unwrap();
        let m = col.layer_medium(10.0, 1500.0).unwrap();
        assert!((m.c - 340.0).abs() < 1e-12);
        assert!((m.wind - f.vec_from_enu(Vec3::new(5.0, -3.0, 0.0))).norm() < 1e-12);
    }

    #[test]
    fn harmonic_mean_speed() {
        let col = Column::new(
            frame(),
            vec![node(0.0, 340.0, 0.0, 0.0), node(1000.0, 330.0, 10.0, 0.0)],
        )
        .unwrap();
        let m = col.layer_medium(0.0, 1000.0).unwrap();
        let want = 1000.0 / ((1000.0 / -10.0) * (330.0f64 / 340.0).ln());
        assert!((m.c - want).abs() < 1e-9, "{} {want}", m.c);
        let east = frame().vec_to_enu(m.wind).x;
        assert!((east - 5.0).abs() < 1e-9, "{east}");
        // Order of the bounds doesn't matter.
        assert!((col.layer_medium(1000.0, 0.0).unwrap().c - want).abs() < 1e-9);
    }

    #[test]
    fn layer_beyond_nodes_constant() {
        let col = Column::new(
            frame(),
            vec![node(100.0, 340.0, 0.0, 0.0), node(200.0, 330.0, 0.0, 0.0)],
        )
        .unwrap();
        assert!((col.layer_medium(500.0, 900.0).unwrap().c - 330.0).abs() < 1e-12);
        assert!((col.layer_medium(-50.0, 50.0).unwrap().c - 340.0).abs() < 1e-12);
        assert!((col.layer_medium(150.0, 150.0).unwrap().c - 335.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_columns() {
        assert!(Column::new(frame(), vec![]).is_none());
        assert!(Column::new(frame(), vec![node(0.0, 0.0, 0.0, 0.0)]).is_none());
        assert!(Column::new(frame(), vec![node(f64::NAN, 340.0, 0.0, 0.0)]).is_none());
    }

    #[test]
    fn sound_speed_at_20c() {
        assert!((Column::sound_speed(20.0) - 343.2).abs() < 0.1);
    }

    #[test]
    fn layered_factor_level_flight() {
        let f = frame();
        let tr = straight(&f, 120.0, 1500.0, 800.0, 120.0, 0.0);
        let obs = f.from_enu(Vec3::ZERO);
        let h_obs = ecef_to_geodetic(obs).h_m;
        let col = Column::new(
            f.clone(),
            vec![
                node(h_obs, 342.0, 2.0, 1.0),
                node(h_obs + 800.0, 338.0, 9.0, -4.0),
                node(h_obs + 2500.0, 333.0, 15.0, -8.0),
            ],
        )
        .unwrap();
        for i in -50..60 {
            let t = f64::from(i) + 0.5;
            let (tau, m) = layered_emission(&tr, obs, t, &col).unwrap();
            let h_src = ecef_to_geodetic(tr.state_at(tau).unwrap().0).h_m;
            let want_m = col.layer_medium(h_obs, h_src).unwrap();
            assert!((m.c - want_m.c).abs() < 1e-9);
            let got = doppler_factor_layered(&tr, obs, t, &col).unwrap();
            let want = doppler_factor(&tr, obs, t, want_m).unwrap();
            assert!((got - want).abs() < 1e-9, "t={t} {got} {want}");
        }
    }
}
