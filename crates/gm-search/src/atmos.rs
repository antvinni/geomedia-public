//! The atmosphere the search assumes: still air, a uniform moving medium, or a height-layered
//! column (sound speed and horizontal wind by height).

use gm_core::{LocalFrame, Vec3, ecef_to_geodetic};
use gm_doppler::physics::{
    Column, ColumnNode, Cpa, Medium, Track, cpa, doppler_factor_layered, received_frequencies,
};

use crate::params::SEARCH_C;

/// Still air at the search's default sound speed, for places that need a `'static` reference.
pub static STILL_AIR: Atmosphere = Atmosphere::Uniform(Medium {
    c: SEARCH_C,
    wind: Vec3::ZERO,
});

#[derive(Debug, Clone)]
pub enum Atmosphere {
    Uniform(Medium),
    Layered(Column),
}

impl Atmosphere {
    /// Still air at the search's default sound speed.
    pub fn still() -> Atmosphere {
        Atmosphere::Uniform(Medium {
            c: SEARCH_C,
            wind: Vec3::ZERO,
        })
    }

    /// Doppler factors (f₀ = 1) for receive times `times` at `obs`.
    pub fn factors(&self, track: &Track, obs: Vec3, times: &[f64]) -> Vec<Option<f64>> {
        match self {
            Atmosphere::Uniform(m) => received_frequencies(track, obs, times, 1.0, *m),
            Atmosphere::Layered(col) => times
                .iter()
                .map(|&t| doppler_factor_layered(track, obs, t, col))
                .collect(),
        }
    }

    /// Geometric CPA and its arrival time; a layered column uses the layer between the
    /// observer and the aircraft at CPA.
    pub fn cpa(&self, track: &Track, obs: Vec3) -> Option<Cpa> {
        match self {
            Atmosphere::Uniform(m) => cpa(track, obs, *m),
            Atmosphere::Layered(col) => {
                let h_obs = ecef_to_geodetic(obs).h_m;
                let first = cpa(track, obs, col.layer_medium(h_obs, h_obs)?)?;
                let h_air = ecef_to_geodetic(track.state_at(first.t_emit)?.0).h_m;
                cpa(track, obs, col.layer_medium(h_obs, h_air)?)
            }
        }
    }

    /// One line for reports, e.g. "still air, c 343.0 m/s" or "layered column (9 levels)".
    pub fn describe(&self) -> String {
        match self {
            Atmosphere::Uniform(m) if m.wind.norm() == 0.0 => {
                format!("still air, c {:.1} m/s", m.c)
            }
            Atmosphere::Uniform(m) => {
                format!("uniform, c {:.1} m/s, wind {:.1} m/s", m.c, m.wind.norm())
            }
            Atmosphere::Layered(col) => format!("layered column ({} levels)", col.nodes().len()),
        }
    }

    /// The same atmosphere with a horizontal wind (east, north; m/s, in `frame`) added everywhere.
    pub fn with_wind_offset(
        &self,
        frame: &LocalFrame,
        east: f64,
        north: f64,
    ) -> Option<Atmosphere> {
        match self {
            Atmosphere::Uniform(m) => {
                let w = m.wind + frame.vec_from_enu(Vec3::new(east, north, 0.0));
                Medium::new(m.c, w).map(Atmosphere::Uniform)
            }
            Atmosphere::Layered(col) => {
                let nodes = col
                    .nodes()
                    .iter()
                    .map(|n| ColumnNode {
                        wind_east: n.wind_east + east,
                        wind_north: n.wind_north + north,
                        ..*n
                    })
                    .collect();
                Column::new(col.frame().clone(), nodes).map(Atmosphere::Layered)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::fixture;

    fn node(h: f64, e: f64, n: f64) -> ColumnNode {
        ColumnNode {
            h_m: h,
            c: 343.0,
            wind_east: e,
            wind_north: n,
        }
    }

    #[test]
    fn layered_matches_uniform_for_constant_column() {
        let f = fixture("two_props.toml", |s| s);
        let frame = LocalFrame::new(f.truth.observer);
        let tr = &f.truth.tracks[0].track;
        let obs = f.truth.mic_ecef;
        let wind = frame.vec_from_enu(Vec3::new(3.0, -2.0, 0.0));
        let uni = Atmosphere::Uniform(Medium::new(343.0, wind).unwrap());
        let times: Vec<f64> = (0..20).map(|i| 10.0 + 3.0 * f64::from(i)).collect();
        let u = uni.factors(tr, obs, &times);
        let uc = uni.cpa(tr, obs).unwrap();
        assert_eq!(u.len(), 20);
        for nodes in [
            vec![node(0.0, 3.0, -2.0), node(5000.0, 3.0, -2.0)],
            vec![node(3000.0, 3.0, -2.0), node(5000.0, 3.0, -2.0)],
        ] {
            let lay = Atmosphere::Layered(Column::new(frame.clone(), nodes).unwrap());
            let l = lay.factors(tr, obs, &times);
            for (a, b) in u.iter().zip(&l) {
                assert!((a.unwrap() - b.unwrap()).abs() < 1e-9, "{a:?} {b:?}");
            }
            let lc = lay.cpa(tr, obs).unwrap();
            assert!((uc.t_arrive - lc.t_arrive).abs() < 1e-6, "{uc:?} {lc:?}");
        }
    }

    #[test]
    fn wind_offset_adds() {
        let frame = LocalFrame::new(gm_core::Geodetic {
            lat_deg: 40.0,
            lon_deg: -73.0,
            h_m: 0.0,
        });
        let a = Atmosphere::still()
            .with_wind_offset(&frame, 3.0, 4.0)
            .unwrap();
        let Atmosphere::Uniform(m) = a else { panic!() };
        assert!((m.wind.norm() - 5.0).abs() < 1e-9);
        let col = Column::new(
            frame.clone(),
            vec![node(0.0, 1.0, 0.0), node(1000.0, 2.0, 1.0)],
        )
        .unwrap();
        let Atmosphere::Layered(c) = Atmosphere::Layered(col)
            .with_wind_offset(&frame, 3.0, 0.0)
            .unwrap()
        else {
            panic!()
        };
        let e: Vec<f64> = c.nodes().iter().map(|n| n.wind_east).collect();
        assert_eq!(e, vec![4.0, 5.0]);
    }
}
