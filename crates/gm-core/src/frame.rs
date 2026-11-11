//! East-north-up tangent-plane frame at a geodetic origin.

use crate::vec3::Vec3;
use crate::wgs84::{Geodetic, ecef_to_geodetic, geodetic_to_ecef};

#[derive(Clone, Debug)]
pub struct LocalFrame {
    origin: Geodetic,
    origin_ecef: Vec3,
    east: Vec3,
    north: Vec3,
    up: Vec3,
}

impl LocalFrame {
    pub fn new(origin: Geodetic) -> Self {
        let (lat, lon) = (origin.lat_deg.to_radians(), origin.lon_deg.to_radians());
        let (sp, cp) = lat.sin_cos();
        let (sl, cl) = lon.sin_cos();
        Self {
            origin,
            origin_ecef: geodetic_to_ecef(origin),
            east: Vec3::new(-sl, cl, 0.0),
            north: Vec3::new(-sp * cl, -sp * sl, cp),
            up: Vec3::new(cp * cl, cp * sl, sp),
        }
    }

    pub fn origin(&self) -> Geodetic {
        self.origin
    }

    pub fn origin_ecef(&self) -> Vec3 {
        self.origin_ecef
    }

    pub fn to_enu(&self, p: Vec3) -> Vec3 {
        self.vec_to_enu(p - self.origin_ecef)
    }

    pub fn from_enu(&self, enu: Vec3) -> Vec3 {
        self.origin_ecef + self.vec_from_enu(enu)
    }

    pub fn vec_to_enu(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.east), v.dot(self.north), v.dot(self.up))
    }

    pub fn vec_from_enu(&self, v: Vec3) -> Vec3 {
        self.east * v.x + self.north * v.y + self.up * v.z
    }

    /// The point at ellipsoidal height `h_m` whose tangent-plane coordinates are (east, north).
    pub fn ground_point(&self, east_m: f64, north_m: f64, h_m: f64) -> Geodetic {
        let (mut e, mut n) = (east_m, north_m);
        let mut g = self.origin;
        for _ in 0..6 {
            g = ecef_to_geodetic(self.from_enu(Vec3::new(e, n, 0.0)));
            g.h_m = h_m;
            let got = self.to_enu(geodetic_to_ecef(g));
            let (de, dn) = (east_m - got.x, north_m - got.y);
            if de.abs() < 1e-6 && dn.abs() < 1e-6 {
                break;
            }
            e += de;
            n += dn;
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgs84::geodetic_to_ecef;
    use proptest::prelude::*;

    fn frame() -> LocalFrame {
        LocalFrame::new(Geodetic {
            lat_deg: 40.8765,
            lon_deg: -73.525,
            h_m: 12.0,
        })
    }

    proptest! {
        #[test]
        fn enu_roundtrip_property(e in -60e3f64..60e3, n in -60e3f64..60e3, u in -100.0f64..12e3) {
            let f = frame();
            let enu = Vec3::new(e, n, u);
            prop_assert!((f.to_enu(f.from_enu(enu)) - enu).norm() < 1e-6);
            let v = Vec3::new(e / 1e3, n / 1e3, u / 1e3);
            prop_assert!((f.vec_to_enu(f.vec_from_enu(v)) - v).norm() < 1e-12);
        }

        #[test]
        fn ground_point_inverts_enu(e in -50e3f64..50e3, n in -50e3f64..50e3, h in -50.0f64..3000.0) {
            let f = frame();
            let g = f.ground_point(e, n, h);
            prop_assert!((g.h_m - h).abs() < 1e-9);
            let enu = f.to_enu(geodetic_to_ecef(g));
            prop_assert!((enu.x - e).abs() < 1e-3 && (enu.y - n).abs() < 1e-3, "{:?} vs {} {}", enu, e, n);
        }
    }

    #[test]
    fn origin_is_zero() {
        let f = frame();
        assert!(f.to_enu(geodetic_to_ecef(f.origin())).norm() < 1e-9);
        let up = f.vec_to_enu(
            geodetic_to_ecef(Geodetic {
                h_m: 112.0,
                ..f.origin()
            }) - geodetic_to_ecef(f.origin()),
        );
        assert!((up - Vec3::new(0.0, 0.0, 100.0)).norm() < 1e-6, "{up:?}");
    }
}
