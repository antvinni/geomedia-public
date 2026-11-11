//! WGS84 ellipsoid: geodetic ↔ ECEF and geodesic distance.

use serde::{Deserialize, Serialize};

use crate::vec3::Vec3;

pub const A: f64 = 6_378_137.0;
pub const F: f64 = 1.0 / 298.257_223_563;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Geodetic {
    pub lat_deg: f64,
    pub lon_deg: f64,
    /// Height above the ellipsoid.
    pub h_m: f64,
}

/// Semi-minor axis.
pub const B: f64 = A * (1.0 - F);
/// First eccentricity squared.
pub const E2: f64 = F * (2.0 - F);
/// Mean Earth radius (IUGG), for the haversine fallback.
pub const MEAN_RADIUS: f64 = 6_371_008.8;

pub fn geodetic_to_ecef(g: Geodetic) -> Vec3 {
    let (lat, lon) = (g.lat_deg.to_radians(), g.lon_deg.to_radians());
    let (sl, cl) = lat.sin_cos();
    let n = A / (1.0 - E2 * sl * sl).sqrt();
    Vec3::new(
        (n + g.h_m) * cl * lon.cos(),
        (n + g.h_m) * cl * lon.sin(),
        (n * (1.0 - E2) + g.h_m) * sl,
    )
}

/// Closed form (Heikkinen 1982 / Zhu 1993), then one Newton polish of latitude.
pub fn ecef_to_geodetic(p: Vec3) -> Geodetic {
    let lon = p.y.atan2(p.x);
    let r = (p.x * p.x + p.y * p.y).sqrt();
    if r < 1e-9 {
        let lat = if p.z >= 0.0 { 90.0 } else { -90.0 };
        return Geodetic {
            lat_deg: lat,
            lon_deg: lon.to_degrees(),
            h_m: p.z.abs() - B,
        };
    }
    let ep2 = (A * A - B * B) / (B * B);
    let z2 = p.z * p.z;
    let ff = 54.0 * B * B * z2;
    let g = r * r + (1.0 - E2) * z2 - E2 * (A * A - B * B);
    let c = E2 * E2 * ff * r * r / (g * g * g);
    let s = (1.0 + c + (c * c + 2.0 * c).sqrt()).cbrt();
    let k = s + 1.0 + 1.0 / s;
    let pp = ff / (3.0 * k * k * g * g);
    let q = (1.0 + 2.0 * E2 * E2 * pp).sqrt();
    let r0 = -(pp * E2 * r) / (1.0 + q)
        + (0.5 * A * A * (1.0 + 1.0 / q)
            - pp * (1.0 - E2) * z2 / (q * (1.0 + q))
            - 0.5 * pp * r * r)
            .max(0.0)
            .sqrt();
    let u = ((r - E2 * r0).powi(2) + z2).sqrt();
    let v = ((r - E2 * r0).powi(2) + (1.0 - E2) * z2).sqrt();
    let z0 = B * B * p.z / (A * v);
    let mut lat = ((p.z + ep2 * z0) / r).atan();
    let mut h = u * (1.0 - B * B / (A * v));
    // One iteration of the standard fixed point removes the closed form's residual.
    let sl = lat.sin();
    let n = A / (1.0 - E2 * sl * sl).sqrt();
    if lat.cos().abs() > 1e-6 {
        h = r / lat.cos() - n;
        lat = p.z.atan2(r * (1.0 - E2 * n / (n + h)));
    }
    Geodetic {
        lat_deg: lat.to_degrees(),
        lon_deg: lon.to_degrees(),
        h_m: h,
    }
}

/// Ellipsoidal distance (Vincenty inverse); haversine on the mean sphere if it fails to converge.
pub fn geodesic_distance_m(a: Geodetic, b: Geodetic) -> f64 {
    vincenty(a, b).unwrap_or_else(|| haversine(a, b))
}

fn vincenty(a: Geodetic, b: Geodetic) -> Option<f64> {
    let l = (b.lon_deg - a.lon_deg).to_radians();
    let u1 = ((1.0 - F) * a.lat_deg.to_radians().tan()).atan();
    let u2 = ((1.0 - F) * b.lat_deg.to_radians().tan()).atan();
    let (su1, cu1) = u1.sin_cos();
    let (su2, cu2) = u2.sin_cos();
    let mut lambda = l;
    for _ in 0..200 {
        let (sl, cl) = lambda.sin_cos();
        let sin_sigma = ((cu2 * sl).powi(2) + (cu1 * su2 - su1 * cu2 * cl).powi(2)).sqrt();
        if sin_sigma == 0.0 {
            return Some(0.0);
        }
        let cos_sigma = su1 * su2 + cu1 * cu2 * cl;
        let sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cu1 * cu2 * sl / sin_sigma;
        let cos2_alpha = 1.0 - sin_alpha * sin_alpha;
        let cos_2sm = if cos2_alpha != 0.0 {
            cos_sigma - 2.0 * su1 * su2 / cos2_alpha
        } else {
            0.0
        };
        let c = F / 16.0 * cos2_alpha * (4.0 + F * (4.0 - 3.0 * cos2_alpha));
        let prev = lambda;
        lambda = l
            + (1.0 - c)
                * F
                * sin_alpha
                * (sigma
                    + c * sin_sigma * (cos_2sm + c * cos_sigma * (-1.0 + 2.0 * cos_2sm * cos_2sm)));
        if (lambda - prev).abs() < 1e-12 {
            let u_sq = cos2_alpha * (A * A - B * B) / (B * B);
            let ka =
                1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
            let kb = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
            let ds = kb
                * sin_sigma
                * (cos_2sm
                    + kb / 4.0
                        * (cos_sigma * (-1.0 + 2.0 * cos_2sm * cos_2sm)
                            - kb / 6.0
                                * cos_2sm
                                * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                                * (-3.0 + 4.0 * cos_2sm * cos_2sm)));
            return Some(B * ka * (sigma - ds));
        }
    }
    None
}

fn haversine(a: Geodetic, b: Geodetic) -> f64 {
    let (p1, p2) = (a.lat_deg.to_radians(), b.lat_deg.to_radians());
    let dp = p2 - p1;
    let dl = (b.lon_deg - a.lon_deg).to_radians();
    let h = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * MEAN_RADIUS * h.sqrt().min(1.0).asin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ecef_roundtrip_property(lat in -89.9f64..89.9, lon in -180.0f64..180.0, h in -500.0f64..12000.0) {
            let g = Geodetic { lat_deg: lat, lon_deg: lon, h_m: h };
            let p = geodetic_to_ecef(g);
            let back = geodetic_to_ecef(ecef_to_geodetic(p));
            prop_assert!((back - p).norm() < 1e-6, "{:?} -> {:?}", g, ecef_to_geodetic(p));
        }
    }

    #[test]
    fn known_ecef() {
        let p = geodetic_to_ecef(Geodetic {
            lat_deg: 0.0,
            lon_deg: 0.0,
            h_m: 0.0,
        });
        assert_eq!(p, Vec3::new(A, 0.0, 0.0));
        let pole = geodetic_to_ecef(Geodetic {
            lat_deg: 90.0,
            lon_deg: 0.0,
            h_m: 0.0,
        });
        assert!((pole.z - 6_356_752.314_245).abs() < 1e-6, "{}", pole.z);
        let g = ecef_to_geodetic(Vec3::new(0.0, 0.0, 6_356_752.314_245 + 100.0));
        assert!(
            (g.lat_deg - 90.0).abs() < 1e-12 && (g.h_m - 100.0).abs() < 1e-6,
            "{g:?}"
        );
    }

    #[test]
    fn vincenty_known() {
        let flinders = Geodetic {
            lat_deg: -(37.0 + 57.0 / 60.0 + 3.72030 / 3600.0),
            lon_deg: 144.0 + 25.0 / 60.0 + 29.52440 / 3600.0,
            h_m: 0.0,
        };
        let buninyong = Geodetic {
            lat_deg: -(37.0 + 39.0 / 60.0 + 10.15610 / 3600.0),
            lon_deg: 143.0 + 55.0 / 60.0 + 35.38390 / 3600.0,
            h_m: 0.0,
        };
        let d = geodesic_distance_m(flinders, buninyong);
        assert!((d - 54_972.271).abs() < 1e-3, "{d}");
        assert_eq!(geodesic_distance_m(flinders, flinders), 0.0);
    }
}
