//! Geometry and time shared by every geomedia method: f64 ECEF/WGS84 geodesy, ENU local
//! frames and a nanosecond epoch, so solvers never work on raw UNIX f64 seconds.

pub mod epoch;
pub mod frame;
pub mod vec3;
pub mod wgs84;

pub use epoch::Epoch;
pub use frame::LocalFrame;
pub use vec3::Vec3;
pub use wgs84::{Geodetic, ecef_to_geodetic, geodesic_distance_m, geodetic_to_ecef};
