//! Forward acoustic physics: track interpolation, emission-time Doppler, CPA, closed forms.

pub mod absorption;
pub mod column;
pub mod cpa;
pub mod cv;
pub mod emission;
pub mod track;

pub use column::{Column, ColumnNode, doppler_factor_layered, layered_emission};
pub use cpa::{Cpa, cpa};
pub use emission::{Medium, doppler_factor, emission_time, received_frequencies};
pub use track::{Knot, Track, TrackError};
