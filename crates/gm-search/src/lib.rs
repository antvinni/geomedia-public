//! Where was it recorded? Score candidate ground points by how well the ADS-B aircraft near
//! them explain the recording's Doppler events, under one shared recording clock.

pub mod atmos;
pub mod bench;
pub mod curve;
pub mod curvematch;
pub mod envelope;
pub mod grid;
pub mod guided;
pub mod matching;
pub mod noise;
pub mod params;
pub mod ridge_truth;
pub mod score;
pub mod search;
#[cfg(test)]
pub(crate) mod testutil;
pub mod trackfirst;
pub mod wind;

pub use curve::{curve_ll, score_cell_refined};
pub use curvematch::curve_matches;
pub use grid::{AircraftClass, Candidate, Cell, Inputs, candidates_from, grid};
pub use guided::{Band, drop_contested, guided_events, refine_events};
pub use matching::{Match, MatchVia, matches};
pub use score::{CellScore, score_cell};
pub use search::{Outcome, Ranked, SearchParams, Verdict, search};
