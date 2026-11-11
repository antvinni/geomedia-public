//! A deterministic flyover simulator. It renders what a microphone would record from aircraft
//! on given tracks, by evaluating each source's phase at the emission time found by plain
//! fixed-point iteration. It deliberately never calls the Doppler solver, so the two check each other.

pub mod render;
pub mod rng;
pub mod scenario;
pub mod wav;

pub use render::{Rendered, Truth, TruthEvent, TruthTrack, emission_fixed_point, render};
pub use scenario::{Scenario, SimError, load, parse};
pub use wav::write_wav;
