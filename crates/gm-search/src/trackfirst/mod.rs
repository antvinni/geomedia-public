//! Track-first location scoring: positions scored directly against every aircraft that would
//! have been audible there and then — per-pair maps over the clock offset (stage 1), a joint
//! score (2), a time-shift null giving a global p (3), and a fine check that can only refine or
//! veto (4). Design: docs/superpowers/specs/2026-11-06-track-first-design.md.

pub mod calibrate;
pub mod combine;
pub mod fine;
pub mod joint;
pub mod maps;
pub mod null;
pub mod run;

pub use combine::{Source, combine};
pub use fine::Contributor;
pub use joint::{Cand, Index, candidates, contribution, joint_at, max_joint};
pub use maps::{Hit, PairMap, pair_maps, slack_s};
pub use run::{TfOutcome, track_first};
