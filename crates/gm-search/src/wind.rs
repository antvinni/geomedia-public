//! Wind as a fitted nuisance: one horizontal wind offset per cell, shared by every event,
//! chosen to best explain the exact Doppler curves under a Gaussian prior.

use gm_core::LocalFrame;

use crate::atmos::Atmosphere;
use crate::curve::score_cell_refined;
use crate::grid::{Cell, Inputs};
use crate::score::CellScore;

/// Polar search speeds (m/s); 12 directions each, speed 0 once.
const SPEEDS: [f64; 6] = [4.0, 8.0, 12.0, 16.0, 20.0, 25.0];
const DIRECTIONS: usize = 12;
/// Local refinement: ±REFINE_STEPS × 1 m/s in east and north around the best grid point.
const REFINE_STEPS: i32 = 2;

/// The cell's refined score under `base` plus the wind offset (east, north) that maximises it,
/// the prior −½|w|²/σ² included in the returned ll.
pub fn fit_wind(
    inp: &Inputs,
    base: &Atmosphere,
    frame: &LocalFrame,
    cell: &Cell,
    sigma: f64,
) -> Option<(CellScore, (f64, f64))> {
    let eval = |e: f64, n: f64| -> Option<CellScore> {
        let atm = base.with_wind_offset(frame, e, n)?;
        let sub = Inputs {
            atmos: &atm,
            ..*inp
        };
        let mut s = score_cell_refined(&sub, cell);
        s.ll -= 0.5 * (e * e + n * n) / (sigma * sigma);
        Some(s)
    };
    let mut best: Option<(CellScore, (f64, f64))> = eval(0.0, 0.0).map(|s| (s, (0.0, 0.0)));
    let consider = |e: f64, n: f64, best: &mut Option<(CellScore, (f64, f64))>| {
        if let Some(s) = eval(e, n)
            && best.as_ref().is_none_or(|(b, _)| s.ll > b.ll)
        {
            *best = Some((s, (e, n)));
        }
    };
    for v in SPEEDS {
        for k in 0..DIRECTIONS {
            let a = std::f64::consts::TAU * k as f64 / DIRECTIONS as f64;
            consider(v * a.sin(), v * a.cos(), &mut best);
        }
    }
    let (e0, n0) = best.as_ref()?.1;
    for j in -REFINE_STEPS..=REFINE_STEPS {
        for i in -REFINE_STEPS..=REFINE_STEPS {
            if i != 0 || j != 0 {
                consider(e0 + f64::from(i), n0 + f64::from(j), &mut best);
            }
        }
    }
    best
}
