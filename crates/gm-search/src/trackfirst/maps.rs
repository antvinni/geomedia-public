//! Stage 1: for every (aircraft, cell) pair in range, the curve sum's z over every clock offset
//! δ at which its pass overlaps the recording — computed once on the coarse guide image. Later
//! stages only look values up and re-sum them.
//!
//! Each map is max-pooled over δ ± `slack_s` (the off-cell error of a coarse cell moves a pass's
//! arrival by up to that much, differently per aircraft); the null is built from the same pooled
//! maps, so the slack costs sensitivity, never calibration. Only δ steps whose pooled z reaches
//! the band's floor are stored: below it a pair contributes nothing.

use gm_core::LocalFrame;
use gm_doppler::audio::guide::GuideImage;

use crate::grid::{Cell, Inputs};
use crate::guided::{Band, DeltaZ, Sweep, curve_z};
use crate::params::{GUIDE_MIN_POINTS, GUIDE_RISE_MARGIN, GUIDE_SPAN_S, TF_DELTA_STEP_S};

/// One stored δ step of a pair map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Index of the δ step (δ = `delta0 + step · TF_DELTA_STEP_S`).
    pub step: u32,
    /// Pooled falling-curve z.
    pub z: f32,
    /// Pooled rising-curve z (NaN when unscored).
    pub zr: f32,
    /// f₀'s coarse log bin at the pooled maximum.
    pub q0: u16,
    /// Offset (δ steps) of the pooled maximum from `step`.
    pub lag: i8,
    /// The falling curve beats the rising one by `GUIDE_RISE_MARGIN`.
    pub rise_ok: bool,
}

/// One (aircraft, cell) pair's z map over δ.
#[derive(Clone, Debug)]
pub struct PairMap {
    /// Index into `Inputs::aircraft`, of its tracks, and of the cells.
    pub aircraft: usize,
    pub track: usize,
    pub cell: usize,
    pub band: Band,
    /// δ of step 0 (a multiple of `TF_DELTA_STEP_S`).
    pub delta0: f64,
    pub n_steps: usize,
    /// Stored steps, sorted by `step`.
    pub hits: Vec<Hit>,
    /// CPA arrival on the track clock.
    pub t_arrive: f64,
    pub v_mps: f64,
    /// Azimuth (degrees from north) from the cell to the aircraft at CPA.
    pub bearing_deg: f64,
    pub slant_m: f64,
    /// The curve: (frame offset from the arrival frame, log-bin shift), on `hop_s` frames.
    pub shape: Vec<(i32, i32)>,
    pub hop_s: f64,
}

impl PairMap {
    pub fn delta_of(&self, step: u32) -> f64 {
        self.delta0 + f64::from(step) * TF_DELTA_STEP_S
    }

    /// The stored hit at clock offset `delta`, if any.
    pub fn hit_at(&self, delta: f64) -> Option<&Hit> {
        let s = ((delta - self.delta0) / TF_DELTA_STEP_S).round();
        if s < 0.0 || s >= self.n_steps as f64 {
            return None;
        }
        let s = s as u32;
        self.hits
            .binary_search_by_key(&s, |h| h.step)
            .ok()
            .map(|i| &self.hits[i])
    }
}

/// δ slack (s) absorbing a `cell_m` cell's off-centre error for a pass at `v_mps` (floored at
/// 20 m/s): ⌈0.71·cell / v⌉.
pub fn slack_s(cell_m: f64, v_mps: f64) -> f64 {
    (0.71 * cell_m / v_mps.max(20.0)).ceil()
}

/// Frames a curve needs on `img`: GUIDE_MIN_POINTS at the guide hop, scaled to this hop.
fn min_points(img: &GuideImage) -> usize {
    ((GUIDE_MIN_POINTS as f64 * 0.25 / img.hop_s).ceil() as usize).max(1)
}

/// Sliding max of `v` over ±`k`: (value, offset of the max), NaN where nothing was scored.
fn pool(v: &[Option<DeltaZ>], k: usize) -> Vec<(f32, i8, u16)> {
    (0..v.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(k), (i + k).min(v.len().saturating_sub(1)));
            let mut best: (f32, i8, u16) = (f32::NAN, 0, 0);
            for (j, d) in v.iter().enumerate().take(b + 1).skip(a) {
                if let Some(d) = d
                    && (best.0.is_nan() || d.z as f32 > best.0)
                {
                    best = (d.z as f32, (j as i64 - i as i64) as i8, d.q0 as u16);
                }
            }
            best
        })
        .collect()
}

/// The map of one pair, keeping steps whose pooled z ≥ `keep_min`. `None` when the pass is out
/// of range or never overlaps `range` and the recording.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_map_with(
    img: &GuideImage,
    band: Band,
    inp: &Inputs,
    frame: &LocalFrame,
    (ai, ti, ci): (usize, usize, usize),
    cell: Cell,
    range: (f64, f64),
    cell_m: f64,
    keep_min: f64,
) -> Option<PairMap> {
    let track = &inp.aircraft.get(ai)?.tracks[ti];
    let c = inp.atmos.cpa(track, cell.ecef)?;
    if c.slant_m > band.max_slant_m() {
        return None;
    }
    // δ where the arrival falls in the recording, on the global δ grid, inside `range`.
    let step = TF_DELTA_STEP_S;
    let lo = ((c.t_arrive - inp.duration_s).max(range.0) / step).ceil() * step;
    let hi = (c.t_arrive.min(range.1) / step).floor() * step;
    if lo > hi {
        return None;
    }
    let sw = Sweep::with_step(img, (lo, hi), band, step)?;
    let mp = min_points(img);
    let (ta, fall) = curve_z(img, &sw, &sw.deltas, inp, track, cell, false, mp)?;
    let rise = curve_z(img, &sw, &sw.deltas, inp, track, cell, true, mp)
        .map(|r| r.1)
        .unwrap_or_else(|| vec![None; fall.len()]);
    let k = (slack_s(cell_m, c.ground_speed_mps) / step).round() as usize;
    let (pf, pr) = (pool(&fall, k), pool(&rise, k));
    let hits: Vec<Hit> = pf
        .iter()
        .zip(&pr)
        .enumerate()
        .filter(|(_, (f, _))| !f.0.is_nan() && f64::from(f.0) >= keep_min)
        .map(|(i, (f, r))| Hit {
            step: i as u32,
            z: f.0,
            zr: r.0,
            q0: f.2,
            lag: f.1,
            rise_ok: !r.0.is_nan() && f64::from(f.0) >= f64::from(r.0) + GUIDE_RISE_MARGIN,
        })
        .collect();
    // The curve's shape around the arrival, on this image's frames.
    let half = (GUIDE_SPAN_S / img.hop_s).round() as i32;
    let taus: Vec<f64> = (-half..=half)
        .map(|o| ta + f64::from(o) * img.hop_s)
        .collect();
    let w = track.window(taus[0] - 120.0, taus[taus.len() - 1] + 60.0)?;
    let shape: Vec<(i32, i32)> = (-half..=half)
        .zip(inp.atmos.factors(&w, cell.ecef, &taus))
        .filter_map(|(o, k)| {
            k.filter(|k| *k > 0.0)
                .map(|k| (o, (img.bpo * k.log2()).round() as i32))
        })
        .collect();
    let pos = track.state_at(c.t_emit)?.0;
    let d = frame.to_enu(pos) - frame.to_enu(cell.ecef);
    let bearing_deg = d.x.atan2(d.y).to_degrees().rem_euclid(360.0);
    Some(PairMap {
        aircraft: ai,
        track: ti,
        cell: ci,
        band,
        delta0: lo,
        n_steps: fall.len(),
        hits,
        t_arrive: ta,
        v_mps: c.ground_speed_mps,
        bearing_deg,
        slant_m: c.slant_m,
        shape,
        hop_s: img.hop_s,
    })
}

/// Every pair's map: aircraft of `band` (`Band::wants`), each track, each cell; δ limited to
/// `range` (the loaded traffic's span, never just the prior). Parallel; deterministic order
/// (aircraft, track, cell).
pub fn pair_maps(
    img: &GuideImage,
    band: Band,
    inp: &Inputs,
    frame: &LocalFrame,
    cells: &[Cell],
    range: (f64, f64),
    cell_m: f64,
) -> Vec<PairMap> {
    let jobs: Vec<(usize, usize, usize)> = inp
        .aircraft
        .iter()
        .enumerate()
        .filter(|(_, a)| band.wants(a.class))
        .flat_map(|(ai, a)| {
            (0..a.tracks.len()).flat_map(move |ti| (0..cells.len()).map(move |ci| (ai, ti, ci)))
        })
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = jobs.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .filter_map(|&(ai, ti, ci)| {
                            pair_map_with(
                                img,
                                band,
                                inp,
                                frame,
                                (ai, ti, ci),
                                cells[ci],
                                range,
                                cell_m,
                                super::joint::floor_cap(band).0,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::atmos::Atmosphere;
    use crate::grid::cell_at;
    use crate::testutil::{Fixture, fixture};
    use gm_doppler::audio::Audio;
    use gm_doppler::audio::guide::{coarsen, guide_image};
    use gm_sources::area::Area;

    pub(crate) fn area(s: &str) -> Area {
        Area::parse(s, &mut |_| unreachable!("no geocoding")).unwrap()
    }

    /// The coarse prop image of `f`'s audio.
    pub(crate) fn coarse(audio: &Audio) -> GuideImage {
        coarsen(
            &guide_image(audio),
            crate::params::TF_COARSE_HOP_S,
            crate::params::TF_COARSE_BPO,
        )
    }

    /// Maps of every aircraft of `f` at the truth cell over δ ∈ `range`, keeping all steps.
    fn truth_maps(f: &Fixture, img: &GuideImage, range: (f64, f64)) -> Vec<PairMap> {
        let a = area("40.8765,-73.5250~6km");
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = Inputs {
            events: &[],
            aircraft: &f.aircraft,
            area: &a,
            when: f.when,
            duration_s: f.duration_s,
            ground: &ground,
            atmos: &atmos,
            noise: None,
            decoys: &[],
        };
        let frame = LocalFrame::new(f.truth.observer);
        let cell = cell_at(&frame, 0.0, 0.0, &ground);
        (0..f.aircraft.len())
            .filter_map(|ai| {
                pair_map_with(
                    img,
                    Band::Prop,
                    &inp,
                    &frame,
                    (ai, 0, 0),
                    cell,
                    range,
                    crate::params::TF_CELL_M,
                    f64::NEG_INFINITY,
                )
            })
            .collect()
    }

    #[test]
    fn slack_covers_off_cell_error() {
        assert_eq!(slack_s(1000.0, 70.0), 11.0);
        assert_eq!(slack_s(1000.0, 5.0), 36.0); // floored at 20 m/s
    }

    #[test]
    fn pair_map_peaks_at_the_true_delta() {
        let f = fixture("suite/crossing.toml", |s| s);
        let img = coarse(&f.audio);
        let maps = truth_maps(&f, &img, (-600.0, 600.0));
        assert_eq!(maps.len(), 2);
        for m in &maps {
            let best = m
                .hits
                .iter()
                .max_by(|a, b| a.z.total_cmp(&b.z))
                .expect("scored");
            let slack = slack_s(crate::params::TF_CELL_M, m.v_mps);
            let d = m.delta_of(best.step);
            // Pooling makes a plateau ±slack wide; its lag points at the unpooled peak.
            assert!(d.abs() <= slack + 2.0, "best δ {d} (slack {slack})");
            let peak = d + f64::from(best.lag) * TF_DELTA_STEP_S;
            assert!(peak.abs() <= 2.0, "unpooled peak at δ {peak}");
            assert!(best.z >= 8.0, "z {}", best.z);
            assert!(best.rise_ok, "z {} zr {}", best.z, best.zr);
            assert!(m.hit_at(d).is_some());
        }
    }

    #[test]
    fn rising_curve_fails_rise_ok() {
        // Time-reversed: the pass's tone rises; the real tracks predict it falling.
        let f = fixture("one_pass.toml", |s| s);
        let mut samples = f.audio.samples.clone();
        samples.reverse();
        let rev = Audio::new(f.audio.sample_rate, samples);
        let img = coarse(&rev);
        let maps = truth_maps(&f, &img, (-3600.0, 3600.0));
        let m = &maps[0];
        let zr = m
            .hits
            .iter()
            .filter(|h| !h.zr.is_nan())
            .map(|h| h.zr)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            zr >= 8.0,
            "the rising construction should find the reversed pass: {zr}"
        );
        assert!(
            m.hits.iter().all(|h| !(h.rise_ok && h.z >= 8.0)),
            "a strong falling score passed the rise check on a rising tone"
        );
    }
}
