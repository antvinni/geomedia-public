//! Guided curve scoring: for each real aircraft near the area, is its predicted Doppler curve in
//! the spectrogram? The image's brightness is summed under the exact emission-time curve over a
//! sweep of f₀, clock offset δ and cell; a curve that stands far above the f₀ sweep's own spread
//! is measured along, fitted, and becomes a guided event — evidence the normal search then
//! weighs like any other (matching, clock, silence, verdict). Nothing is located by guidance
//! alone.
//!
//! On the log-frequency guide image a curve f₀·k(t) is f₀'s bin shifted by GUIDE_BPO·log₂ k(t),
//! so one pass over the frames sums every f₀ at once (colsto's "brightness under the modelled
//! curve", swept over f₀ and CPA time).

use gm_core::{Geodetic, LocalFrame};
use gm_doppler::audio::guide::{GuideImage, measure_along};
use gm_doppler::audio::{Event, EventParams, event_from_points};
use gm_doppler::physics::Track;

use crate::grid::{Candidate, Cell, Inputs, cell_at, grid};
use crate::params::{
    GUIDE_CELL_M, GUIDE_CLIP, GUIDE_COVERAGE_MIN, GUIDE_DELTA_STEP_S, GUIDE_F0, GUIDE_MAX_SLANT_M,
    GUIDE_MIN_LEVEL, GUIDE_MIN_POINTS, GUIDE_NULL_MARGIN, GUIDE_SPAN_S, GUIDE_TOL, GUIDE_Z_MIN,
};

/// The best curve for one (track, cell): its robust z over the f₀ sweep, where it lies, and how
/// much of it is lit.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scored {
    pub(crate) z: f64,
    pub(crate) delta: f64,
    /// f₀'s log bin.
    pub(crate) q0: usize,
    /// CPA arrival on the track clock.
    pub(crate) t_arrive: f64,
    pub(crate) coverage: f64,
    pub(crate) mean_db: f64,
    pub(crate) cell: Cell,
}

/// The fixed sweep for one image and prior: frame step `step` (frames per δ step), the δ values
/// (track clock − audio time), and the f₀ bins.
pub(crate) struct Sweep {
    pub(crate) deltas: Vec<f64>,
    pub(crate) step: usize,
    pub(crate) q_lo: usize,
    pub(crate) q_hi: usize,
    pub(crate) max_slant_m: f64,
    pub(crate) band: Band,
}

/// Which tones a guided pass looks for, and in whom: propellers and rotors' blade-passing
/// combs (f₀ 40–400 Hz, any range to GUIDE_MAX_SLANT_M), or jets' airframe and engine tones
/// (f₀ 300–3000 Hz on the jet guide image; they carry only ~3 km — B §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Prop,
    Jet,
}

impl Band {
    fn f0(self) -> (f64, f64) {
        match self {
            Band::Prop => GUIDE_F0,
            Band::Jet => crate::params::JET_F0,
        }
    }
    /// The coarse cell grid: fine enough that a pass within this band's slant limit is sampled
    /// by several cells (jets: 3 km, so 1 km cells).
    pub(crate) fn cell_m(self) -> f64 {
        match self {
            Band::Prop => GUIDE_CELL_M,
            Band::Jet => crate::params::JET_CELL_M,
        }
    }
    pub(crate) fn max_slant_m(self) -> f64 {
        match self {
            Band::Prop => GUIDE_MAX_SLANT_M,
            Band::Jet => crate::params::JET_MAX_SLANT_M,
        }
    }
    /// Whether this band's tones come from aircraft of class `c`.
    pub fn wants(self, c: crate::grid::AircraftClass) -> bool {
        (c == crate::grid::AircraftClass::Jet) == (self == Band::Jet)
    }
}

impl Sweep {
    /// δ from `range.0` to `range.1` (track clock − audio time).
    fn new(img: &GuideImage, range: (f64, f64), band: Band) -> Option<Sweep> {
        Sweep::with_step(img, range, band, GUIDE_DELTA_STEP_S)
    }

    /// `new` with δ every `step_s` (rounded to whole frames).
    pub(crate) fn with_step(
        img: &GuideImage,
        range: (f64, f64),
        band: Band,
        step_s: f64,
    ) -> Option<Sweep> {
        let hop = img.hop_s;
        if img.times.is_empty() || hop.is_nan() || hop <= 0.0 {
            return None;
        }
        // δ on a whole number of frames, so every δ reuses one factor grid.
        let step = (step_s / hop).round().max(1.0) as usize;
        let (lo, hi) = range;
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return None;
        }
        let n = ((hi - lo) / (step as f64 * hop)).floor() as usize;
        let deltas = (0..=n).map(|i| lo + (i * step) as f64 * hop).collect();
        let nb = img.n_bins();
        let (f_a, f_b) = band.f0();
        let q_lo = img.bin_of(f_a).ceil().max(0.0) as usize;
        let q_hi = (img.bin_of(f_b).floor().max(0.0) as usize).min(nb.saturating_sub(1));
        (q_lo < q_hi).then_some(Sweep {
            deltas,
            step,
            q_lo,
            q_hi,
            max_slant_m: band.max_slant_m(),
            band,
        })
    }
}

/// Median and normalised MAD.
fn robust(v: &[f32]) -> (f64, f64) {
    let mut s: Vec<f32> = v.to_vec();
    s.sort_by(f32::total_cmp);
    let med = f64::from(s[s.len() / 2]);
    let mut d: Vec<f64> = s.iter().map(|&x| (f64::from(x) - med).abs()).collect();
    d.sort_by(f64::total_cmp);
    (med, 1.4826 * d[d.len() / 2])
}

/// Score `track` heard at `cell` over the whole δ × f₀ sweep.
fn score(
    img: &GuideImage,
    sw: &Sweep,
    deltas: &[f64],
    inp: &Inputs,
    track: &Track,
    cell: Cell,
) -> Option<Scored> {
    let (ta, v) = curve_z(img, sw, deltas, inp, track, cell, false, GUIDE_MIN_POINTS)?;
    // The first δ reaching the highest z.
    let mut best: Option<(usize, DeltaZ)> = None;
    for (i, d) in v.iter().enumerate() {
        if let Some(d) = d
            && best.is_none_or(|(_, b)| d.z > b.z)
        {
            best = Some((i, *d));
        }
    }
    let (i, d) = best?;
    Some(Scored {
        z: d.z,
        delta: deltas[i],
        q0: d.q0,
        t_arrive: ta,
        coverage: d.coverage,
        mean_db: f64::from(d.bmax) / d.n as f64,
        cell,
    })
}

/// One δ's curve sum: the robust z over the f₀ sweep, f₀'s log bin, the summed value at that
/// bin and the frames it used.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeltaZ {
    pub z: f64,
    pub q0: usize,
    pub bmax: f32,
    pub n: usize,
    /// Fraction of the used frames lit (level ≥ GUIDE_MIN_LEVEL) at f₀'s bin.
    pub coverage: f64,
}

/// The curve sum of `track` heard at `cell` for every δ of `deltas` (spaced `sw.step` frames):
/// one entry per δ, `None` where too few frames were usable. `rising`: each frame takes its
/// mirror time's shift around the CPA arrival (the impossible rising sweep — `rise_z`'s
/// construction). Also returns the arrival on the track clock. At least `min_points` frames.
#[allow(clippy::too_many_arguments)]
pub(crate) fn curve_z(
    img: &GuideImage,
    sw: &Sweep,
    deltas: &[f64],
    inp: &Inputs,
    track: &Track,
    cell: Cell,
    rising: bool,
    min_points: usize,
) -> Option<(f64, Vec<Option<DeltaZ>>)> {
    let c = inp.atmos.cpa(track, cell.ecef)?;
    if c.slant_m > sw.max_slant_m {
        return None;
    }
    let (d_lo, d_hi) = (*deltas.first()?, *deltas.last()?);
    let ta = c.t_arrive;
    if ta - d_hi > inp.duration_s || ta - d_lo < 0.0 {
        return None;
    }
    let (t0, hop) = (*img.times.first()?, img.hop_s);
    let nf = img.times.len();
    let frame = |t: f64| ((t - t0) / hop).round();
    let j0 = frame(ta - GUIDE_SPAN_S - d_hi).max(0.0) as usize;
    let j1 = (frame(ta + GUIDE_SPAN_S - d_lo).max(0.0) as usize).min(nf - 1);
    if j0 > j1 {
        return None;
    }
    let m_len = (j1 - j0) + sw.step * (deltas.len() - 1) + 1;
    let taus: Vec<f64> = (0..m_len)
        .map(|m| img.times[j0] + d_lo + m as f64 * hop)
        .collect();
    let eval: Vec<f64> = if rising {
        taus.iter().map(|t| 2.0 * ta - t).collect()
    } else {
        taus.clone()
    };
    let (lo_t, hi_t) = eval
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| {
            (a.min(t), b.max(t))
        });
    let w = track.window(lo_t - 120.0, hi_t + 60.0)?;
    let shifts: Vec<Option<i64>> = inp
        .atmos
        .factors(&w, cell.ecef, &eval)
        .into_iter()
        .map(|k| {
            k.filter(|k| *k > 0.0)
                .map(|k| (img.bpo * k.log2()).round() as i64)
        })
        .collect();
    let nb = img.n_bins() as i64;
    let width = sw.q_hi - sw.q_lo + 1;
    let mut b = vec![0f32; width];
    let mut out = Vec::with_capacity(deltas.len());
    let mut used: Vec<(usize, i64)> = Vec::new();
    for (i, &delta) in deltas.iter().enumerate() {
        b.iter_mut().for_each(|x| *x = 0.0);
        used.clear();
        for j in j0..=j1 {
            if (img.times[j] + delta - ta).abs() > GUIDE_SPAN_S {
                continue;
            }
            let Some(s) = shifts[(j - j0) + i * sw.step] else {
                continue;
            };
            let lo = (sw.q_lo as i64).max(-s);
            let hi = (sw.q_hi as i64).min(nb - 1 - s);
            if lo > hi {
                continue;
            }
            let row = img.score_row(j);
            let src = &row[(lo + s) as usize..=(hi + s) as usize];
            let dst = &mut b[(lo - sw.q_lo as i64) as usize..=(hi - sw.q_lo as i64) as usize];
            // Saturated: a curve must be lit consistently, not cross one bright ridge.
            for (d, x) in dst.iter_mut().zip(src) {
                *d += x.clamp(-GUIDE_CLIP, GUIDE_CLIP);
            }
            used.push((j, s));
        }
        let n = used.len();
        if n < min_points {
            out.push(None);
            continue;
        }
        let (med, mad) = robust(&b);
        let Some((k, &bmax)) = b.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)) else {
            out.push(None);
            continue;
        };
        if mad.is_nan() || mad <= 0.0 {
            out.push(None);
            continue;
        }
        let q0 = sw.q_lo + k;
        let lit = used
            .iter()
            .filter(|&&(j, s)| {
                img.level[j]
                    .get((q0 as i64 + s) as usize)
                    .is_some_and(|&x| x >= GUIDE_MIN_LEVEL)
            })
            .count();
        out.push(Some(DeltaZ {
            z: (f64::from(bmax) - med) / mad,
            q0,
            bmax,
            n,
            coverage: lit as f64 / n as f64,
        }));
    }
    Some((ta, out))
}

/// The same-time rising-curve null of an accepted curve: over the very frames and f₀ sweep it was
/// scored on, each frame takes the shift of its mirror frame around the CPA arrival — the
/// impossible rising sweep (a passing source only falls). Its robust z. Clutter that a falling
/// curve explains, a rising one explains about as well (Oyster Bay's 65 Hz feature: +1.2…+1.7);
/// a real pass beats it clearly (+3.4…+17). Independent of the guide's own geometry.
fn rise_z(img: &GuideImage, sw: &Sweep, inp: &Inputs, track: &Track, s: &Scored) -> Option<f64> {
    let (t0, hop) = (img.times[0], img.hop_s);
    let n = img.times.len() as i64;
    let jc = ((s.t_arrive - s.delta - t0) / hop).round() as i64;
    let half = (GUIDE_SPAN_S / hop).round() as i64;
    // A window symmetric about jc, clipped symmetrically at the image's edges.
    let r = half.min(jc).min(n - 1 - jc);
    if r <= 0 {
        return None;
    }
    let js: Vec<i64> = (jc - r..=jc + r).collect();
    let taus: Vec<f64> = js
        .iter()
        .map(|&j| img.times[j as usize] + s.delta)
        .collect();
    let w = track.window(taus[0] - 120.0, taus[taus.len() - 1] + 60.0)?;
    let shifts: Vec<Option<i64>> = inp
        .atmos
        .factors(&w, s.cell.ecef, &taus)
        .into_iter()
        .map(|k| {
            k.filter(|k| *k > 0.0)
                .map(|k| (img.bpo * k.log2()).round() as i64)
        })
        .collect();
    let nb = img.n_bins() as i64;
    let width = sw.q_hi - sw.q_lo + 1;
    let mut b = vec![0f32; width];
    let mut used = 0;
    for (pos, &j) in js.iter().enumerate() {
        let Some(sh) = shifts[js.len() - 1 - pos] else {
            continue;
        };
        let lo = (sw.q_lo as i64).max(-sh);
        let hi = (sw.q_hi as i64).min(nb - 1 - sh);
        if lo > hi {
            continue;
        }
        let row = img.score_row(j as usize);
        let src = &row[(lo + sh) as usize..=(hi + sh) as usize];
        let dst = &mut b[(lo - sw.q_lo as i64) as usize..=(hi - sw.q_lo as i64) as usize];
        for (d, x) in dst.iter_mut().zip(src) {
            *d += x.clamp(-GUIDE_CLIP, GUIDE_CLIP);
        }
        used += 1;
    }
    if used < GUIDE_MIN_POINTS {
        return None;
    }
    let (med, mad) = robust(&b);
    let bmax = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mad > 0.0).then(|| (f64::from(bmax) - med) / mad)
}

/// `rise_z` maximised over the sweep's δ values and the 5×5 cells (a quarter of the band's coarse
/// step apart) around the chosen one — the look-elsewhere the falling score had nearby.
fn rise_search(
    img: &GuideImage,
    sw: &Sweep,
    inp: &Inputs,
    frame: &LocalFrame,
    tr: &Track,
    s: &Scored,
) -> Option<f64> {
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let step = sw.band.cell_m() / 4.0;
    let mut top: Option<f64> = None;
    for i in -2..=2 {
        for k in -2..=2 {
            let cell = cell_at(
                frame,
                s.cell.east + f64::from(i) * step,
                s.cell.north + f64::from(k) * step,
                &ground,
            );
            let Some(c) = inp.atmos.cpa(tr, cell.ecef) else {
                continue;
            };
            for &delta in &sw.deltas {
                let probe = Scored {
                    delta,
                    t_arrive: c.t_arrive,
                    cell,
                    ..*s
                };
                if let Some(z) = rise_z(img, sw, inp, tr, &probe) {
                    top = Some(top.map_or(z, |t: f64| t.max(z)));
                }
            }
        }
    }
    top
}

/// Drop every guided event that another aircraft's guided event claims too (same pass, or one
/// riding the other's ridge) — across bands as within one.
pub fn drop_contested(events: Vec<Event>) -> Vec<Event> {
    let clash = |a: &Event, b: &Event| {
        a.guide_icao != b.guide_icao && (same_pass(a, b) || rides_on(a, b) || rides_on(b, a))
    };
    let bad: Vec<bool> = events
        .iter()
        .map(|e| events.iter().any(|o| clash(e, o)))
        .collect();
    events
        .into_iter()
        .zip(bad)
        .filter_map(|(e, b)| (!b).then_some(e))
        .collect()
}

/// The best curve of one aircraft: a coarse cell grid over the area, then two finer grids
/// around the best cell (the δ prior is only seconds wide, so a coarse cell's CPA time can be
/// further off than δ can absorb).
fn best_for(
    img: &GuideImage,
    sw: &Sweep,
    inp: &Inputs,
    frame: &LocalFrame,
    coarse: &[Cell],
    track: &Track,
) -> Option<(Scored, Track)> {
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let mut best = coarse
        .iter()
        .filter_map(|&c| score(img, sw, &sw.deltas, inp, track, c))
        .max_by(|a, b| a.z.total_cmp(&b.z))?;
    let mut step = sw.band.cell_m();
    for _ in 0..2 {
        step /= 4.0;
        let (e0, n0) = (best.cell.east, best.cell.north);
        for i in -6..=6 {
            for k in -6..=6 {
                let c = cell_at(
                    frame,
                    e0 + f64::from(i) * step,
                    n0 + f64::from(k) * step,
                    &ground,
                );
                if let Some(s) = score(img, sw, &sw.deltas, inp, track, c)
                    && s.z > best.z
                {
                    best = s;
                }
            }
        }
    }
    Some((best, track.clone()))
}

/// Clock shifts for the null: whole multiples of GUIDE_NULL_STEP_S beyond the prior's span
/// plus GUIDE_NULL_MIN_SHIFT_S, keeping the shifted arrival in the recording and its span off
/// every pass already found.
pub(crate) fn null_shifts(sw: &Sweep, inp: &Inputs, s: &Scored) -> Option<Vec<f64>> {
    use crate::params::{GUIDE_NULL_MIN_SHIFT_S, GUIDE_NULL_STEP_S};
    let span = sw.deltas.last()? - sw.deltas.first()?;
    let n = ((inp.duration_s + span) / GUIDE_NULL_STEP_S).ceil() as i64;
    Some(
        (-n..=n)
            .map(|k| k as f64 * GUIDE_NULL_STEP_S)
            .filter(|sh| sh.abs() >= span + GUIDE_NULL_MIN_SHIFT_S)
            .filter(|sh| {
                // The shifted curve's arrival must still fall in the recording, and its span must
                // not overlap a pass already found: those times are explained by other aircraft,
                // not chance.
                let a = s.t_arrive - s.delta - sh;
                let (lo, hi) = (a - GUIDE_SPAN_S, a + GUIDE_SPAN_S);
                a >= 0.0
                    && a <= inp.duration_s
                    && inp.events.iter().all(|e| e.t_end < lo || e.t_start > hi)
            })
            .collect(),
    )
}

/// Decoy passes for the null of curve `s` (aircraft `icao`): every track of `inp.decoys` the
/// band wants, other than `icao` and every prior-window candidate (either could really be in the
/// audio), audible at the cell, as (track, δ offset). Adding the offset to the sweep's δ values
/// moves the decoy's CPA arrival onto the tested curve's audio time: `score()` uses frame j iff
/// |times[j] + δ − t_arrive| ≤ GUIDE_SPAN_S and gates on t_arrive − δ, so with
/// δ' = δ + (t_arrive_decoy − s.t_arrive) the decoy is read over exactly the frames the tested
/// curve was. At most GUIDE_NULL_MAX_SHIFTS, evenly spread over the decoys' arrival times.
pub(crate) fn decoy_shifts<'t>(
    sw: &Sweep,
    inp: &Inputs<'t>,
    icao: &str,
    s: &Scored,
) -> Vec<(&'t Track, f64)> {
    use crate::params::GUIDE_NULL_MAX_SHIFTS;
    let decoys: &'t [Candidate] = inp.decoys;
    let mut v: Vec<(&'t Track, f64)> = decoys
        .iter()
        .filter(|d| {
            sw.band.wants(d.class)
                && d.icao != icao
                && inp.aircraft.iter().all(|a| a.icao != d.icao)
        })
        .flat_map(|d| d.tracks.iter())
        .filter_map(|tr| {
            let c = inp.atmos.cpa(tr, s.cell.ecef)?;
            (c.slant_m <= sw.max_slant_m).then_some((tr, c.t_arrive - s.t_arrive))
        })
        .collect();
    v.sort_by(|x, y| x.1.total_cmp(&y.1));
    if v.len() > GUIDE_NULL_MAX_SHIFTS {
        let k = v.len();
        v = (0..GUIDE_NULL_MAX_SHIFTS)
            .map(|i| v[i * k / GUIDE_NULL_MAX_SHIFTS])
            .collect();
    }
    v
}

/// A curve's null: the largest z it reaches, and how many values it took.
#[derive(Clone, Copy, Debug)]
struct NullZ {
    z: f64,
    shifts: usize,
    decoys: usize,
}

/// The null of an accepted curve. Real recordings are full of curve-like structure (traffic,
/// engines, other aircraft) that a noise-calibrated z can't know about, so the curve is compared
/// with (a) the same curve at the same cell, its δ sweep moved by ≥ GUIDE_NULL_MIN_SHIFT_S (whole
/// sweeps, every GUIDE_NULL_STEP_S) to other times of this recording, and (b) decoy aircraft's
/// passes re-timed onto it (`decoy_shifts`) — other real shapes over the same frames, which a
/// clip too short for (a) still has. At each value the cells around the chosen one are searched
/// too (the real z was maximised over cells); the largest z found.
///
/// Shifts start beyond the whole prior (or they re-score the tested curve itself), at most
/// GUIDE_NULL_MAX_SHIFTS of them evenly spread, none over a pass already found. `None` — no
/// usable null — when fewer than GUIDE_NULL_MIN_SHIFTS values (shifts and decoys together) could
/// be scored: then the curve can't be checked.
///
/// Scoring stops as soon as a value exceeds `reject_above` (the null z at which the caller
/// rejects the curve): the decision can no longer change, so the remaining values are not
/// scored and the returned z is only a lower bound. Pass `f64::INFINITY` for the full null.
#[allow(clippy::too_many_arguments)]
fn null_z(
    img: &GuideImage,
    sw: &Sweep,
    inp: &Inputs,
    frame: &LocalFrame,
    track: &Track,
    icao: &str,
    s: &Scored,
    reject_above: f64,
) -> Option<NullZ> {
    use crate::params::{GUIDE_NULL_MAX_SHIFTS, GUIDE_NULL_MIN_SHIFTS};
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let mut shifts = null_shifts(sw, inp, s).unwrap_or_default();
    if shifts.len() > GUIDE_NULL_MAX_SHIFTS {
        let k = shifts.len();
        shifts = (0..GUIDE_NULL_MAX_SHIFTS)
            .map(|i| shifts[i * k / GUIDE_NULL_MAX_SHIFTS])
            .collect();
    }
    let decoys = decoy_shifts(sw, inp, icao, s);
    if shifts.len() + decoys.len() < GUIDE_NULL_MIN_SHIFTS {
        return None;
    }
    let step = sw.band.cell_m() / 4.0;
    let cells: Vec<Cell> = (-3..=3)
        .flat_map(|i| (-3..=3).map(move |k| (i, k)))
        .map(|(i, k)| {
            cell_at(
                frame,
                s.cell.east + f64::from(i) * step,
                s.cell.north + f64::from(k) * step,
                &ground,
            )
        })
        .collect();
    let best = |tr: &Track, sh: f64| {
        let d: Vec<f64> = sw.deltas.iter().map(|d| d + sh).collect();
        cells
            .iter()
            .filter_map(|&c| score(img, sw, &d, inp, tr, c).map(|n| n.z))
            .max_by(f64::total_cmp)
    };
    let mut n = NullZ {
        z: f64::NEG_INFINITY,
        shifts: 0,
        decoys: 0,
    };
    for sh in shifts {
        if let Some(z) = best(track, sh) {
            n.shifts += 1;
            n.z = n.z.max(z);
            if n.z > reject_above {
                return Some(n);
            }
        }
    }
    for (tr, sh) in decoys {
        if let Some(z) = best(tr, sh) {
            n.decoys += 1;
            n.z = n.z.max(z);
            if n.z > reject_above {
                return Some(n);
            }
        }
    }
    (n.shifts + n.decoys >= GUIDE_NULL_MIN_SHIFTS).then_some(n)
}

/// The guided event for an accepted curve: points measured along it, fitted; `None` when they
/// don't make a pass consistent with the curve.
fn measure(img: &GuideImage, inp: &Inputs, s: &Scored, track: &Track, icao: &str) -> Option<Event> {
    let frames: Vec<usize> = (0..img.times.len())
        .filter(|&j| (img.times[j] + s.delta - s.t_arrive).abs() <= GUIDE_SPAN_S)
        .collect();
    let taus: Vec<f64> = frames.iter().map(|&j| img.times[j] + s.delta).collect();
    let w = track.window(taus.first()? - 120.0, taus.last()? + 60.0)?;
    let f0 = img.freq_of(s.q0 as f64);
    let curve: Vec<(usize, f64)> = frames
        .iter()
        .zip(inp.atmos.factors(&w, s.cell.ecef, &taus))
        .filter_map(|(&j, k)| k.map(|k| (j, f0 * k)))
        .collect();
    let pts = measure_along(img, &curve, GUIDE_TOL, GUIDE_MIN_LEVEL);
    if pts.len() < GUIDE_MIN_POINTS {
        return None;
    }
    if centred_fraction(img, &curve, &pts) < crate::params::GUIDE_CENTRED_MIN {
        return None;
    }
    let mut e = event_from_points(pts, s.mean_db as f32, &EventParams::default())?;
    // The fitted pass must be the curve's own: CPA within seconds of the predicted arrival.
    if (e.fit.tc - (s.t_arrive - s.delta)).abs() > 10.0 {
        return None;
    }
    e.guide_icao = Some(icao.to_string());
    e.guide_z = Some(s.z);
    Some(e)
}

/// Guide-independent check of points measured along `curve` (frame, Hz): the fraction within
/// half a log bin of the curve. A real ridge puts the peaks on the curve; noise picks spread
/// evenly over the ±1-bin window (≈ 1/3 within half a bin).
pub(crate) fn centred_fraction(
    img: &GuideImage,
    curve: &[(usize, f64)],
    pts: &[(f64, f64)],
) -> f64 {
    if pts.is_empty() {
        return 0.0;
    }
    let pred: std::collections::HashMap<u64, f64> = curve
        .iter()
        .map(|&(j, f)| (img.times[j].to_bits(), f))
        .collect();
    let centred = pts
        .iter()
        .filter(|&&(t, f)| {
            pred.get(&t.to_bits())
                .is_some_and(|&p| (img.bin_of(f) - img.bin_of(p)).abs() <= 0.5)
        })
        .count();
    centred as f64 / pts.len() as f64
}

/// Does guided event `g` ride on `e`'s ridge: ≥ GUIDE_RIDE_FRAC of its points within half a
/// frame of one of `e`'s, at a small-integer tone ratio (either way) within 0.5 %?
fn rides_on(g: &Event, e: &Event) -> bool {
    use crate::params::GUIDE_RIDE_FRAC;
    if g.points.is_empty() || e.points.is_empty() {
        return false;
    }
    let harmonic = |a: f64, b: f64| {
        let r = a.max(b) / a.min(b);
        r.is_finite() && r.round() <= 4.0 && (r - r.round()).abs() / r <= 0.005
    };
    let on = g
        .points
        .iter()
        .filter(|&&(t, f)| {
            let i = e.points.partition_point(|p| p.0 < t - 0.13);
            e.points[i..]
                .iter()
                .take_while(|p| p.0 <= t + 0.13)
                .any(|p| f > 0.0 && p.1 > 0.0 && harmonic(f, p.1))
        })
        .count();
    on as f64 >= GUIDE_RIDE_FRAC * g.points.len() as f64
}

/// Is `b` the same pass as `a` (overlapping in time, tones at a small-integer ratio)?
fn same_pass(a: &Event, b: &Event) -> bool {
    let overlap = a.t_start <= b.t_end && b.t_start <= a.t_end;
    let (x, y) = (a.fit.f0.max(b.fit.f0), a.fit.f0.min(b.fit.f0));
    if !(overlap && y > 0.0) {
        return false;
    }
    let r = x / y;
    r.round() <= 4.0 && (r - r.round()).abs() / r <= 0.01
}

/// One accepted curve's event: the gates (z, coverage), the time-shifted null (abstain without
/// one), then points measured along it.
/// What every candidate of one guided pass shares.
struct Ctx<'a> {
    img: &'a GuideImage,
    sw: &'a Sweep,
    inp: &'a Inputs<'a>,
    frame: &'a LocalFrame,
}

fn accept(c: &Ctx, s: &Scored, tr: &Track, icao: &str, z_min: f64) -> Option<Event> {
    let Ctx {
        img,
        sw,
        inp,
        frame,
    } = *c;
    if s.z < z_min || s.coverage < GUIDE_COVERAGE_MIN {
        return None;
    }
    // The same curve at other times of this recording: a real pass must stand clear of it;
    // with no usable null, abstain.
    let null = null_z(img, sw, inp, frame, tr, icao, s, s.z - GUIDE_NULL_MARGIN)?;
    if s.z < null.z + GUIDE_NULL_MARGIN {
        return None;
    }
    // …and the same curve run backwards in time, at the same moment, searched as widely as
    // the falling score was nearby (every δ, the cells around): with no usable rising null,
    // abstain.
    let rise = rise_search(img, sw, inp, frame, tr, s)?;
    if s.z < rise + crate::params::GUIDE_RISE_MARGIN {
        return None;
    }
    let mut e = measure(img, inp, s, tr, icao)?;
    e.guide_null_z = Some(null.z);
    e.guide_null_n = Some((null.shifts, null.decoys));
    Some(e)
}

/// Run `best` for every audible aircraft's tracks (except `skip`) in parallel, accept, dedup
/// against `inp.events` and each other, and number after the existing events.
fn propose(
    img: &GuideImage,
    sw: &Sweep,
    inp: &Inputs,
    frame: &LocalFrame,
    skip: &[String],
    z_min: f64,
    best: &(dyn Fn(&Track) -> Option<(Scored, Track)> + Sync),
) -> Vec<Event> {
    let jobs: Vec<(&str, &Track)> = inp
        .aircraft
        .iter()
        .filter(|a| sw.band.wants(a.class) && !skip.contains(&a.icao))
        .flat_map(|a| a.tracks.iter().map(move |t| (a.icao.as_str(), t)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = jobs.len().div_ceil(threads).max(1);
    let mut found: Vec<Event> = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .filter_map(|&(icao, tr)| {
                            let (s, tr) = best(tr)?;
                            accept(
                                &Ctx {
                                    img,
                                    sw,
                                    inp,
                                    frame,
                                },
                                &s,
                                &tr,
                                icao,
                                z_min,
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
    });
    // Strongest first; drop passes already found, freely or by a stronger guide.
    found.sort_by(|a, b| {
        b.guide_z
            .unwrap_or(0.0)
            .total_cmp(&a.guide_z.unwrap_or(0.0))
    });
    // Two aircraft claiming one feature: the attribution is ambiguous — drop every claimant
    // (keeping the strongest would choose by noise; Oyster Bay had three aircraft on one 65 Hz
    // clutter feature).
    let clash = |a: &Event, b: &Event| {
        a.guide_icao != b.guide_icao && (same_pass(a, b) || rides_on(a, b) || rides_on(b, a))
    };
    let contested: Vec<bool> = found
        .iter()
        .map(|e| found.iter().any(|o| clash(e, o)))
        .collect();
    let mut out: Vec<Event> = Vec::new();
    for (e, bad) in found.into_iter().zip(contested) {
        if !bad
            && inp
                .events
                .iter()
                .chain(&out)
                .all(|x| !same_pass(x, &e) && !rides_on(&e, x))
        {
            out.push(e);
        }
    }
    out.sort_by(|a, b| a.fit.tc.total_cmp(&b.fit.tc));
    for (i, e) in out.iter_mut().enumerate() {
        e.id = inp.events.len() + i + 1;
    }
    out
}

fn area_frame(inp: &Inputs) -> LocalFrame {
    let (clat, clon) = inp.area.centre();
    LocalFrame::new(Geodetic {
        lat_deg: clat,
        lon_deg: clon,
        h_m: 0.0,
    })
}

/// Guided events for the aircraft in `inp` (whose `events` are the ones already found: a pass
/// they cover is not proposed again). Each records the aircraft that proposed it, its z and
/// its null.
pub fn guided_events(img: &GuideImage, inp: &Inputs, band: Band) -> Vec<Event> {
    let Some(sw) = Sweep::new(img, (-inp.when.minus_s - 1.0, inp.when.plus_s + 1.0), band) else {
        return Vec::new();
    };
    let frame = area_frame(inp);
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let coarse = grid(inp.area, &frame, band.cell_m(), &ground);
    let best = |tr: &Track| best_for(img, &sw, inp, &frame, &coarse, tr);
    propose(img, &sw, inp, &frame, &[], GUIDE_Z_MIN, &best)
}

/// Refine-only guidance: the recording is already located at plane cell (`east`, `north`) of
/// the area's frame with clock offset `delta_s`. Each audible aircraft not in `matched` is
/// tested there only — δ within ±REFINE_DELTA_S, cells within ±REFINE_RADIUS_M — far fewer
/// trials than the free sweep, so a fainter pass can clear its null and add evidence. These
/// events may only sharpen a location already found, never make one.
pub fn refine_events(
    img: &GuideImage,
    inp: &Inputs,
    band: Band,
    east: f64,
    north: f64,
    delta_s: f64,
    matched: &[String],
) -> Vec<Event> {
    use crate::params::{REFINE_DELTA_S, REFINE_RADIUS_M, REFINE_STEP_M, REFINE_Z_MIN};
    let Some(sw) = Sweep::new(
        img,
        (delta_s - REFINE_DELTA_S, delta_s + REFINE_DELTA_S),
        band,
    ) else {
        return Vec::new();
    };
    let frame = area_frame(inp);
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let n = (REFINE_RADIUS_M / REFINE_STEP_M).round() as i32;
    let cells: Vec<Cell> = (-n..=n)
        .flat_map(|i| (-n..=n).map(move |k| (i, k)))
        .map(|(i, k)| {
            cell_at(
                &frame,
                east + f64::from(i) * REFINE_STEP_M,
                north + f64::from(k) * REFINE_STEP_M,
                &ground,
            )
        })
        .collect();
    let best = |tr: &Track| {
        cells
            .iter()
            .filter_map(|&c| score(img, &sw, &sw.deltas, inp, tr, c))
            .max_by(|a, b| a.z.total_cmp(&b.z))
            .map(|s| (s, tr.clone()))
    };
    propose(img, &sw, inp, &frame, matched, REFINE_Z_MIN, &best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmos::Atmosphere;
    use crate::grid::{AircraftClass, Candidate};
    use crate::testutil::{Fixture, decoys, fixture};
    use gm_doppler::audio::guide::guide_image;
    use gm_sources::area::{Area, When};

    /// The suite's area for a scenario (centre ~3 km off the truth, as in sim-suite.toml).
    fn area(s: &str) -> Area {
        Area::parse(s, &mut |_| unreachable!("no geocoding")).unwrap()
    }

    /// The ridge stage's events only: guidance is tested as if discovery had missed what the
    /// blind template stage now finds (template events carry a z; ridge events don't).
    fn ridge_only(f: &Fixture) -> Vec<Event> {
        f.events
            .iter()
            .filter(|e| e.guided || e.guide_z.is_none())
            .cloned()
            .collect()
    }

    fn run(f: &Fixture, area: &Area, when: When) -> Vec<Event> {
        run_with(f, area, when, &[])
    }

    fn run_with(f: &Fixture, area: &Area, when: When, decoys: &[Candidate]) -> Vec<Event> {
        let ridge = ridge_only(f);
        let img = guide_image(&f.audio);
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = Inputs {
            events: &ridge,
            aircraft: &f.aircraft,
            area,
            when,
            duration_s: f.duration_s,
            ground: &ground,
            atmos: &atmos,
            noise: None,
            decoys,
        };
        guided_events(&img, &inp, Band::Prop)
    }

    #[test]
    fn curve_z_argmax_is_guided_score() {
        let (f, a) = quiet();
        let img = guide_image(&f.audio);
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
        let sw = Sweep::new(&img, (-6.0, 6.0), Band::Prop).unwrap();
        let frame = LocalFrame::new(f.truth.observer);
        let cell = cell_at(&frame, 0.0, 0.0, &ground);
        for a in &f.aircraft {
            let tr = &a.tracks[0];
            let s = score(&img, &sw, &sw.deltas, &inp, tr, cell).expect("scored");
            let (ta, v) = curve_z(
                &img,
                &sw,
                &sw.deltas,
                &inp,
                tr,
                cell,
                false,
                GUIDE_MIN_POINTS,
            )
            .unwrap();
            let (i, d) = v
                .iter()
                .enumerate()
                .filter_map(|(i, d)| d.map(|d| (i, d)))
                .fold(None::<(usize, DeltaZ)>, |b, (i, d)| match b {
                    Some((_, bd)) if bd.z >= d.z => b,
                    _ => Some((i, d)),
                })
                .unwrap();
            assert_eq!(ta, s.t_arrive);
            assert_eq!(d.z, s.z);
            assert_eq!(d.q0, s.q0);
            assert_eq!(sw.deltas[i], s.delta);
        }
    }

    /// suite/quiet, recorded for 500 s. A time-shift null needs eight places clear of every
    /// other aircraft's sound: next to the B350's pass, heard from 334 s on, the suite's 400 s
    /// leave the far pass seven.
    fn quiet() -> (Fixture, Area) {
        (
            fixture("suite/quiet.toml", |s| {
                s.replace("duration_s = 400.0", "duration_s = 500.0")
            }),
            area("40.8900,-73.4900~6km"),
        )
    }

    /// The truth event of the faint PC12 pass (bbbb01).
    fn far_pass(f: &Fixture) -> gm_sim::TruthEvent {
        f.truth
            .events
            .iter()
            .find(|e| e.icao24 == "bbbb01")
            .unwrap()
            .clone()
    }

    #[test]
    fn guided_finds_quiet_far_pass() {
        let (f, a) = quiet();
        let t = far_pass(&f);
        // The ridge stage misses it (the reason for guidance; the template stage now finds it).
        assert!(
            ridge_only(&f)
                .iter()
                .all(|e| (e.fit.tc - t.cpa_arrival_audio_s).abs() > 3.0),
            "{:#?}",
            f.events
        );
        let g = run(&f, &a, f.when);
        let e = g
            .iter()
            .find(|e| e.guide_icao.as_deref() == Some("bbbb01"))
            .unwrap_or_else(|| panic!("no guided event from bbbb01: {g:#?}"));
        assert!(e.guided && e.guide_z.is_some());
        assert!(
            (e.fit.tc - t.cpa_arrival_audio_s).abs() < 3.0,
            "tc {} vs truth {}",
            e.fit.tc,
            t.cpa_arrival_audio_s
        );
    }

    #[test]
    fn guided_event_fits_truth() {
        let (f, a) = quiet();
        let t = far_pass(&f);
        let g = run(&f, &a, f.when);
        let e = g
            .iter()
            .find(|e| e.guide_icao.as_deref() == Some("bbbb01"))
            .unwrap_or_else(|| panic!("{g:#?}"));
        assert!(
            (e.fit.v / t.speed_mps - 1.0).abs() < 0.10,
            "v {} vs {}",
            e.fit.v,
            t.speed_mps
        );
        assert!(
            (e.fit.d / t.slant_m - 1.0).abs() < 0.25,
            "d {} vs {}",
            e.fit.d,
            t.slant_m
        );
    }

    #[test]
    fn noise_gives_no_guided_events() {
        // The same aircraft fly past, but the recording is noise only.
        let f = fixture("suite/quiet.toml", |s| {
            s.replace("amplitude = 1.0", "amplitude = 0.0")
        });
        let (_, a) = quiet();
        let g = run(&f, &a, f.when);
        assert!(g.is_empty(), "{g:#?}");
    }

    #[test]
    fn wrong_clock_no_guided() {
        // The prior says 10 minutes later than the truth: tracks are timed from the prior's
        // centre, so every knot reads 600 s earlier.
        let (mut f, a) = quiet();
        for c in &mut f.aircraft {
            for t in &mut c.tracks {
                let knots = t
                    .knots
                    .iter()
                    .map(|k| gm_doppler::physics::Knot {
                        t: k.t - 600.0,
                        ..*k
                    })
                    .collect();
                *t = gm_doppler::physics::Track::new(t.id.clone(), knots).unwrap();
            }
        }
        let when = When {
            center: f.when.center.plus_seconds(600.0),
            ..f.when
        };
        let g = run(&f, &a, when);
        for t in &f.truth.events {
            assert!(
                g.iter()
                    .all(|e| (e.fit.tc - t.cpa_arrival_audio_s).abs() > 3.0),
                "{t:?}: {g:#?}"
            );
        }
    }

    #[test]
    fn no_duplicate_of_discovered() {
        // Both passes are discovered freely: guidance adds nothing on top of them.
        let f = fixture("two_props.toml", |s| s);
        assert_eq!(f.events.len(), 2, "{:#?}", f.events);
        let o = f.truth.observer;
        let a = area(&format!(
            "{:.4},{:.4}~5km",
            o.lat_deg + 0.01,
            o.lon_deg + 0.02
        ));
        let g = run(&f, &a, f.when);
        for e in &g {
            assert!(
                f.events.iter().all(|d| (d.fit.tc - e.fit.tc).abs() > 10.0),
                "duplicate {e:#?}"
            );
        }
    }

    #[test]
    fn rides_on_detects_a_shared_ridge() {
        // An existing event's ridge, and guided points that sit on it at its 2nd harmonic for
        // half their length: the guided curve rides that ridge. Elsewhere it doesn't.
        let proto = fixture("one_pass.toml", |s| s).events[0].clone();
        let ev = |pts: Vec<(f64, f64)>| Event {
            points: pts,
            ..proto.clone()
        };
        let base: Vec<(f64, f64)> = (0..80)
            .map(|i| {
                let t = f64::from(i) * 0.25;
                (t, 100.0 - t)
            })
            .collect();
        let existing = ev(base.clone());
        let riding: Vec<(f64, f64)> = base
            .iter()
            .enumerate()
            .map(|(i, &(t, f))| (t, if i % 2 == 0 { 2.0 * f } else { 2.0 * f + 7.0 }))
            .collect();
        assert!(rides_on(&ev(riding), &existing));
        let apart: Vec<(f64, f64)> = base.iter().map(|&(t, f)| (t, 1.37 * f)).collect();
        assert!(!rides_on(&ev(apart), &existing));
        let later: Vec<(f64, f64)> = base.iter().map(|&(t, f)| (t + 100.0, f)).collect();
        assert!(!rides_on(&ev(later), &existing));
    }

    /// The quiet scenario cut to 100 s: the far pass (CPA arrival ≈ 57 s) is inside, but no
    /// in-recording null shift fits.
    fn quiet_100s(edit: impl Fn(String) -> String) -> Fixture {
        fixture("suite/quiet.toml", |s| {
            edit(s.replace("duration_s = 400.0", "duration_s = 100.0"))
        })
    }

    #[test]
    fn short_recording_with_decoys_finds_the_far_pass() {
        // 100 s: no in-recording null fits; decoy tracks give one, so the far pass is checked.
        let f = quiet_100s(|s| s);
        let (_, a) = quiet();
        let g = run_with(&f, &a, f.when, &decoys(&f, 24));
        assert!(
            g.iter().any(|e| e.guide_icao.as_deref() == Some("bbbb01")),
            "{g:#?}"
        );
        assert!(g.iter().all(|e| e.guide_null_z.is_some()));
    }

    #[test]
    fn short_noise_with_decoys_gives_no_guided_events() {
        let f = quiet_100s(|s| s.replace("amplitude = 1.0", "amplitude = 0.0"));
        let (_, a) = quiet();
        let g = run_with(&f, &a, f.when, &decoys(&f, 24));
        assert!(g.is_empty(), "{g:#?}");
    }

    #[test]
    fn too_few_decoys_still_abstain() {
        // 7 decoys and no in-recording shifts: fewer than GUIDE_NULL_MIN_SHIFTS values.
        let f = quiet_100s(|s| s);
        let (_, a) = quiet();
        let g = run_with(&f, &a, f.when, &decoys(&f, 7));
        assert!(g.is_empty(), "{g:#?}");
    }

    #[test]
    fn a_decoy_with_a_candidate_icao_is_ignored() {
        // Decoys carrying a real candidate's ICAO must not count: it could be in the audio.
        let f = quiet_100s(|s| s);
        let (_, a) = quiet();
        let mut d = decoys(&f, 24);
        for x in &mut d {
            x.icao = "bbbb01".into();
        }
        let g = run_with(&f, &a, f.when, &d);
        assert!(g.is_empty(), "{g:#?}");
    }

    #[test]
    fn short_recording_without_a_null_gets_no_guided_events() {
        // 100 s: no time-shifted null fits, so the curve can't be checked against chance in this
        // recording — guidance abstains rather than trust a noise-calibrated z.
        let f = fixture("suite/quiet.toml", |s| {
            s.replace("duration_s = 400.0", "duration_s = 100.0")
        });
        let (_, a) = quiet();
        let g = run(&f, &a, f.when);
        assert!(g.is_empty(), "{g:#?}");
    }

    #[test]
    fn null_shifts_clear_the_whole_prior() {
        // ±60 s prior: every null shift lies beyond the prior's span (+ the minimum), or the
        // null would re-score the very curve under test and nothing could beat it. (A faint
        // pass may then still abstain: a wide prior means more trials — that's the null
        // doing its job.)
        let (f, a) = quiet();
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let when = When {
            minus_s: 60.0,
            plus_s: 60.0,
            ..f.when
        };
        let inp = Inputs {
            events: &[],
            aircraft: &f.aircraft,
            area: &a,
            when,
            duration_s: 1200.0,
            ground: &ground,
            atmos: &atmos,
            noise: None,
            decoys: &[],
        };
        let img = GuideImage {
            times: (0..4800).map(|j| 0.5 + f64::from(j) * 0.25).collect(),
            hop_s: 0.25,
            f_lo: gm_doppler::audio::guide::GUIDE_F_LO,
            bpo: gm_doppler::audio::guide::GUIDE_BPO,
            score: Vec::new(),
            level: vec![vec![0.0; 1000]; 4800],
        };
        let sw = Sweep::new(
            &img,
            (-inp.when.minus_s - 1.0, inp.when.plus_s + 1.0),
            Band::Prop,
        )
        .unwrap();
        let span = sw.deltas.last().unwrap() - sw.deltas.first().unwrap();
        let s = Scored {
            z: 10.0,
            delta: 0.0,
            q0: 300,
            t_arrive: 600.0,
            coverage: 0.5,
            mean_db: 1.0,
            cell: crate::grid::cell_at(&LocalFrame::new(f.truth.observer), 0.0, 0.0, &ground),
        };
        let sh = null_shifts(&sw, &inp, &s).unwrap();
        assert!(sh.len() >= crate::params::GUIDE_NULL_MIN_SHIFTS, "{sh:?}");
        assert!(
            sh.iter()
                .all(|x| x.abs() >= span + crate::params::GUIDE_NULL_MIN_SHIFT_S),
            "{sh:?}"
        );
    }

    /// Refine-only guidance at the truth cell and clock of `f`, with bbbb02 already matched.
    fn refine_at_truth(f: &Fixture, a: &Area) -> Vec<Event> {
        let ridge = ridge_only(f);
        let img = guide_image(&f.audio);
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = Inputs {
            events: &ridge,
            aircraft: &f.aircraft,
            area: a,
            when: f.when,
            duration_s: f.duration_s,
            ground: &ground,
            atmos: &atmos,
            noise: None,
            decoys: &[],
        };
        let p = area_frame(&inp).to_enu(gm_core::geodetic_to_ecef(f.truth.observer));
        refine_events(
            &img,
            &inp,
            Band::Prop,
            p.x,
            p.y,
            0.0,
            &["bbbb02".to_string()],
        )
    }

    #[test]
    fn refine_adds_the_far_pass_at_the_located_cell() {
        let (f, a) = quiet();
        let t = far_pass(&f);
        let g = refine_at_truth(&f, &a);
        assert!(
            g.iter().any(|e| e.guide_icao.as_deref() == Some("bbbb01")
                && (e.fit.tc - t.cpa_arrival_audio_s).abs() < 3.0),
            "{g:#?}"
        );
        assert!(g.iter().all(|e| e.guide_icao.as_deref() != Some("bbbb02")));
    }

    #[test]
    fn refine_on_noise_adds_nothing() {
        let f = fixture("suite/quiet.toml", |s| {
            s.replace("amplitude = 1.0", "amplitude = 0.0")
        });
        let (_, a) = quiet();
        assert!(refine_at_truth(&f, &a).is_empty());
    }

    /// jet_tone.toml with cccc01 marked as the jet it is.
    fn jet_fixture(edit: impl Fn(String) -> String) -> Fixture {
        let mut f = fixture("jet_tone.toml", edit);
        for c in &mut f.aircraft {
            if c.icao == "cccc01" {
                c.class = AircraftClass::Jet;
            }
        }
        f
    }

    fn run_band(f: &Fixture, area: &Area, band: Band) -> Vec<Event> {
        let ridge = ridge_only(f);
        use gm_doppler::audio::guide::{GuideParams, guide_image_with};
        let params = match band {
            Band::Prop => GuideParams::prop(),
            Band::Jet => GuideParams::jet(),
        };
        let img = guide_image_with(&f.audio, &params);
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = Inputs {
            events: &ridge,
            aircraft: &f.aircraft,
            area,
            when: f.when,
            duration_s: f.duration_s,
            ground: &ground,
            atmos: &atmos,
            noise: None,
            decoys: &[],
        };
        guided_events(&img, &inp, band)
    }

    #[test]
    fn jet_tone_found_guided() {
        // A jet's ~1.5 kHz airframe/engine tone at 1.2 km: invisible to free discovery (band),
        // found by the jet band of guided scoring.
        let f = jet_fixture(|s| s);
        let t = f
            .truth
            .events
            .iter()
            .find(|e| e.icao24 == "cccc01")
            .unwrap()
            .clone();
        // The ridge stage misses it (the jet-band templates now find it; guidance is tested as
        // if they hadn't).
        assert!(
            ridge_only(&f)
                .iter()
                .all(|e| (e.fit.tc - t.cpa_arrival_audio_s).abs() > 10.0),
            "the ridge stage already finds it: {:#?}",
            f.events
        );
        let a = area("40.8900,-73.4900~6km");
        let g = run_band(&f, &a, Band::Jet);
        assert!(
            g.iter().any(|e| e.guide_icao.as_deref() == Some("cccc01")
                && (e.fit.tc - t.cpa_arrival_audio_s).abs() < 3.0),
            "{g:#?}"
        );
        // The prop band never proposes a jet.
        let p = run_band(&f, &a, Band::Prop);
        assert!(
            p.iter().all(|e| e.guide_icao.as_deref() != Some("cccc01")),
            "{p:#?}"
        );
    }

    #[test]
    fn jet_far_not_proposed() {
        // The same jet 5 km off: its tones don't carry that far; not proposed.
        let f = jet_fixture(|s| s.replace("cpa_offset_m = 600.0", "cpa_offset_m = 5000.0"));
        let a = area("40.8900,-73.4900~6km");
        let g = run_band(&f, &a, Band::Jet);
        assert!(
            g.iter().all(|e| e.guide_icao.as_deref() != Some("cccc01")),
            "{g:#?}"
        );
    }

    /// A synthetic guide image for quiet's bbbb01 at its truth cell and clock: Gaussian-ish noise,
    /// plus +3 σ along its falling curve and/or along that curve mirrored in time (rising).
    fn lit_image(f: &Fixture, inp: &Inputs, falling: bool, rising: bool) -> GuideImage {
        let mut x: u64 = 99;
        let mut noise = || {
            // Sum of 4 uniforms: ~N(0, 1) after scaling.
            let mut acc = 0.0f32;
            for _ in 0..4 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                acc += (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            acc * 1.732
        };
        let n_frames = (f.duration_s / 0.25) as usize;
        let mut img = GuideImage {
            times: (0..n_frames).map(|j| 0.5 + j as f64 * 0.25).collect(),
            hop_s: 0.25,
            f_lo: gm_doppler::audio::guide::GUIDE_F_LO,
            bpo: gm_doppler::audio::guide::GUIDE_BPO,
            score: Vec::new(),
            level: Vec::new(),
        };
        img.level = (0..n_frames)
            .map(|_| (0..1100).map(|_| noise()).collect())
            .collect();
        let frame = area_frame(inp);
        let p = frame.to_enu(gm_core::geodetic_to_ecef(f.truth.observer));
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let c = crate::grid::cell_at(&frame, p.x, p.y, &move |_, _| h);
        let tr = &f
            .aircraft
            .iter()
            .find(|a| a.icao == "bbbb01")
            .unwrap()
            .tracks[0];
        let ta = inp.atmos.cpa(tr, c.ecef).unwrap().t_arrive;
        let q0 = img.bin_of(87.0);
        let times = img.times.clone();
        let k = inp.atmos.factors(tr, c.ecef, &times);
        let jc = ((ta - times[0]) / 0.25).round() as i64;
        for (j, kj) in k.iter().enumerate() {
            let Some(kj) = kj else { continue };
            if (times[j] - ta).abs() > GUIDE_SPAN_S {
                continue;
            }
            let q = (q0 + gm_doppler::audio::guide::GUIDE_BPO * kj.log2()).round() as usize;
            if falling {
                img.level[j][q] += 3.0;
            }
            let jm = (2 * jc - j as i64) as usize;
            if rising && jm < n_frames {
                img.level[jm][q] += 3.0;
            }
        }
        img
    }

    fn quiet_inputs<'a>(
        f: &'a Fixture,
        a: &'a Area,
        ground: &'a (dyn Fn(f64, f64) -> f64 + Sync),
        atmos: &'a Atmosphere,
    ) -> Inputs<'a> {
        Inputs {
            events: &[],
            aircraft: &f.aircraft,
            area: a,
            when: f.when,
            duration_s: f.duration_s,
            ground,
            atmos,
            noise: None,
            decoys: &[],
        }
    }

    #[test]
    fn rising_null_rejects_time_symmetric_clutter() {
        // Lit along the falling curve only: a real pass, kept. Lit along it AND its mirror in
        // time: a passing source only falls, so a feature the impossible rising curve explains
        // as well is clutter — rejected, though it passes z, coverage, the time-shifted null
        // and centredness.
        let (f, a) = quiet();
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = quiet_inputs(&f, &a, &ground, &atmos);
        let by = |g: &[Event]| g.iter().any(|e| e.guide_icao.as_deref() == Some("bbbb01"));
        let real = guided_events(&lit_image(&f, &inp, true, false), &inp, Band::Prop);
        assert!(by(&real), "a falling curve is a pass: {real:#?}");
        let both = guided_events(&lit_image(&f, &inp, true, true), &inp, Band::Prop);
        assert!(!by(&both), "symmetric clutter: {both:#?}");
    }

    #[test]
    fn jet_band_stops_at_its_tone_range() {
        // The same curve at a cell where the pass is ~3.8 km away: the jet band (3 km) refuses
        // to score it, the prop band (10 km) does.
        let (f, a) = quiet();
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = quiet_inputs(&f, &a, &ground, &atmos);
        let img = lit_image(&f, &inp, true, false);
        let frame = area_frame(&inp);
        let p = frame.to_enu(gm_core::geodetic_to_ecef(f.truth.observer));
        let c = crate::grid::cell_at(&frame, p.x, p.y - 3000.0, &ground);
        let tr = &f
            .aircraft
            .iter()
            .find(|x| x.icao == "bbbb01")
            .unwrap()
            .tracks[0];
        let slant = atmos.cpa(tr, c.ecef).unwrap().slant_m;
        assert!(slant > crate::params::JET_MAX_SLANT_M + 300.0, "{slant}");
        let jet = Sweep::new(&img, (-6.0, 6.0), Band::Jet).unwrap();
        let prop = Sweep::new(&img, (-6.0, 6.0), Band::Prop).unwrap();
        assert!(score(&img, &jet, &jet.deltas, &inp, tr, c).is_none());
        assert!(score(&img, &prop, &prop.deltas, &inp, tr, c).is_some());
    }

    #[test]
    fn two_aircraft_one_feature_both_dropped() {
        // Two aircraft whose curves explain the same feature: the attribution is ambiguous, so
        // neither gets the event (keeping the stronger would pick one by noise).
        let (mut f, a) = quiet();
        let mut twin = f
            .aircraft
            .iter()
            .find(|c| c.icao == "bbbb01")
            .unwrap()
            .clone();
        twin.icao = "bbbb09".into();
        f.aircraft.push(twin);
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = quiet_inputs(&f, &a, &ground, &atmos);
        let g = guided_events(&lit_image(&f, &inp, true, false), &inp, Band::Prop);
        assert!(
            g.iter()
                .all(|e| !matches!(e.guide_icao.as_deref(), Some("bbbb01" | "bbbb09"))),
            "{g:#?}"
        );
    }

    #[test]
    fn jets_not_proposed() {
        let (mut f, a) = quiet();
        for c in &mut f.aircraft {
            c.class = AircraftClass::Jet;
        }
        assert!(run(&f, &a, f.when).is_empty());
    }
}
