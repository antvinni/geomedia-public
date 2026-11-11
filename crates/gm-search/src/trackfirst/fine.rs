//! Stage 4: refine an accepted candidate and check each aircraft at full resolution, with f₀
//! free to drift (a band-limited Viterbi path on the de-Dopplered stripe). It can only refine
//! or veto: the decision was made in stage 3, over the whole search.

use gm_core::LocalFrame;
use gm_doppler::audio::guide::{GuideImage, measure_along};
use gm_doppler::physics::Track;
use serde::Serialize;

use super::joint::{Cand, floor_cap};
use super::maps::{PairMap, slack_s};
use super::null::different_geometry;
use crate::grid::{Candidate, Cell, Inputs, cell_at};
use crate::guided::{Band, Scored, Sweep, centred_fraction, curve_z, decoy_shifts, null_shifts};
use crate::params::{
    GUIDE_CENTRED_MIN, GUIDE_CLIP, GUIDE_MIN_LEVEL, GUIDE_MIN_POINTS, GUIDE_NULL_MARGIN,
    GUIDE_NULL_MAX_SHIFTS, GUIDE_NULL_MIN_SHIFTS, GUIDE_SPAN_S, GUIDE_TOL, TF_CELL_M,
    TF_DELTA_STEP_S, TF_FINE_DELTA_S, TF_FINE_RADIUS_M, TF_FINE_STEP_M, TF_VITERBI_DRIFT,
    TF_VITERBI_STEP_BINS_PER_S,
};

/// One aircraft's part in a track-first answer.
#[derive(Clone, Debug, Serialize)]
pub struct Contributor {
    pub icao: String,
    pub type_code: Option<String>,
    pub slant_m: f64,
    pub bearing_deg: f64,
    /// Its pooled coarse z and contribution to J at the candidate.
    pub z_coarse: f64,
    pub contribution: f64,
    /// Full-resolution constant-f₀ z near the fine optimum, and the largest z of its null.
    pub z_fine: Option<f64>,
    pub null_fine: Option<f64>,
    /// Values the fine null took (in-recording shifts, decoy passes).
    pub null_n: Option<(usize, usize)>,
    pub centred: Option<f64>,
    pub coverage: Option<f64>,
    /// Source f₀ drift along the path, percent (last vs first frame), and its start/end Hz.
    pub drift_pct: Option<f64>,
    pub f0_hz: Option<(f64, f64)>,
    /// Still counts after the fine check.
    pub counts: bool,
}

/// What the fine stage found: every fine cell's log-likelihood at its best δ, the polished
/// optimum (cell, δ) and the contributors as checked there.
pub struct FineResult {
    pub cells: Vec<(Cell, f64, f64)>,
    pub best: (Cell, f64),
    pub contributors: Vec<Contributor>,
}

/// Frames of `track` heard at `cell` at offset `delta` on `img`: (frame, log-bin shift).
fn stripe(
    img: &GuideImage,
    inp: &Inputs,
    track: &Track,
    cell: Cell,
    delta: f64,
) -> Option<(f64, Vec<(usize, i64)>)> {
    let c = inp.atmos.cpa(track, cell.ecef)?;
    let js: Vec<usize> = (0..img.times.len())
        .filter(|&j| (img.times[j] + delta - c.t_arrive).abs() <= GUIDE_SPAN_S)
        .collect();
    let taus: Vec<f64> = js.iter().map(|&j| img.times[j] + delta).collect();
    let w = track.window(taus.first()? - 120.0, taus.last()? + 60.0)?;
    let frames = js
        .iter()
        .zip(inp.atmos.factors(&w, cell.ecef, &taus))
        .filter_map(|(&j, k)| {
            k.filter(|k| *k > 0.0)
                .map(|k| (j, (img.bpo * k.log2()).round() as i64))
        })
        .collect();
    Some((c.t_arrive, frames))
}

/// The best source-bin path through `frames` (frame, Doppler bin shift) starting near `q0`:
/// states are source bins within ±TF_VITERBI_DRIFT of q0; the path may move one bin per block of
/// frames, a block being as many frames as TF_VITERBI_STEP_BINS_PER_S allows per bin (at 192
/// bins/octave, scaled to `img.bpo`), each move costing √(block frames); emission = the clipped
/// score at (frame, q + shift).
/// Returns the path's sum and its source bin per frame.
pub fn viterbi(img: &GuideImage, frames: &[(usize, i64)], q0: usize) -> Option<(f64, Vec<usize>)> {
    viterbi_on(img, frames, q0, false)
}

/// `viterbi` on the score image, or (`level`) on the dB image the points are measured on.
fn viterbi_on(
    img: &GuideImage,
    frames: &[(usize, i64)],
    q0: usize,
    level: bool,
) -> Option<(f64, Vec<usize>)> {
    if frames.is_empty() || img.hop_s <= 0.0 {
        return None;
    }
    let nb = img.n_bins() as i64;
    let d = ((1.0 + TF_VITERBI_DRIFT).log2() * img.bpo).floor() as i64;
    let (lo, hi) = ((q0 as i64 - d).max(0), (q0 as i64 + d).min(nb - 1));
    if lo > hi {
        return None;
    }
    let ns = (hi - lo + 1) as usize;
    let rate = TF_VITERBI_STEP_BINS_PER_S * img.hop_s * img.bpo / 192.0; // bins per frame
    let per = (1.0 / rate).ceil().max(1.0) as usize;
    let blocks: Vec<&[(usize, i64)]> = frames.chunks(per).collect();
    // A move must buy more than a block's noise: the image is standardised (noise ≈ 1 per
    // frame), so a block of `per` frames has noise ≈ √per — charge that per bin moved, or the
    // path wanders on noise wherever the tone is faint.
    let move_cost = (per as f64).sqrt();
    let emit = |b: &[(usize, i64)], q: i64| -> f64 {
        b.iter()
            .filter_map(|&(j, s)| {
                let k = q + s;
                (0..nb)
                    .contains(&k)
                    .then(|| {
                        let row = if level {
                            &img.level[j]
                        } else {
                            img.score_row(j)
                        };
                        f64::from(row[k as usize].clamp(-GUIDE_CLIP, GUIDE_CLIP))
                    })
                    .filter(|x| x.is_finite())
            })
            .sum()
    };
    let mut score: Vec<f64> = (0..ns).map(|i| emit(blocks[0], lo + i as i64)).collect();
    let mut back: Vec<Vec<u8>> = Vec::with_capacity(blocks.len());
    for b in &blocks[1..] {
        let mut next = vec![f64::NEG_INFINITY; ns];
        let mut from = vec![1u8; ns];
        for i in 0..ns {
            for (k, di) in [(0u8, -1i64), (1, 0), (2, 1)] {
                let p = i as i64 + di;
                if p < 0 || p >= ns as i64 {
                    continue;
                }
                let v = score[p as usize] - if di == 0 { 0.0 } else { move_cost };
                if v > next[i] {
                    next[i] = v;
                    from[i] = k;
                }
            }
            next[i] += emit(b, lo + i as i64);
        }
        back.push(from);
        score = next;
    }
    let (mut i, &total) = score.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    let mut states = vec![i; blocks.len()];
    for (bi, from) in back.iter().enumerate().rev() {
        i = (i as i64 + i64::from(from[i]) - 1) as usize;
        states[bi] = i;
    }
    let path = blocks
        .iter()
        .zip(&states)
        .flat_map(|(b, &st)| std::iter::repeat_n((lo + st as i64) as usize, b.len()))
        .collect();
    Some((total, path))
}

/// Fine cells: TF_FINE_STEP_M apart within TF_FINE_RADIUS_M of `c`, inside the area.
fn fine_cells(inp: &Inputs, frame: &LocalFrame, c: &Cell, step: f64, radius: f64) -> Vec<Cell> {
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let n = (radius / step).round() as i32;
    (-n..=n)
        .flat_map(|i| (-n..=n).map(move |k| (i, k)))
        .filter(|&(i, k)| f64::from(i * i + k * k).sqrt() * step <= radius + 1e-6)
        .map(|(i, k)| {
            cell_at(
                frame,
                c.east + f64::from(i) * step,
                c.north + f64::from(k) * step,
                &ground,
            )
        })
        .filter(|x| inp.area.contains(x.geo.lat_deg, x.geo.lon_deg))
        .collect()
}

/// The image of `band` in `imgs`.
fn img_of(imgs: &[(Band, GuideImage)], band: Band) -> Option<&GuideImage> {
    imgs.iter().find(|(b, _)| *b == band).map(|(_, i)| i)
}

/// For each contributing pair, the shared-δ log-likelihood Σ ½·z² of the aircraft clearing their
/// floor, at every δ of `deltas` (step `step_s`) on `imgs`, at `cell`; the best (ll, δ).
fn best_ll(
    imgs: &[(Band, GuideImage)],
    inp: &Inputs,
    pairs: &[&PairMap],
    cell: Cell,
    range: (f64, f64),
    step_s: f64,
) -> Option<(f64, f64)> {
    let mut acc: Vec<f64> = Vec::new();
    let mut deltas: Vec<f64> = Vec::new();
    for m in pairs {
        let img = img_of(imgs, m.band)?;
        let sw = Sweep::with_step(img, range, m.band, step_s)?;
        if deltas.is_empty() {
            deltas = sw.deltas.clone();
            acc = vec![0.0; deltas.len()];
        }
        let track = &inp.aircraft[m.aircraft].tracks[m.track];
        let mp = ((GUIDE_MIN_POINTS as f64 * 0.25 / img.hop_s).ceil() as usize).max(1);
        let Some((_, v)) = curve_z(img, &sw, &sw.deltas, inp, track, cell, false, mp) else {
            continue;
        };
        let (floor, _) = floor_cap(m.band);
        for (a, d) in acc.iter_mut().zip(&v) {
            if let Some(d) = d
                && d.z >= floor
            {
                *a += 0.5 * d.z * d.z;
            }
        }
    }
    acc.iter()
        .zip(&deltas)
        .max_by(|a, b| a.0.total_cmp(b.0))
        .map(|(&l, &d)| (l, d))
}

/// Stage 4 for candidate `cand`: (a) 250 m cells within 3 km on the coarse images, δ on 1 s
/// steps over the candidate's δ ± (largest slack + TF_FINE_DELTA_S); (b) cells 62.5 m apart
/// within 250 m of the best, δ ± TF_FINE_DELTA_S at the image hop, on the full-resolution
/// images' drifting-path fit; (c) each contributor's Viterbi check at the optimum.
#[allow(clippy::too_many_arguments)]
pub fn fine(
    imgs: &[(Band, GuideImage)],
    coarse: &[(Band, GuideImage)],
    inp: &Inputs,
    frame: &LocalFrame,
    maps: &[PairMap],
    cells: &[Cell],
    cand: &Cand,
) -> Option<FineResult> {
    let pairs: Vec<&PairMap> = cand.maps.iter().map(|&(m, _)| &maps[m]).collect();
    let slack = pairs
        .iter()
        .map(|m| slack_s(TF_CELL_M, m.v_mps))
        .fold(0.0, f64::max);
    let centre = cells.get(cand.cell)?;
    let half = slack + TF_FINE_DELTA_S;
    let range = (cand.delta - half, cand.delta + half);
    let scored: Vec<(Cell, f64, f64)> =
        fine_cells(inp, frame, centre, TF_FINE_STEP_M, TF_FINE_RADIUS_M)
            .into_iter()
            .filter_map(|c| {
                best_ll(coarse, inp, &pairs, c, range, TF_DELTA_STEP_S).map(|(l, d)| (c, l, d))
            })
            .collect();
    let (c1, _, d1) = *scored.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    // (b) polish at full resolution on the drifting-path fit: Σ over contributors of the
    // Viterbi sum on the dB image (the image points are measured on), from each one's
    // constant-f₀ start — sharp enough that every curve lands on its ridge.
    let hop = imgs.first().map_or(0.25, |(_, i)| i.hop_s);
    let polish = fine_cells(inp, frame, &c1, TF_FINE_STEP_M / 4.0, TF_FINE_STEP_M);
    let fit = |c: Cell, d: f64| -> f64 {
        pairs
            .iter()
            .filter_map(|m| {
                let img = img_of(imgs, m.band)?;
                let sw = Sweep::with_step(img, (d, d), m.band, hop)?;
                let track = &inp.aircraft[m.aircraft].tracks[m.track];
                let (_, frames) = stripe(img, inp, track, c, d)?;
                let q0 = curve_z(img, &sw, &[d], inp, track, c, false, GUIDE_MIN_POINTS)?
                    .1
                    .first()
                    .copied()
                    .flatten()?
                    .q0;
                viterbi_on(img, &frames, q0, true).map(|(v, _)| v)
            })
            .sum()
    };
    let n_d = (TF_FINE_DELTA_S / hop).round() as i32;
    let (best_cell, best_delta) = polish
        .into_iter()
        .flat_map(|c| (-n_d..=n_d).map(move |k| (c, d1 + f64::from(k) * hop)))
        .map(|(c, d)| (c, d, fit(c, d)))
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map_or((c1, d1), |(c, d, _)| (c, d));
    let contributors = cand
        .maps
        .iter()
        .map(|&(mi, contrib)| {
            let m = &maps[mi];
            let mut out = Contributor {
                icao: inp.aircraft[m.aircraft].icao.clone(),
                type_code: inp.aircraft[m.aircraft].type_code.clone(),
                slant_m: m.slant_m,
                bearing_deg: m.bearing_deg,
                z_coarse: m.hit_at(cand.delta).map_or(f64::NAN, |h| f64::from(h.z)),
                contribution: contrib,
                z_fine: None,
                null_fine: None,
                null_n: None,
                centred: None,
                coverage: None,
                drift_pct: None,
                f0_hz: None,
                counts: false,
            };
            if let Some(img) = img_of(imgs, m.band) {
                // The other contributors' passes are explained, not chance: the null avoids them.
                let busy: Vec<(f64, f64)> = cand
                    .maps
                    .iter()
                    .filter(|&&(o, _)| maps[o].aircraft != m.aircraft)
                    .filter_map(|&(o, _)| {
                        let om = &maps[o];
                        let tr = &inp.aircraft[om.aircraft].tracks[om.track];
                        let a = inp.atmos.cpa(tr, best_cell.ecef)?.t_arrive - best_delta;
                        Some((a - GUIDE_SPAN_S, a + GUIDE_SPAN_S))
                    })
                    .collect();
                check(img, inp, frame, m, best_cell, best_delta, &busy, &mut out);
            }
            out
        })
        .collect();
    Some(FineResult {
        cells: scored,
        best: (best_cell, best_delta),
        contributors,
    })
}

/// The constant-f₀ z of `track` over the 3×3 cells 62.5 m apart around `cell` and every δ of
/// `deltas` (the sweep's step): the best (z, cell, δ, f₀ bin).
fn local_z(
    img: &GuideImage,
    sw: &Sweep,
    inp: &Inputs,
    frame: &LocalFrame,
    track: &Track,
    cell: Cell,
    deltas: &[f64],
) -> Option<(f64, Cell, f64, usize)> {
    fine_cells(
        inp,
        frame,
        &cell,
        TF_FINE_STEP_M / 4.0,
        TF_FINE_STEP_M / 4.0 * 1.5,
    )
    .into_iter()
    .chain(std::iter::once(cell))
    .filter_map(|c| {
        let (_, v) = curve_z(img, sw, deltas, inp, track, c, false, GUIDE_MIN_POINTS)?;
        v.iter()
            .enumerate()
            .filter_map(|(i, d)| d.map(|d| (d.z, c, deltas[i], d.q0)))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    })
    .max_by(|a, b| a.0.total_cmp(&b.0))
}

/// Fill `out` with map `m`'s full-resolution check near (`cell`, `delta`): its constant-f₀ z
/// (best over the 3×3 cells 62.5 m apart and δ ± TF_FINE_DELTA_S), its null (the same search at
/// the guided null's in-recording shifts — none over `busy`, the other contributors' audio
/// spans — and at decoy passes re-timed onto it), and the centred fraction and drift along the
/// Viterbi f₀ path there, on the dB image the points are measured on. It still counts when it
/// beats its null by GUIDE_NULL_MARGIN and its points lie on the drifting path.
#[allow(clippy::too_many_arguments)]
fn check(
    img: &GuideImage,
    inp: &Inputs,
    frame: &LocalFrame,
    m: &PairMap,
    cell: Cell,
    delta: f64,
    busy: &[(f64, f64)],
    out: &mut Contributor,
) {
    let me: &Candidate = &inp.aircraft[m.aircraft];
    let track = &me.tracks[m.track];
    let Some(sw) = Sweep::with_step(
        img,
        (delta - TF_FINE_DELTA_S, delta + TF_FINE_DELTA_S),
        m.band,
        img.hop_s,
    ) else {
        return;
    };
    let Some((z, c, d, q0)) = local_z(img, &sw, inp, frame, track, cell, &sw.deltas) else {
        return;
    };
    out.z_fine = Some(z);
    // The shape check at the polished optimum itself (chosen on the path fit, which is sharper
    // than z's peak), from that point's constant-f₀ start.
    let q_here = curve_z(
        img,
        &sw,
        &[delta],
        inp,
        track,
        cell,
        false,
        GUIDE_MIN_POINTS,
    )
    .and_then(|(_, v)| v.first().copied().flatten())
    .map_or(q0, |x| x.q0);
    let Some((ta, frames)) = stripe(img, inp, track, cell, delta) else {
        return;
    };
    if let Some((_, path)) = viterbi_on(img, &frames, q_here, true) {
        let (f_a, f_b) = (
            img.freq_of(path[0] as f64),
            img.freq_of(path[path.len() - 1] as f64),
        );
        out.drift_pct = Some(100.0 * (f_b / f_a - 1.0));
        out.f0_hz = Some((f_a, f_b));
        let curve: Vec<(usize, f64)> = frames
            .iter()
            .zip(&path)
            .map(|(&(j, s), &p)| (j, img.freq_of((p as i64 + s) as f64)))
            .collect();
        let pts = measure_along(img, &curve, GUIDE_TOL, GUIDE_MIN_LEVEL);
        out.coverage = Some(pts.len() as f64 / frames.len().max(1) as f64);
        out.centred = Some(if pts.len() >= GUIDE_MIN_POINTS {
            centred_fraction(img, &curve, &pts)
        } else {
            0.0
        });
    }
    // The null: the guided null's shift set and decoy passes, searched like the tested curve.
    let ta = inp.atmos.cpa(track, c.ecef).map_or(ta, |x| x.t_arrive);
    let s = Scored {
        z,
        delta: d,
        q0,
        t_arrive: ta,
        coverage: 0.0,
        mean_db: 0.0,
        cell: c,
    };
    let others: Vec<Candidate> = inp
        .aircraft
        .iter()
        .filter(|a| a.icao != me.icao)
        .cloned()
        .collect();
    // The blind leg's events never shape track-first: the shifts avoid only the other
    // contributors' passes (`busy`).
    let one = Inputs {
        events: &[],
        aircraft: std::slice::from_ref(me),
        decoys: &others,
        ..*inp
    };
    let mut shifts: Vec<f64> = null_shifts(&sw, &one, &s)
        .unwrap_or_default()
        .into_iter()
        .filter(|sh| {
            let a = ta - d - sh;
            busy.iter()
                .all(|&(b0, b1)| a + GUIDE_SPAN_S < b0 || a - GUIDE_SPAN_S > b1)
        })
        .collect();
    if shifts.len() > GUIDE_NULL_MAX_SHIFTS {
        let k = shifts.len();
        shifts = (0..GUIDE_NULL_MAX_SHIFTS)
            .map(|i| shifts[i * k / GUIDE_NULL_MAX_SHIFTS])
            .collect();
    }
    let decoys = decoy_shifts(&sw, &one, &me.icao, &s);
    let mut nz = f64::NEG_INFINITY;
    let (mut n_shift, mut n_decoy) = (0usize, 0usize);
    let probe = |tr: &Track, sh: f64| {
        let ds: Vec<f64> = sw.deltas.iter().map(|x| x + sh).collect();
        local_z(img, &sw, &one, frame, tr, cell, &ds).map(|v| v.0)
    };
    for &sh in &shifts {
        if let Some(v) = probe(track, sh) {
            nz = nz.max(v);
            n_shift += 1;
        }
    }
    for &(tr, sh) in &decoys {
        if let Some(v) = probe(tr, sh) {
            nz = nz.max(v);
            n_decoy += 1;
        }
    }
    out.null_n = Some((n_shift, n_decoy));
    if n_shift + n_decoy >= GUIDE_NULL_MIN_SHIFTS {
        out.null_fine = Some(nz);
        out.counts =
            z >= nz + GUIDE_NULL_MARGIN && out.centred.is_some_and(|x| x >= GUIDE_CENTRED_MIN);
    }
}

/// Do at least two counting contributors have different geometry?
pub fn still_fixes(c: &[Contributor]) -> bool {
    let b: Vec<f64> = c
        .iter()
        .filter(|x| x.counts)
        .map(|x| x.bearing_deg)
        .collect();
    b.iter()
        .enumerate()
        .any(|(i, &x)| b[i + 1..].iter().any(|&y| different_geometry(x, y)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 120 frames at 0.25 s, 192 bpo, 800 bins; a line at `q(j)`.
    fn line(q: impl Fn(usize) -> usize) -> GuideImage {
        let mut img = GuideImage {
            hop_s: 0.25,
            f_lo: 30.0,
            ..Default::default()
        };
        img.times = (0..120).map(|j| 0.125 + 0.25 * j as f64).collect();
        img.level = (0..120)
            .map(|j| {
                let mut r = vec![0.0f32; 800];
                r[q(j)] = 3.0;
                r
            })
            .collect();
        img
    }

    #[test]
    fn viterbi_follows_a_drifting_tone() {
        // +5 % over 30 s: 13.5 bins, under the 1 bin/s limit.
        let q = |j: usize| 200 + (j as f64 * 13.5 / 119.0).round() as usize;
        let img = line(q);
        let frames: Vec<(usize, i64)> = (0..120).map(|j| (j, 0)).collect();
        let (sum, path) = viterbi(&img, &frames, 200).unwrap();
        for (j, &p) in path.iter().enumerate() {
            assert!(
                (p as i64 - q(j) as i64).abs() <= 1,
                "frame {j}: {p} vs {}",
                q(j)
            );
        }
        let constant: f64 = (0..120).filter(|&j| q(j) == 200).count() as f64 * 3.0;
        assert!(constant < 0.5 * sum, "constant {constant} vs viterbi {sum}");
    }

    #[test]
    fn viterbi_respects_the_drift_bound() {
        // A jump to +20 % halfway: the path never leaves ±8 % of q0.
        let up = (192.0 * 1.2f64.log2()).round() as usize;
        let img = line(|j| if j < 60 { 200 } else { 200 + up });
        let frames: Vec<(usize, i64)> = (0..120).map(|j| (j, 0)).collect();
        let (_, path) = viterbi(&img, &frames, 200).unwrap();
        let d = (192.0 * 1.08f64.log2()).floor() as usize;
        assert!(path.iter().all(|&p| p >= 200 - d && p <= 200 + d));
    }

    #[test]
    fn viterbi_follows_the_doppler_shift() {
        // A constant source seen through a falling Doppler shift: the path stays at q0.
        let shift = |j: usize| 20 - (j as i64 * 40 / 119);
        let img = line(|j| (300 + shift(j)) as usize);
        let frames: Vec<(usize, i64)> = (0..120).map(|j| (j, shift(j))).collect();
        let (_, path) = viterbi(&img, &frames, 300).unwrap();
        assert!(path.iter().all(|&p| p.abs_diff(300) <= 1), "{path:?}");
    }

    #[test]
    fn blind_events_do_not_shape_the_fine_null() {
        use crate::atmos::Atmosphere;
        use crate::testutil::fixture;
        use crate::trackfirst::joint::tests::mk;
        use crate::trackfirst::maps::tests::area;
        use gm_doppler::audio::guide::guide_image;
        let f = fixture("suite/crossing.toml", |s| s);
        assert!(!f.events.is_empty());
        let a = area("40.8765,-73.5250~3km");
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let mut ac = f.aircraft.clone();
        ac.extend(crate::testutil::decoys(&f, 24));
        let img = guide_image(&f.audio);
        let frame = LocalFrame::new(f.truth.observer);
        let cell = cell_at(&frame, 0.0, 0.0, &ground);
        let run = |events: &[gm_doppler::audio::Event]| {
            let inp = Inputs {
                events,
                aircraft: &ac,
                area: &a,
                when: f.when,
                duration_s: f.duration_s,
                ground: &ground,
                atmos: &atmos,
                noise: None,
                decoys: &[],
            };
            let mut out = Contributor {
                icao: String::new(),
                type_code: None,
                slant_m: 0.0,
                bearing_deg: 0.0,
                z_coarse: 0.0,
                contribution: 0.0,
                z_fine: None,
                null_fine: None,
                null_n: None,
                centred: None,
                coverage: None,
                drift_pct: None,
                f0_hz: None,
                counts: false,
            };
            check(
                &img,
                &inp,
                &frame,
                &mk(0, 0, 0.0, 0, &[]),
                cell,
                0.0,
                &[],
                &mut out,
            );
            (out.null_fine, out.null_n)
        };
        let clean = run(&[]);
        assert!(clean.0.is_some());
        assert_eq!(
            run(&f.events),
            clean,
            "the blind leg's events changed the fine null"
        );
    }
}
