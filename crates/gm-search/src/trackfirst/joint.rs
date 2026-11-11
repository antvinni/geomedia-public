//! Stage 2: the joint score J(cell, δ) = Σ over aircraft of each one's capped excess z above the
//! floor, with one ridge credited to one aircraft only. Everything here is a lookup into the
//! stage-1 maps, so the null (stage 3) can re-sum with every aircraft shifted in time.

use super::maps::{Hit, PairMap};
use crate::guided::Band;
use crate::params::{
    TF_DELTA_STEP_S, TF_EXPLAIN_BINS, TF_EXPLAIN_FRAC, TF_TOP_K, TF_Z_CAP, TF_Z_CAP_JET,
    TF_Z_FLOOR, TF_Z_FLOOR_JET,
};

/// The band's per-aircraft floor and cap.
pub fn floor_cap(band: Band) -> (f64, f64) {
    match band {
        Band::Prop => (TF_Z_FLOOR, TF_Z_CAP),
        Band::Jet => (TF_Z_FLOOR_JET, TF_Z_CAP_JET),
    }
}

/// One aircraft's contribution from its map value: the excess over its band's floor, capped;
/// 0 when the rising curve explains it as well, or nothing was scored.
pub fn contribution(z: f32, rise_ok: bool, band: Band) -> f64 {
    if !rise_ok || z.is_nan() {
        return 0.0;
    }
    let (floor, cap) = floor_cap(band);
    (f64::from(z) - floor).clamp(0.0, cap)
}

/// A scored (cell, δ): its J and the maps that contributed (map index, contribution).
#[derive(Clone, Debug, PartialEq)]
pub struct Cand {
    pub cell: usize,
    pub delta: f64,
    pub j: f64,
    pub maps: Vec<(usize, f64)>,
}

/// The maps touching each cell.
pub struct Index {
    by_cell: Vec<Vec<usize>>,
}

impl Index {
    pub fn new(maps: &[PairMap], n_cells: usize) -> Index {
        let mut by_cell = vec![Vec::new(); n_cells];
        for (i, m) in maps.iter().enumerate() {
            if let Some(v) = by_cell.get_mut(m.cell) {
                v.push(i);
            }
        }
        Index { by_cell }
    }
    pub fn n_cells(&self) -> usize {
        self.by_cell.len()
    }
    pub fn cell(&self, c: usize) -> &[usize] {
        self.by_cell.get(c).map_or(&[], Vec::as_slice)
    }
}

/// A read curve: (audio frame, log bin) per point.
type Curve = Vec<(i64, i64)>;

/// The curve of map `m` read at hit `h` (clock offset `d_read`), as (audio frame, log bin).
fn curve_cells(m: &PairMap, h: &Hit, d_read: f64) -> Vec<(i64, i64)> {
    // The pooled maximum sits `lag` steps away: place the curve where it was found.
    let d = d_read + f64::from(h.lag) * TF_DELTA_STEP_S;
    let arrive = ((m.t_arrive - d) / m.hop_s).round() as i64;
    m.shape
        .iter()
        .map(|&(o, s)| (arrive + i64::from(o), i64::from(h.q0) + i64::from(s)))
        .collect()
}

/// Do two read curves lie on one ridge: the same frames, within TF_EXPLAIN_BINS, over at least
/// TF_EXPLAIN_FRAC of the shorter one?
fn overlap(a: &[(i64, i64)], b: &[(i64, i64)]) -> bool {
    let short = a.len().min(b.len());
    if short == 0 {
        return false;
    }
    let bm: std::collections::HashMap<i64, i64> = b.iter().copied().collect();
    let shared = a
        .iter()
        .filter(|(f, q)| {
            bm.get(f)
                .is_some_and(|qb| (qb - q).abs() <= i64::from(TF_EXPLAIN_BINS))
        })
        .count();
    shared as f64 >= TF_EXPLAIN_FRAC * short as f64
}

/// J at (`cell`, `delta`), every map of aircraft `a` read at δ − `shift(a)`. One value per
/// aircraft (its best track); then, strongest first, a curve on the same ridge as one already
/// kept contributes 0. Returns J and the kept (map, contribution) pairs.
pub fn joint_at(
    maps: &[PairMap],
    idx: &Index,
    cell: usize,
    delta: f64,
    shift: &dyn Fn(usize) -> f64,
) -> (f64, Vec<(usize, f64)>) {
    // Best track per aircraft.
    let mut per: Vec<(usize, f64, f64)> = Vec::new(); // (map, contribution, read δ)
    for &mi in idx.cell(cell) {
        let m = &maps[mi];
        let d = delta - shift(m.aircraft);
        let Some(h) = m.hit_at(d) else { continue };
        let c = contribution(h.z, h.rise_ok, m.band);
        if c <= 0.0 {
            continue;
        }
        match per.iter_mut().find(|p| maps[p.0].aircraft == m.aircraft) {
            Some(p) if p.1 >= c => {}
            Some(p) => *p = (mi, c, d),
            None => per.push((mi, c, d)),
        }
    }
    per.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut kept: Vec<(usize, f64, Curve)> = Vec::new();
    for (mi, c, d) in per {
        let m = &maps[mi];
        let h = m.hit_at(d).copied().expect("read above");
        let cur = curve_cells(m, &h, d);
        if kept.iter().all(|k| !overlap(&k.2, &cur)) {
            kept.push((mi, c, cur));
        }
    }
    let j = kept.iter().map(|k| k.1).sum();
    (j, kept.into_iter().map(|k| (k.0, k.1)).collect())
}

/// Per cell, J's upper bound over the δ grid inside `prior` (every hit's contribution summed,
/// before the one-per-aircraft and explain-away rules), then exact J at the bound's largest
/// values until no remaining bound can beat the best exact one. The best (J, δ, maps) of the
/// cell, or `None` when nothing contributes.
fn best_in_cell(
    maps: &[PairMap],
    idx: &Index,
    cell: usize,
    prior: (f64, f64),
    shift: &dyn Fn(usize) -> f64,
    buf: &mut Vec<f64>,
) -> Option<Cand> {
    let step = TF_DELTA_STEP_S;
    let p0 = (prior.0 / step).ceil();
    let n = ((prior.1 / step).floor() - p0 + 1.0).max(0.0) as usize;
    if n == 0 {
        return None;
    }
    buf.clear();
    buf.resize(n, 0.0);
    let mut touched: Vec<usize> = Vec::new();
    for &mi in idx.cell(cell) {
        let m = &maps[mi];
        let sh = shift(m.aircraft);
        for h in &m.hits {
            let c = contribution(h.z, h.rise_ok, m.band);
            if c <= 0.0 {
                continue;
            }
            let k = ((m.delta_of(h.step) + sh) / step).round() - p0;
            if k < 0.0 || k >= n as f64 {
                continue;
            }
            let k = k as usize;
            if buf[k] == 0.0 {
                touched.push(k);
            }
            buf[k] += c;
        }
    }
    touched.sort_by(|a, b| buf[*b].total_cmp(&buf[*a]).then(a.cmp(b)));
    let mut best: Option<Cand> = None;
    for k in touched {
        if best.as_ref().is_some_and(|b| b.j >= buf[k]) {
            break;
        }
        let delta = (p0 + k as f64) * step;
        let (j, kept) = joint_at(maps, idx, cell, delta, shift);
        if j > 0.0 && best.as_ref().is_none_or(|b| j > b.j) {
            best = Some(Cand {
                cell,
                delta,
                j,
                maps: kept,
            });
        }
    }
    best
}

/// The largest J over every cell and every δ inside `prior`, with each aircraft shifted by
/// `shift`: (J, cell, δ); (0, 0, NaN) when nothing contributes.
pub fn max_joint(
    maps: &[PairMap],
    idx: &Index,
    prior: (f64, f64),
    shift: &dyn Fn(usize) -> f64,
) -> (f64, usize, f64) {
    let mut buf = Vec::new();
    (0..idx.n_cells())
        .filter_map(|c| best_in_cell(maps, idx, c, prior, shift, &mut buf))
        .max_by(|a, b| a.j.total_cmp(&b.j).then(b.cell.cmp(&a.cell)))
        .map_or((0.0, 0, f64::NAN), |c| (c.j, c.cell, c.delta))
}

/// Candidates: the best δ of each cell, the top TF_TOP_K cells by J, plus the best seeded δ of
/// each cell among `seeds` (cell, δ) — always kept, so a strong blind pass is never cut —
/// deduplicated, strongest first. Unshifted; δ inside `prior` only.
pub fn candidates(
    maps: &[PairMap],
    idx: &Index,
    prior: (f64, f64),
    seeds: &[(usize, f64)],
) -> Vec<Cand> {
    let none = |_: usize| 0.0;
    let mut buf = Vec::new();
    let mut out: Vec<Cand> = (0..idx.n_cells())
        .filter_map(|c| best_in_cell(maps, idx, c, prior, &none, &mut buf))
        .collect();
    out.sort_by(|a, b| b.j.total_cmp(&a.j).then(a.cell.cmp(&b.cell)));
    out.truncate(TF_TOP_K);
    let step = TF_DELTA_STEP_S;
    let mut best_seed: Vec<Option<Cand>> = vec![None; idx.n_cells()];
    for &(cell, d) in seeds {
        let d = (d / step).round() * step;
        if cell >= idx.n_cells() || d < prior.0 || d > prior.1 {
            continue;
        }
        let (j, kept) = joint_at(maps, idx, cell, d, &none);
        if j > 0.0 && best_seed[cell].as_ref().is_none_or(|b| j > b.j) {
            best_seed[cell] = Some(Cand {
                cell,
                delta: d,
                j,
                maps: kept,
            });
        }
    }
    for s in best_seed.into_iter().flatten() {
        if !out.iter().any(|c| c.cell == s.cell && c.delta == s.delta) {
            out.push(s);
        }
    }
    out.sort_by(|a, b| b.j.total_cmp(&a.j).then(a.cell.cmp(&b.cell)));
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A synthetic map: aircraft `a` at `cell`, arriving at `t_arrive` (track clock), a flat
    /// 41-frame curve at bin 0 shift, hits (step, z) with q0 = `q0`, δ0 = 0, 1 s steps.
    pub(crate) fn mk(
        a: usize,
        cell: usize,
        t_arrive: f64,
        q0: u16,
        hits: &[(u32, f32)],
    ) -> PairMap {
        PairMap {
            aircraft: a,
            track: 0,
            cell,
            band: Band::Prop,
            delta0: 0.0,
            n_steps: 4000,
            hits: hits
                .iter()
                .map(|&(step, z)| Hit {
                    step,
                    z,
                    zr: 0.0,
                    q0,
                    lag: 0,
                    rise_ok: true,
                })
                .collect(),
            t_arrive,
            v_mps: 60.0,
            bearing_deg: 90.0 * a as f64,
            slant_m: 1000.0,
            shape: (-20..=20).map(|o| (o, -o / 4)).collect(),
            hop_s: 1.0,
        }
    }

    fn z(c: f64) -> f32 {
        (TF_Z_FLOOR + c) as f32
    }

    #[test]
    fn contribution_is_floored_capped_and_rise_gated() {
        let p = Band::Prop;
        assert_eq!(contribution(TF_Z_FLOOR as f32 - 1.0, true, p), 0.0);
        assert_eq!(
            contribution((TF_Z_FLOOR + TF_Z_CAP + 5.0) as f32, true, p),
            TF_Z_CAP
        );
        assert_eq!(contribution(50.0, false, p), 0.0);
        assert_eq!(contribution(f32::NAN, true, p), 0.0);
        assert!((contribution(z(1.5), true, p) - 1.5).abs() < 1e-6);
        // Jets have their own floor.
        let j = (TF_Z_FLOOR_JET + 1.0) as f32;
        assert!((contribution(j, true, Band::Jet) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn joint_sums_two_aircraft_at_shared_delta() {
        let maps = vec![
            mk(0, 0, 500.0, 50, &[(10, z(2.0))]),
            mk(1, 0, 900.0, 80, &[(10, z(3.0))]),
        ];
        let idx = Index::new(&maps, 1);
        let none = |_: usize| 0.0;
        let (j, kept) = joint_at(&maps, &idx, 0, 10.0, &none);
        assert!((j - 5.0).abs() < 1e-5, "{j}");
        assert_eq!(kept.len(), 2);
        assert_eq!(joint_at(&maps, &idx, 0, 30.0, &none).0, 0.0);
        let (mj, c, d) = max_joint(&maps, &idx, (-100.0, 100.0), &none);
        assert!((mj - 5.0).abs() < 1e-5 && c == 0 && d == 10.0);
        // Outside the prior: nothing.
        assert_eq!(max_joint(&maps, &idx, (20.0, 100.0), &none).0, 0.0);
    }

    #[test]
    fn one_ridge_counts_for_one_aircraft() {
        let maps = vec![
            mk(0, 0, 500.0, 50, &[(10, z(2.0))]),
            mk(1, 0, 500.0, 51, &[(10, z(3.0))]),
        ];
        let idx = Index::new(&maps, 1);
        let (j, kept) = joint_at(&maps, &idx, 0, 10.0, &|_| 0.0);
        assert!((j - 3.0).abs() < 1e-5, "{j}");
        assert_eq!(kept, vec![(1, (z(3.0) as f64) - TF_Z_FLOOR)]);
    }

    #[test]
    fn pooled_neighbours_on_one_ridge_count_once() {
        // Arrivals 20 s apart; b's pooled maximum lies 20 steps on, so both read the same ridge.
        let a = mk(0, 0, 500.0, 50, &[(0, z(2.0))]);
        let mut b = mk(1, 0, 520.0, 50, &[(0, z(3.0))]);
        b.hits[0].lag = 20;
        let maps = vec![a, b];
        let idx = Index::new(&maps, 1);
        let (j, kept) = joint_at(&maps, &idx, 0, 0.0, &|_| 0.0);
        assert_eq!(kept.len(), 1, "{kept:?}");
        assert!((j - 3.0).abs() < 1e-5);
    }

    #[test]
    fn shift_moves_a_map() {
        let maps = vec![mk(0, 0, 500.0, 50, &[(10, z(2.0))])];
        let idx = Index::new(&maps, 1);
        let sh = |_: usize| 100.0;
        assert!(joint_at(&maps, &idx, 0, 110.0, &sh).0 > 0.0);
        assert_eq!(joint_at(&maps, &idx, 0, 10.0, &sh).0, 0.0);
    }

    #[test]
    fn max_joint_matches_brute_force() {
        // Several aircraft over three cells with overlapping hits.
        let mut maps = Vec::new();
        for a in 0..6usize {
            for cell in 0..3usize {
                let hits: Vec<(u32, f32)> = (0..5u32)
                    .map(|k| {
                        (
                            ((a * 7 + cell * 3) as u32 + k * 11) % 60,
                            z(((a + k as usize) % 4) as f64),
                        )
                    })
                    .collect();
                let mut hs = hits.clone();
                hs.sort_by_key(|h| h.0);
                hs.dedup_by_key(|h| h.0);
                maps.push(mk(
                    a,
                    cell,
                    300.0 + 37.0 * a as f64,
                    (40 + 9 * a) as u16,
                    &hs,
                ));
            }
        }
        let idx = Index::new(&maps, 3);
        let sh = |a: usize| (a % 3) as f64 * 2.0;
        let mut brute = 0.0f64;
        for cell in 0..3 {
            for d in -10..=80 {
                brute = brute.max(joint_at(&maps, &idx, cell, f64::from(d), &sh).0);
            }
        }
        let (mj, _, _) = max_joint(&maps, &idx, (-10.0, 80.0), &sh);
        assert!((mj - brute).abs() < 1e-9, "{mj} vs {brute}");
    }

    #[test]
    fn seeds_are_always_candidates() {
        // 25 cells each with a strong pair; cell 25 holds a weaker one only a seed reaches.
        let mut maps = Vec::new();
        for c in 0..25usize {
            maps.push(mk(0, c, 500.0, 50, &[(10, z(5.0))]));
            maps.push(mk(1, c, 900.0, 80, &[(10, z(5.0))]));
        }
        maps.push(mk(0, 25, 500.0, 50, &[(40, z(0.5))]));
        let idx = Index::new(&maps, 26);
        let plain = candidates(&maps, &idx, (0.0, 100.0), &[]);
        assert_eq!(plain.len(), TF_TOP_K);
        assert!(plain.iter().all(|c| c.cell != 25));
        let seeded = candidates(&maps, &idx, (0.0, 100.0), &[(25, 40.0)]);
        assert!(seeded.iter().any(|c| c.cell == 25 && c.delta == 40.0));
    }
}
