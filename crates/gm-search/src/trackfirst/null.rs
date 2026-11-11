//! Stage 3, the decision: how often does chance alone reach a joint score this high? Each null
//! replicate shifts every aircraft independently in time (within ±TF_SHIFT_WINDOW_S, so busy
//! hours stay busy) and takes the largest J over the whole search — every cell, every δ of the
//! prior. That keeps each aircraft's own brightness and breaks only the coincidence of several
//! aircraft at one place and time, which is what a location claims. The observed J's global p
//! comes from those maxima (empirical rank, or a generalized Pareto tail beyond them).

use super::joint::{Cand, Index, joint_at, max_joint};
use super::maps::PairMap;
use crate::params::{
    TF_BEARING_MIN_DEG, TF_P_ACCEPT, TF_P_DROP, TF_REPLICATES, TF_SHIFT_WINDOW_S, TF_TAIL_QUANTILE,
};

/// splitmix64: small, fast, reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The shifts open to an aircraft with knot extent `extent` inside `span`: u in
/// [−TF_SHIFT_WINDOW_S, TF_SHIFT_WINDOW_S] keeping the shifted extent inside `span`, minus
/// (−min_abs, min_abs) — as up to two intervals.
fn feasible(extent: (f64, f64), span: (f64, f64), min_abs: f64) -> Vec<(f64, f64)> {
    let lo = (-TF_SHIFT_WINDOW_S).max(span.0 - extent.0);
    let hi = TF_SHIFT_WINDOW_S.min(span.1 - extent.1);
    [(lo, hi.min(-min_abs)), (lo.max(min_abs), hi)]
        .into_iter()
        .filter(|(a, b)| a <= b)
        .collect()
}

/// Can the null move this aircraft at all? (Decided once: an aircraft airborne through the whole
/// span — several rotations, a survey flight — never can.)
pub fn shiftable(extent: (f64, f64), span: (f64, f64), min_abs: f64) -> bool {
    !feasible(extent, span, min_abs).is_empty()
}

/// One shift per aircraft, uniform over the shifts open to it (`feasible`): within
/// ±TF_SHIFT_WINDOW_S, |u| ≥ `min_abs`, its shifted extent (first..last knot over all its tracks)
/// inside `span`; `None` only for an aircraft that can't be shifted at all.
pub fn draw_shifts(
    rng: &mut Rng,
    extents: &[(f64, f64)],
    span: (f64, f64),
    min_abs: f64,
) -> Vec<Option<f64>> {
    extents
        .iter()
        .map(|&e| {
            let iv = feasible(e, span, min_abs);
            if iv.is_empty() {
                return None;
            }
            let total: f64 = iv.iter().map(|(a, b)| b - a).sum();
            let mut x = rng.uniform() * total;
            for &(a, b) in &iv {
                if x <= b - a {
                    return Some(a + x);
                }
                x -= b - a;
            }
            iv.last().map(|&(_, b)| b)
        })
        .collect()
}

/// The maps of aircraft the null can shift, and how many aircraft were dropped: an aircraft the
/// null can never move would count in the observed J and never in a replicate, biasing p low.
pub fn drop_unshiftable(
    maps: Vec<PairMap>,
    extents: &[(f64, f64)],
    span: (f64, f64),
    min_abs: f64,
) -> (Vec<PairMap>, usize) {
    let ok = |a: usize| extents.get(a).is_some_and(|&e| shiftable(e, span, min_abs));
    let mut dropped: Vec<usize> = maps
        .iter()
        .map(|m| m.aircraft)
        .filter(|&a| !ok(a))
        .collect();
    dropped.sort_unstable();
    dropped.dedup();
    let kept = maps.into_iter().filter(|m| ok(m.aircraft)).collect();
    (kept, dropped.len())
}

/// The null: `r` replicate maxima of J over every cell and every δ in `prior`, each with every
/// aircraft independently shifted (`draw_shifts`). A replicate in which fewer than 2 of the
/// mapped aircraft could be shifted is discarded. Replicate k uses `Rng::new(seed ^ k·φ)`, so
/// the result is reproducible and independent of the thread count.
#[allow(clippy::too_many_arguments)]
pub fn replicate_maxima(
    maps: &[PairMap],
    idx: &Index,
    extents: &[(f64, f64)],
    span: (f64, f64),
    min_abs: f64,
    prior: (f64, f64),
    seed: u64,
    r: usize,
) -> Vec<f64> {
    let mut mapped: Vec<usize> = maps.iter().map(|m| m.aircraft).collect();
    mapped.sort_unstable();
    mapped.dedup();
    let one = |k: usize| -> Option<f64> {
        let mut rng = Rng::new(seed ^ (k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let u = draw_shifts(&mut rng, extents, span, min_abs);
        let shifted = mapped
            .iter()
            .filter(|&&a| u.get(a).copied().flatten().is_some())
            .count();
        if shifted < 2 {
            return None;
        }
        let shift = |a: usize| u.get(a).copied().flatten().unwrap_or(f64::INFINITY);
        Some(max_joint(maps, idx, prior, &shift).0)
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let ks: Vec<usize> = (0..r).collect();
    let chunk = ks.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = ks
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(|&k| one(k)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .flatten()
            .collect()
    })
}

/// Generalized Pareto (ξ, σ) of the excesses by probability-weighted moments (Hosking & Wallis
/// 1987); `None` for fewer than 10 values or a degenerate fit.
pub fn fit_gpd(excess: &[f64]) -> Option<(f64, f64)> {
    let n = excess.len();
    if n < 10 {
        return None;
    }
    let mut y = excess.to_vec();
    y.sort_by(f64::total_cmp);
    let a0 = y.iter().sum::<f64>() / n as f64;
    let a1 = y
        .iter()
        .enumerate()
        .map(|(i, v)| v * (n - 1 - i) as f64 / (n - 1) as f64)
        .sum::<f64>()
        / n as f64;
    let d = a0 - 2.0 * a1;
    if d <= 0.0 || !d.is_finite() {
        return None;
    }
    let k = a0 / d - 2.0;
    let sigma = 2.0 * a0 * a1 / d;
    (sigma > 0.0 && sigma.is_finite()).then_some((-k, sigma))
}

/// The observed J's global p against the replicate maxima: the empirical
/// (1 + #{m ≥ obs}) / (1 + R) while obs lies within the replicates (or the tail fit fails);
/// beyond them, the generalized Pareto tail above the TF_TAIL_QUANTILE quantile. With
/// R = 200 the accept rule (p ≤ 0.01) is decided by rank; the tail only refines the reported p.
pub fn global_p(obs: f64, maxima: &[f64]) -> f64 {
    let r = maxima.len();
    let ge = maxima.iter().filter(|&&m| m >= obs).count();
    let empirical = (1 + ge) as f64 / (1 + r) as f64;
    if r == 0 || ge > 0 {
        return empirical;
    }
    let mut s = maxima.to_vec();
    s.sort_by(f64::total_cmp);
    let u = s[((r - 1) as f64 * TF_TAIL_QUANTILE).round() as usize];
    let excess: Vec<f64> = s.iter().filter(|&&m| m > u).map(|m| m - u).collect();
    let Some((xi, sigma)) = fit_gpd(&excess) else {
        return empirical;
    };
    let y = obs - u;
    let surv = if xi.abs() < 1e-9 {
        (-y / sigma).exp()
    } else {
        let b = 1.0 + xi * y / sigma;
        if b <= 0.0 { 0.0 } else { b.powf(-1.0 / xi) }
    };
    ((1.0 - TF_TAIL_QUANTILE) * surv).clamp(1e-12, empirical)
}

/// What stage 3 decides for one candidate.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Accept,
    /// p small enough, but every contributing aircraft lies on one bearing line.
    OneStream,
    Reject(String),
}

/// Do two bearings (degrees) differ by at least TF_BEARING_MIN_DEG, as lines (mod 180°)?
pub fn different_geometry(a: f64, b: f64) -> bool {
    let d = (a - b).rem_euclid(180.0);
    d.min(180.0 - d) >= TF_BEARING_MIN_DEG
}

/// The spec's accept rule for `cand` against the replicate maxima: (1) global p ≤ TF_P_ACCEPT;
/// (3) with the strongest aircraft dropped, p still ≤ TF_P_DROP; (2) different geometry — the
/// aircraft off the strongest one's bearing line (≥ TF_BEARING_MIN_DEG, as lines) must by
/// themselves reach p ≤ TF_P_DROP, so a noise-level aircraft off the line cannot turn one
/// approach stream into a fix. Rejects outright when the null is too small or degenerate (most
/// replicates score nothing). Returns the decision, p and the drop-strongest p.
pub fn decide(
    maps: &[PairMap],
    idx: &Index,
    cand: &Cand,
    maxima: &[f64],
) -> (Decision, f64, Option<f64>) {
    if maxima.len() < TF_REPLICATES / 2 {
        return (Decision::Reject("no usable null".into()), 1.0, None);
    }
    // A null that never reaches the statistic can't calibrate it: too few aircraft are ever
    // shifted into the prior (sparse traffic).
    if maxima.iter().filter(|&&m| m > 0.0).count() * 2 < maxima.len() {
        return (
            Decision::Reject(
                "traffic too sparse for a null (most shifted replicates score nothing)".into(),
            ),
            1.0,
            None,
        );
    }
    let p = global_p(cand.j, maxima);
    if p > TF_P_ACCEPT {
        return (
            Decision::Reject(format!("chance could explain it (p {p:.3})")),
            p,
            None,
        );
    }
    let Some(&(top, _)) = cand.maps.iter().max_by(|a, b| a.1.total_cmp(&b.1)) else {
        return (Decision::Reject("no contributing aircraft".into()), p, None);
    };
    let (strongest, b0) = (maps[top].aircraft, maps[top].bearing_deg);
    let without = |keep: &dyn Fn(usize) -> bool| {
        let shift = |a: usize| if keep(a) { 0.0 } else { f64::INFINITY };
        global_p(joint_at(maps, idx, cand.cell, cand.delta, &shift).0, maxima)
    };
    let p_drop = without(&|a| a != strongest);
    if p_drop > TF_P_DROP {
        return (
            Decision::Reject(format!(
                "rests on one aircraft (p {p_drop:.3} without the strongest)"
            )),
            p,
            Some(p_drop),
        );
    }
    let off: Vec<usize> = cand
        .maps
        .iter()
        .filter(|&&(m, _)| different_geometry(maps[m].bearing_deg, b0))
        .map(|&(m, _)| maps[m].aircraft)
        .collect();
    if off.is_empty() || without(&|a| off.contains(&a)) > TF_P_DROP {
        return (Decision::OneStream, p, Some(p_drop));
    }
    (Decision::Accept, p, Some(p_drop))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::TF_Z_FLOOR;
    use crate::trackfirst::joint::candidates;
    use crate::trackfirst::joint::tests::mk;

    fn z(c: f64) -> f32 {
        (TF_Z_FLOOR + c) as f32
    }

    #[test]
    fn rng_is_deterministic() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..10 {
            assert_eq!(a.uniform(), b.uniform());
        }
        let x = Rng::new(8).uniform();
        assert!((0.0..1.0).contains(&x));
    }

    #[test]
    fn shifts_stay_inside_span() {
        let mut rng = Rng::new(1);
        let span = (-20_000.0, 20_000.0);
        // One extent in the middle, two near the span's edges.
        let ext = [
            (-100.0, 100.0),
            (-19_900.0, -19_800.0),
            (19_000.0, 19_900.0),
        ];
        for _ in 0..1000 {
            for (u, &(a, b)) in draw_shifts(&mut rng, &ext, span, 900.0).iter().zip(&ext) {
                if let Some(u) = *u {
                    assert!(u.abs() >= 900.0 && u.abs() <= TF_SHIFT_WINDOW_S, "{u}");
                    assert!(a + u >= span.0 && b + u <= span.1);
                }
            }
        }
    }

    #[test]
    fn gpd_recovers_exponential() {
        let mut rng = Rng::new(3);
        let x: Vec<f64> = (0..5000).map(|_| -(1.0 - rng.uniform()).ln()).collect();
        let (xi, sigma) = fit_gpd(&x).unwrap();
        assert!(xi.abs() < 0.1, "ξ {xi}");
        assert!((sigma - 1.0).abs() < 0.1, "σ {sigma}");
    }

    #[test]
    fn p_is_empirical_inside_the_replicates() {
        let m: Vec<f64> = (1..=200).map(f64::from).collect();
        assert!((global_p(100.5, &m) - 101.0 / 201.0).abs() < 1e-12);
    }

    #[test]
    fn p_extrapolates_beyond_the_replicates() {
        let mut rng = Rng::new(5);
        let m: Vec<f64> = (0..200).map(|_| -(1.0 - rng.uniform()).ln()).collect();
        let top = m.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let p = global_p(top + 3.0, &m);
        assert!(p > 0.0 && p < 1.0 / 201.0, "{p}");
    }

    #[test]
    fn bearings_compare_as_lines() {
        assert!(!different_geometry(10.0, 190.0));
        assert!(!different_geometry(350.0, 15.0));
        assert!(different_geometry(0.0, 45.0));
        assert!(different_geometry(0.0, 135.0));
    }

    /// Noise-like traffic: `n` aircraft at cell 0 whose maps span ±12 000 s, each with hits just
    /// over the floor at ~1% of δ steps (as silent audio gives), bearings cycling; ids from
    /// `first`.
    fn background(first: usize, n: usize) -> Vec<PairMap> {
        let mut rng = Rng::new(first as u64 * 1000 + n as u64);
        (0..n)
            .map(|i| {
                let a = first + i;
                let mut steps: Vec<u32> = (0..240)
                    .map(|_| (rng.uniform() * 24_000.0) as u32)
                    .collect();
                steps.sort_unstable();
                steps.dedup();
                let hs: Vec<(u32, f32)> = steps.iter().map(|&s| (s, z(rng.uniform()))).collect();
                let mut m = mk(a, 0, 2000.0 + 400.0 * i as f64, (30 + 7 * i) as u16, &hs);
                m.delta0 = -12_000.0;
                m.n_steps = 24_000;
                m.bearing_deg = (37.0 * i as f64) % 360.0;
                m
            })
            .collect()
    }

    fn null_for(maps: &[PairMap], n_aircraft: usize) -> Vec<f64> {
        let idx = Index::new(maps, 1);
        let ext = vec![(-200.0, 200.0); n_aircraft];
        replicate_maxima(
            maps,
            &idx,
            &ext,
            (-12_000.0, 12_000.0),
            900.0,
            (0.0, 1000.0),
            11,
            TF_REPLICATES,
        )
    }

    #[test]
    fn one_aircraft_never_accepts() {
        // One very strong aircraft amid 24 noise-like ones: rule 3 (or 2) must stop it.
        let mut maps = background(1, 24);
        maps.push(mk(0, 0, 500.0, 50, &[(100, z(50.0))]));
        let maxima = null_for(&maps, 25);
        let idx = Index::new(&maps, 1);
        for c in candidates(&maps, &idx, (0.0, 1000.0), &[]) {
            let (d, _, _) = decide(&maps, &idx, &c, &maxima);
            assert_ne!(d, Decision::Accept, "{c:?}");
        }
    }

    #[test]
    fn two_aircraft_same_bearing_is_one_stream() {
        let mut maps = background(2, 24);
        let mut a = mk(0, 0, 500.0, 50, &[(100, z(7.0))]);
        let mut b = mk(1, 0, 900.0, 90, &[(100, z(7.0))]);
        a.bearing_deg = 40.0;
        b.bearing_deg = 50.0;
        maps.push(a);
        maps.push(b);
        let maxima = null_for(&maps, 26);
        let idx = Index::new(&maps, 1);
        let c = &candidates(&maps, &idx, (0.0, 1000.0), &[])[0];
        assert_eq!(c.delta, 100.0);
        assert_eq!(decide(&maps, &idx, c, &maxima).0, Decision::OneStream);
    }

    #[test]
    fn two_aircraft_different_geometry_accept() {
        let mut maps = background(2, 24);
        let mut a = mk(0, 0, 500.0, 50, &[(100, z(7.0))]);
        let mut b = mk(1, 0, 900.0, 90, &[(100, z(7.0))]);
        a.bearing_deg = 40.0;
        b.bearing_deg = 130.0;
        maps.push(a);
        maps.push(b);
        let maxima = null_for(&maps, 26);
        let idx = Index::new(&maps, 1);
        let c = &candidates(&maps, &idx, (0.0, 1000.0), &[])[0];
        let (d, p, pd) = decide(&maps, &idx, c, &maxima);
        assert_eq!(d, Decision::Accept, "p {p} p_drop {pd:?}");
    }

    #[test]
    fn drop_strongest_guards_a_single_carrier() {
        // One aircraft at the cap, one barely over the floor, different bearings.
        let mut maps = background(2, 24);
        let mut a = mk(0, 0, 500.0, 50, &[(100, z(7.0))]);
        let mut b = mk(1, 0, 900.0, 90, &[(100, z(0.2))]);
        a.bearing_deg = 0.0;
        b.bearing_deg = 90.0;
        maps.push(a);
        maps.push(b);
        let maxima = null_for(&maps, 26);
        let idx = Index::new(&maps, 1);
        let c = candidates(&maps, &idx, (0.0, 1000.0), &[])
            .into_iter()
            .find(|c| c.delta == 100.0)
            .unwrap();
        let (d, _, pd) = decide(&maps, &idx, &c, &maxima);
        assert!(
            matches!(d, Decision::Reject(ref w) if w.contains("one aircraft")),
            "{d:?} {pd:?}"
        );
    }

    #[test]
    fn sparse_traffic_gives_no_null() {
        // The loaded span is the recording only: no aircraft can be shifted.
        let mut maps = background(2, 4);
        maps.push(mk(0, 0, 500.0, 50, &[(100, z(7.0))]));
        maps.push(mk(1, 0, 900.0, 90, &[(100, z(7.0))]));
        let idx = Index::new(&maps, 1);
        let ext = vec![(-200.0, 200.0); 6];
        let maxima = replicate_maxima(
            &maps,
            &idx,
            &ext,
            (-300.0, 300.0),
            900.0,
            (0.0, 1000.0),
            1,
            TF_REPLICATES,
        );
        assert!(maxima.is_empty());
        let c = &candidates(&maps, &idx, (0.0, 1000.0), &[])[0];
        assert_eq!(
            decide(&maps, &idx, c, &maxima).0,
            Decision::Reject("no usable null".into())
        );
    }

    #[test]
    fn shiftability_is_decided_once() {
        let span = (-12_000.0, 12_000.0);
        // In the middle with room either side: shiftable.
        assert!(shiftable((-200.0, 200.0), span, 900.0));
        // Airborne all through the span (several rotations): never shiftable.
        assert!(!shiftable((-11_900.0, 11_900.0), span, 900.0));
        // Only a narrow window of shifts fits: still shiftable, and every draw finds it.
        let ext = [(-11_000.0, 9_000.0)];
        assert!(shiftable(ext[0], span, 900.0));
        let mut rng = Rng::new(9);
        for _ in 0..500 {
            let u = draw_shifts(&mut rng, &ext, span, 900.0)[0].expect("feasible");
            assert!(
                u.abs() >= 900.0 && ext[0].0 + u >= span.0 && ext[0].1 + u <= span.1,
                "{u}"
            );
        }
    }

    #[test]
    fn unshiftable_aircraft_are_left_out() {
        // Aircraft 1 is airborne through the whole span: the null could never move it, so it
        // must not count in the observed score either.
        let maps = vec![
            mk(0, 0, 500.0, 50, &[(100, z(7.0))]),
            mk(1, 0, 900.0, 90, &[(100, z(7.0))]),
        ];
        let ext = vec![(-200.0, 200.0), (-11_990.0, 11_990.0)];
        let (kept, dropped) = drop_unshiftable(maps, &ext, (-12_000.0, 12_000.0), 900.0);
        assert_eq!(dropped, 1);
        assert!(kept.iter().all(|m| m.aircraft == 0));
    }

    #[test]
    fn degenerate_null_rejects() {
        // Every replicate scored nothing: the null can't calibrate any J.
        let mut a = mk(0, 0, 500.0, 50, &[(100, z(7.0))]);
        let mut b = mk(1, 0, 900.0, 90, &[(100, z(7.0))]);
        a.bearing_deg = 0.0;
        b.bearing_deg = 90.0;
        let maps = vec![a, b];
        let idx = Index::new(&maps, 1);
        let c = &candidates(&maps, &idx, (0.0, 1000.0), &[])[0];
        let maxima = vec![0.0; TF_REPLICATES];
        let (d, _, _) = decide(&maps, &idx, c, &maxima);
        assert!(
            matches!(d, Decision::Reject(ref w) if w.contains("too sparse")),
            "{d:?}"
        );
    }
}
