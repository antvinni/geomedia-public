//! The track-first run: maps → joint → null and decision → fine check → verdict.

use std::time::Instant;

use gm_core::{Geodetic, LocalFrame};
use gm_doppler::audio::guide::{GuideImage, coarsen};
use serde::Serialize;

use super::fine::{Contributor, fine, still_fixes};
use super::joint::{Cand, Index, candidates, floor_cap, joint_at};
use super::maps::{PairMap, pair_maps, slack_s};
use super::null::{Decision, decide, drop_unshiftable, replicate_maxima};
use crate::grid::{Cell, Inputs, grid};
use crate::guided::Band;
use crate::params::{
    GUIDE_SPAN_S, TF_AGREE_M, TF_CELL_M, TF_COARSE_BPO, TF_COARSE_HOP_S, TF_FINE_RADIUS_M,
    TF_FINE_STEP_M, TF_REPLICATES,
};
use crate::search::{
    MAX_REGION_KM2, MIN_MARGIN_LL, MIRROR_FRAC, MIRROR_MIN_M, Outcome, Ranked, Verdict, softmax,
};

/// A track-first result: the outcome (same shape as the blind search's), the decision's
/// numbers, the aircraft that carried it, and how long each stage took.
#[derive(Clone, Debug, Serialize)]
pub struct TfOutcome {
    pub outcome: Outcome,
    pub p: Option<f64>,
    pub p_drop: Option<f64>,
    /// Null replicates used (after discarding those with fewer than 2 shiftable aircraft).
    pub replicates: usize,
    pub pairs: usize,
    /// Aircraft left out because the null could never shift them (airborne through the span).
    pub unshiftable: usize,
    pub aircraft: usize,
    pub contributors: Vec<Contributor>,
    pub timings: Vec<(&'static str, f64)>,
    /// Diagnostic with a known truth: aircraft contributing at the truth's coarse cell, at its
    /// best δ.
    pub truth_probe: Option<usize>,
    /// Maps whose δ window covers the winning δ at the winning cell (traffic density there).
    pub overlapping: Option<usize>,
}

fn area_frame(inp: &Inputs) -> (Geodetic, LocalFrame) {
    let (clat, clon) = inp.area.centre();
    let o = Geodetic {
        lat_deg: clat,
        lon_deg: clon,
        h_m: 0.0,
    };
    (o, LocalFrame::new(o))
}

fn sep(a: &Cell, b: &Cell) -> f64 {
    ((a.east - b.east).powi(2) + (a.north - b.north).powi(2)).sqrt()
}

/// The δ prior of `inp`: (−minus, +plus).
fn prior(inp: &Inputs) -> (f64, f64) {
    (-inp.when.minus_s, inp.when.plus_s)
}

/// Each aircraft's (first, last) knot time over all its tracks (NaN pair when none).
fn extents(inp: &Inputs) -> Vec<(f64, f64)> {
    inp.aircraft
        .iter()
        .map(|a| {
            a.tracks
                .iter()
                .flat_map(|t| t.knots.iter().map(|k| k.t))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), t| {
                    (lo.min(t), hi.max(t))
                })
        })
        .collect()
}

/// Candidates near each other in place and time are one answer: the strongest of each cluster.
fn cluster(cands: Vec<Cand>, maps: &[PairMap], cells: &[Cell]) -> Vec<Cand> {
    let mut heads: Vec<Cand> = Vec::new();
    for c in cands {
        let slack = c
            .maps
            .iter()
            .map(|&(m, _)| slack_s(TF_CELL_M, maps[m].v_mps))
            .fold(0.0, f64::max);
        let near = heads.iter().any(|h| {
            sep(&cells[h.cell], &cells[c.cell]) <= TF_FINE_RADIUS_M
                && (h.delta - c.delta).abs() <= 2.0 * slack
        });
        if !near {
            heads.push(c);
        }
    }
    heads
}

/// The coarse log-likelihood of every cell at δ: Σ ½·(floor + contribution)² over the aircraft
/// counted there.
fn coarse_ll(maps: &[PairMap], idx: &Index, n: usize, delta: f64) -> Vec<f64> {
    (0..n)
        .map(|c| {
            joint_at(maps, idx, c, delta, &|_| 0.0)
                .1
                .iter()
                .map(|&(m, x)| {
                    let z = floor_cap(maps[m].band).0 + x;
                    0.5 * z * z
                })
                .sum()
        })
        .collect()
}

fn ranked(c: &Cell, ll: f64, prob: f64) -> Ranked {
    Ranked {
        geo: c.geo,
        east: c.east,
        north: c.north,
        ll,
        prob,
    }
}

/// Track-first location for `inp` (its `aircraft` the whole loaded traffic, its `when` the δ
/// prior; `decoys` unused) from the full-resolution guide images `imgs`. `seed_tc`: blind
/// events' CPA times (audio clock), used only to seed candidate δ values. `traffic_span`: the
/// track-clock span the traffic was loaded for (null shifts stay inside it). `seed`: the null's
/// random seed. `truth`: an evaluation-only (lat, lon) for `truth_probe`.
pub fn track_first(
    imgs: &[(Band, GuideImage)],
    inp: &Inputs,
    seed: u64,
    seed_tc: &[f64],
    traffic_span: (f64, f64),
    truth: Option<(f64, f64)>,
) -> TfOutcome {
    let mut timings = Vec::new();
    let (origin, frame) = area_frame(inp);
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let cells = grid(inp.area, &frame, TF_CELL_M, &ground);
    let mut out = TfOutcome {
        outcome: Outcome {
            verdict: Verdict::Inconclusive(String::new()),
            best: None,
            top: vec![],
            mirror: None,
            delta_s: 0.0,
            matched: vec![],
            silent: vec![],
            region90_km2: 0.0,
            on_boundary: false,
            heat: vec![],
            frame_origin: origin,
            wind_enu: None,
            atmosphere: inp.atmos.describe(),
        },
        p: None,
        p_drop: None,
        replicates: 0,
        pairs: 0,
        unshiftable: 0,
        aircraft: inp.aircraft.len(),
        contributors: vec![],
        timings: vec![],
        truth_probe: None,
        overlapping: None,
    };
    let stop = |mut o: TfOutcome, why: &str, timings: Vec<(&'static str, f64)>| {
        o.outcome.verdict = Verdict::Inconclusive(why.to_string());
        o.timings = timings;
        o
    };
    // Stage 1.
    let t = Instant::now();
    let coarse: Vec<(Band, GuideImage)> = imgs
        .iter()
        .map(|(b, i)| (*b, coarsen(i, TF_COARSE_HOP_S, TF_COARSE_BPO)))
        .collect();
    let range = (traffic_span.0 - inp.duration_s, traffic_span.1);
    let maps: Vec<PairMap> = coarse
        .iter()
        .flat_map(|(b, img)| pair_maps(img, *b, inp, &frame, &cells, range, TF_CELL_M))
        .collect();
    timings.push(("maps", t.elapsed().as_secs_f64()));
    // Aircraft the null can never shift can't be calibrated: they don't count at all.
    let min_abs = inp.duration_s + 2.0 * GUIDE_SPAN_S;
    let ext = extents(inp);
    let (maps, unshiftable) = drop_unshiftable(maps, &ext, traffic_span, min_abs);
    out.unshiftable = unshiftable;
    out.pairs = maps.len();
    if maps.is_empty() || cells.is_empty() {
        return stop(out, "no aircraft in range of the area", timings);
    }
    let idx = Index::new(&maps, cells.len());
    if let Some((la, lo)) = truth {
        let near = (0..cells.len()).min_by(|&a, &b| {
            let d = |c: &Cell| (c.geo.lat_deg - la).powi(2) + (c.geo.lon_deg - lo).powi(2);
            d(&cells[a]).total_cmp(&d(&cells[b]))
        });
        out.truth_probe = near.map(|c| {
            candidates(&maps, &idx, prior(inp), &[])
                .into_iter()
                .find(|x| x.cell == c)
                .map_or(0, |x| x.maps.len())
        });
    }
    // Stage 2.
    let t = Instant::now();
    let seeds: Vec<(usize, f64)> = maps
        .iter()
        .flat_map(|m| seed_tc.iter().map(move |tc| (m.cell, m.t_arrive - tc)))
        .collect();
    let cands = candidates(&maps, &idx, prior(inp), &seeds);
    timings.push(("joint", t.elapsed().as_secs_f64()));
    if cands.is_empty() {
        return stop(out, "no aircraft's curve clears its floor", timings);
    }
    // Stage 3.
    let t = Instant::now();
    let maxima = replicate_maxima(
        &maps,
        &idx,
        &ext,
        traffic_span,
        min_abs,
        prior(inp),
        seed,
        TF_REPLICATES,
    );
    timings.push(("null", t.elapsed().as_secs_f64()));
    out.replicates = maxima.len();
    let mut accepted: Vec<(Cand, f64, Option<f64>)> = Vec::new();
    let mut stream: Option<(Cand, f64)> = None;
    let mut first_reason: Option<String> = None;
    for c in cands {
        let (d, p, pd) = decide(&maps, &idx, &c, &maxima);
        match d {
            Decision::Accept => accepted.push((c, p, pd)),
            Decision::OneStream => {
                if stream.is_none() {
                    stream = Some((c, p));
                }
            }
            Decision::Reject(why) => {
                if first_reason.is_none() {
                    first_reason = Some(why);
                }
            }
        }
    }
    if accepted.is_empty() {
        if let Some((c, p)) = stream {
            let cell = &cells[c.cell];
            out.outcome.verdict = Verdict::AmbiguousStream;
            out.outcome.best = Some(ranked(cell, c.j, 1.0));
            out.outcome.delta_s = c.delta;
            out.p = Some(p);
            out.timings = timings;
            return out;
        }
        let why = first_reason.unwrap_or_else(|| "no candidate".into());
        return stop(out, &why, timings);
    }
    // Stage 4, one fine run per cluster of accepted candidates.
    let t = Instant::now();
    let pmap: Vec<(Cand, f64, Option<f64>)> = accepted;
    let heads = cluster(pmap.iter().map(|x| x.0.clone()).collect(), &maps, &cells);
    let mut survivors = Vec::new();
    for h in &heads {
        let Some(f) = fine(imgs, &coarse, inp, &frame, &maps, &cells, h) else {
            continue;
        };
        if still_fixes(&f.contributors) {
            let (p, pd) = pmap
                .iter()
                .find(|x| x.0 == *h)
                .map_or((None, None), |x| (Some(x.1), x.2));
            survivors.push((h.clone(), f, p, pd));
        }
    }
    timings.push(("fine", t.elapsed().as_secs_f64()));
    let Some((head, f, p, pd)) = survivors.first() else {
        return stop(
            out,
            "the fine check left fewer than two aircraft with different geometry",
            timings,
        );
    };
    out.timings = timings;
    out.p = *p;
    out.p_drop = *pd;
    out.contributors = f.contributors.clone();
    out.outcome.delta_s = f.best.1;
    out.overlapping = Some(
        idx.cell(head.cell)
            .iter()
            .filter(|&&m| {
                let mp = &maps[m];
                let d = head.delta - mp.delta0;
                d >= 0.0 && d < mp.n_steps as f64
            })
            .count(),
    );
    // Posterior over the fine cells; the 90 % region from the coarse cells at the winning δ.
    let lls: Vec<f64> = f.cells.iter().map(|c| c.1).collect();
    let probs = softmax(&lls);
    let mut order: Vec<usize> = (0..f.cells.len()).collect();
    order.sort_by(|&a, &b| lls[b].total_cmp(&lls[a]));
    let best_cell = f.best.0;
    let best_ll = lls[order[0]];
    let best = ranked(&best_cell, best_ll, probs[order[0]]);
    out.outcome.top = order
        .iter()
        .take(10)
        .map(|&i| ranked(&f.cells[i].0, lls[i], probs[i]))
        .collect();
    let mirror = order
        .iter()
        .find(|&&i| {
            sep(&f.cells[i].0, &best_cell) >= MIRROR_MIN_M && probs[i] >= MIRROR_FRAC * best.prob
        })
        .map(|&i| ranked(&f.cells[i].0, lls[i], probs[i]));
    let cll = coarse_ll(&maps, &idx, cells.len(), head.delta);
    let cp = softmax(&cll);
    out.outcome.heat = cells
        .iter()
        .zip(&cp)
        .map(|(c, p)| (c.east, c.north, *p))
        .collect();
    let mut sorted = cp.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (mut acc, mut n90) = (0.0, 0usize);
    for x in sorted {
        acc += x;
        n90 += 1;
        if acc >= 0.9 {
            break;
        }
    }
    out.outcome.region90_km2 = n90 as f64 * TF_CELL_M * TF_CELL_M / 1e6;
    out.outcome.on_boundary = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        .iter()
        .any(|(dx, dy)| {
            let g = frame.ground_point(
                best_cell.east + dx * TF_FINE_STEP_M,
                best_cell.north + dy * TF_FINE_STEP_M,
                0.0,
            );
            !inp.area.contains(g.lat_deg, g.lon_deg)
        });
    let other = survivors
        .iter()
        .skip(1)
        .find(|s| sep(&s.1.best.0, &best_cell) > TF_AGREE_M)
        .map(|s| ranked(&s.1.best.0, f64::NAN, f64::NAN));
    out.outcome.verdict = if let Some(o) = other {
        out.outcome.mirror = Some(o);
        Verdict::AmbiguousMirror
    } else if mirror.is_some() {
        out.outcome.mirror = mirror;
        Verdict::AmbiguousMirror
    } else if out.outcome.region90_km2 > MAX_REGION_KM2 {
        Verdict::Inconclusive(format!(
            "evidence too weak: the 90% region is {:.1} km²",
            out.outcome.region90_km2
        ))
    } else if best_ll < MIN_MARGIN_LL {
        Verdict::Inconclusive(format!(
            "evidence too weak: only {best_ll:.1} log-likelihood above explaining nothing"
        ))
    } else {
        Verdict::Located
    };
    out.outcome.best = Some(best);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmos::Atmosphere;
    use crate::grid::Candidate;
    use crate::params::TF_SHIFT_WINDOW_S;
    use crate::testutil::{Fixture, fixture};
    use crate::trackfirst::maps::tests::area;
    use gm_doppler::audio::guide::guide_image;
    use gm_sources::area::When;

    /// `n` decoys spread over ±3 h of the recording (outside it ±600 s): other traffic of the
    /// day, at the stationary density a real day has.
    fn spread(f: &Fixture, n: usize) -> Vec<Candidate> {
        let mut d = crate::testutil::decoys(f, n);
        for (k, c) in d.iter_mut().enumerate() {
            // CPA times spread over [600 + dur, 3 h] on either side.
            let lo = 600.0 + f.duration_s;
            let span = TF_SHIFT_WINDOW_S - lo;
            let t = lo + span * (k / 2) as f64 / (n / 2).max(1) as f64;
            let target = if k % 2 == 0 { t } else { -t };
            let old = 3.0 * 3600.0 + 30.0 * k as f64;
            for tr in &mut c.tracks {
                for kn in &mut tr.knots {
                    kn.t += target - old;
                }
            }
        }
        d
    }

    fn run(f: &Fixture, aircraft: &[Candidate], when: When) -> TfOutcome {
        let a = area("40.8800,-73.5150~3km");
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = Inputs {
            events: &[],
            aircraft,
            area: &a,
            when,
            duration_s: f.duration_s,
            ground: &ground,
            atmos: &atmos,
            noise: None,
            decoys: &[],
        };
        let imgs = vec![(Band::Prop, guide_image(&f.audio))];
        let span = (
            -when.minus_s - TF_SHIFT_WINDOW_S,
            when.plus_s + f.duration_s + TF_SHIFT_WINDOW_S,
        );
        track_first(&imgs, &inp, 42, &[], span, None)
    }

    fn when(f: &Fixture, pm: f64) -> When {
        When {
            minus_s: pm,
            plus_s: pm,
            ..f.when
        }
    }

    fn dist(o: &TfOutcome, f: &Fixture) -> f64 {
        let b = o.outcome.best.as_ref().unwrap();
        let p = gm_core::geodetic_to_ecef(b.geo);
        let t = gm_core::geodetic_to_ecef(f.truth.observer);
        ((p.x - t.x).powi(2) + (p.y - t.y).powi(2) + (p.z - t.z).powi(2)).sqrt()
    }

    fn with_traffic(f: &Fixture) -> Vec<Candidate> {
        let mut ac = f.aircraft.clone();
        ac.extend(spread(f, 60));
        ac
    }

    #[test]
    fn crossing_is_located_by_track_first() {
        let f = fixture("suite/crossing.toml", |s| s);
        let o = run(&f, &with_traffic(&f), when(&f, 5.0));
        assert_eq!(
            o.outcome.verdict,
            Verdict::Located,
            "{:#?}",
            o.outcome.verdict
        );
        assert!(dist(&o, &f) < 500.0, "{} m", dist(&o, &f));
        assert!(o.outcome.delta_s.abs() <= 2.0, "δ {}", o.outcome.delta_s);
        assert!(o.p.unwrap() <= crate::params::TF_P_ACCEPT);
        assert_eq!(
            o.contributors.iter().filter(|c| c.counts).count(),
            2,
            "{:#?}",
            o.contributors
        );
    }

    #[test]
    fn neighbouring_cells_do_not_make_a_false_ambiguity() {
        let f = fixture("suite/crossing.toml", |s| s);
        let o = run(&f, &with_traffic(&f), when(&f, 5.0));
        assert_ne!(o.outcome.verdict, Verdict::AmbiguousMirror);
    }

    #[test]
    fn narrow_prior_null_is_not_degenerate() {
        let f = fixture("suite/crossing.toml", |s| s);
        let o = run(&f, &with_traffic(&f), when(&f, 5.0));
        assert!(o.replicates >= TF_REPLICATES / 2, "{}", o.replicates);
    }

    #[test]
    fn two_passes_under_a_wide_prior_do_not_locate() {
        // Each aircraft's own brightness reappears somewhere in a wide prior's null: two
        // passes can't survive dropping the strongest.
        let f = fixture("suite/crossing.toml", |s| s);
        let o = run(&f, &with_traffic(&f), when(&f, 1800.0));
        assert_ne!(
            o.outcome.verdict,
            Verdict::Located,
            "{:?}",
            o.outcome.verdict
        );
    }

    #[test]
    fn single_pass_is_not_located() {
        let f = fixture("suite/single.toml", |s| s);
        let o = run(&f, &with_traffic(&f), when(&f, 5.0));
        assert_ne!(
            o.outcome.verdict,
            Verdict::Located,
            "{:?}",
            o.outcome.verdict
        );
    }

    #[test]
    fn prior_excluding_truth_is_not_located() {
        let f = fixture("suite/crossing.toml", |s| s);
        let mut ac = with_traffic(&f);
        for a in ac.iter_mut().take(f.aircraft.len()) {
            for tr in &mut a.tracks {
                for k in &mut tr.knots {
                    k.t += 7200.0;
                }
            }
        }
        let o = run(&f, &ac, when(&f, 5.0));
        assert_ne!(
            o.outcome.verdict,
            Verdict::Located,
            "{:?}",
            o.outcome.verdict
        );
    }

    #[test]
    fn no_aircraft_in_range_abstains() {
        let f = fixture("suite/crossing.toml", |s| s);
        let o = run(&f, &[], when(&f, 5.0));
        assert!(
            matches!(&o.outcome.verdict, Verdict::Inconclusive(w) if w.contains("no aircraft in range")),
            "{:?}",
            o.outcome.verdict
        );
    }

    #[test]
    fn short_clip_never_locates() {
        let mut f = fixture("suite/crossing.toml", |s| s);
        let n = (30.0 * f64::from(f.audio.sample_rate)) as usize;
        f.audio = std::sync::Arc::new(gm_doppler::audio::Audio::new(
            f.audio.sample_rate,
            f.audio.samples[..n].to_vec(),
        ));
        f.duration_s = 30.0;
        let o = run(&f, &with_traffic(&f), when(&f, 5.0));
        assert_ne!(o.outcome.verdict, Verdict::Located);
    }

    #[test]
    fn narrow_prior_with_sparse_traffic_abstains() {
        // Only the two real aircraft: nothing else can be shifted into the prior.
        let f = fixture("suite/crossing.toml", |s| s);
        let o = run(&f, &f.aircraft.clone(), when(&f, 5.0));
        assert!(
            matches!(&o.outcome.verdict, Verdict::Inconclusive(w) if w.contains("null")),
            "{:?}",
            o.outcome.verdict
        );
    }
}
