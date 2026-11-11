//! Coarse-to-fine search over the area, the posterior, and the verdict (located, ambiguous
//! mirror, or inconclusive). Never a confident pin the evidence doesn't support.

use gm_core::{Geodetic, LocalFrame};
use serde::Serialize;

use crate::atmos::Atmosphere;
use crate::curve::score_cell_refined;
use crate::grid::{Cell, Inputs, cell_at, grid};
use crate::score::{CellScore, score_cell};
use crate::wind::fit_wind;

#[derive(Debug, Clone)]
pub struct SearchParams {
    pub coarse_m: f64,
    pub refine: Vec<f64>,
    pub keep: usize,
    pub threads: usize,
    /// Prior σ (m/s) of the final-stage wind fit; `None` disables the fit.
    pub wind_sigma: Option<f64>,
    /// Atmosphere for the final refinement (e.g. a weather column), the wind fit's base;
    /// the coarse and fine search use `Inputs::atmos`.
    pub final_atmos: Option<Atmosphere>,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            coarse_m: 200.0,
            refine: vec![50.0, 25.0],
            keep: 20,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            wind_sigma: Some(crate::params::WIND_SIGMA_FREE),
            final_atmos: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Verdict {
    Located,
    AmbiguousMirror,
    /// Track-first: chance can't explain the fit, but every aircraft that carries it lies on
    /// one bearing line (an approach stream), so the position along or across it is open.
    AmbiguousStream,
    Inconclusive(String),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ranked {
    pub geo: Geodetic,
    pub east: f64,
    pub north: f64,
    pub ll: f64,
    pub prob: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub verdict: Verdict,
    pub best: Option<Ranked>,
    pub top: Vec<Ranked>,
    pub mirror: Option<Ranked>,
    pub delta_s: f64,
    /// (event index, icao)
    pub matched: Vec<(usize, String)>,
    pub silent: Vec<String>,
    pub region90_km2: f64,
    pub on_boundary: bool,
    /// Coarse-grid probabilities: (east, north, prob).
    pub heat: Vec<(f64, f64, f64)>,
    pub frame_origin: Geodetic,
    /// Fitted horizontal wind offset (east, north; m/s) at the best cell, if a fit ran. A
    /// nuisance that absorbs propagation model error, not a wind measurement.
    pub wind_enu: Option<(f64, f64)>,
    /// The atmosphere the search assumed (before any wind offset).
    pub atmosphere: String,
}

/// Mirror candidates must be at least this far from the best cell.
pub const MIRROR_MIN_M: f64 = 1000.0;
/// …and carry at least this fraction of the best cell's probability.
pub const MIRROR_FRAC: f64 = 0.3;
/// A 90% region larger than this is too weak to report as a location.
pub const MAX_REGION_KM2: f64 = 25.0;
/// A located cell must beat "no event explained" by at least this much.
pub const MIN_MARGIN_LL: f64 = 10.0;

fn score_all(
    inp: &Inputs,
    cells: Vec<Cell>,
    refined: bool,
    threads: usize,
) -> Vec<(Cell, CellScore)> {
    let threads = threads.max(1);
    let chunk = cells.len().div_ceil(threads).max(1);
    let mut out: Vec<(Cell, CellScore)> = Vec::with_capacity(cells.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = cells
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|c| {
                            (
                                *c,
                                if refined {
                                    score_cell_refined(inp, c)
                                } else {
                                    score_cell(inp, c)
                                },
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            if let Ok(mut v) = h.join() {
                out.append(&mut v);
            }
        }
    });
    out
}

/// `f` over `items` on `threads` scoped threads, results in input order.
fn par_map<T: Sync, R: Send>(items: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let chunk = items.len().div_ceil(threads.max(1)).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// Cells to fit a wind at: like `top`, separate modes first (≥ MIRROR_MIN_M apart), so a
/// mirror or second mode is scored under the same model as the best; only cells matching at
/// least two aircraft.
fn wind_candidates(level: &[(Cell, CellScore)], keep: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..level.len())
        .filter(|&i| distinct_aircraft(&level[i].1) >= 2)
        .collect();
    order.sort_by(|&a, &b| {
        level[b]
            .1
            .ll
            .total_cmp(&level[a].1.ll)
            .then(level[a].0.east.total_cmp(&level[b].0.east))
            .then(level[a].0.north.total_cmp(&level[b].0.north))
    });
    let sep = |a: &Cell, b: &Cell| (a.east - b.east).hypot(a.north - b.north);
    let mut out: Vec<usize> = Vec::new();
    for &i in &order {
        if out.len() >= keep / 2 {
            break;
        }
        if out
            .iter()
            .all(|&o| sep(&level[o].0, &level[i].0) >= MIRROR_MIN_M)
        {
            out.push(i);
        }
    }
    for &i in &order {
        if out.len() >= keep {
            break;
        }
        if !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

/// After a wind fit, only cells scored under the fitted model compete (an unfitted cell's
/// still-air score isn't comparable); without any fit, all cells.
fn final_pool(winds: &[Option<(f64, f64)>]) -> Vec<usize> {
    let fitted: Vec<usize> = (0..winds.len()).filter(|&i| winds[i].is_some()).collect();
    if fitted.is_empty() {
        (0..winds.len()).collect()
    } else {
        fitted
    }
}

fn distinct_aircraft(s: &CellScore) -> usize {
    s.matched
        .iter()
        .map(|(_, m)| m.aircraft)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

/// A ranked cell, its score and its fitted wind offset.
type RankedCell<'a> = (Ranked, &'a CellScore, Option<(f64, f64)>);

pub(crate) fn softmax(lls: &[f64]) -> Vec<f64> {
    let m = lls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = lls.iter().map(|l| (l - m).exp()).collect();
    let sum: f64 = w.iter().sum();
    w.into_iter()
        .map(|x| x / sum.max(f64::MIN_POSITIVE))
        .collect()
}

/// Top `keep` cells by score, best first, ties broken by position (deterministic). Half the
/// slots go to the best cells of separate modes (≥ MIRROR_MIN_M apart), so a mirror solution on
/// the other side of a flight track survives refinement.
fn top(mut v: Vec<(Cell, CellScore)>, keep: usize) -> Vec<(Cell, CellScore)> {
    v.sort_by(|a, b| {
        b.1.ll
            .total_cmp(&a.1.ll)
            .then(a.0.east.total_cmp(&b.0.east))
            .then(a.0.north.total_cmp(&b.0.north))
    });
    let sep = |a: &Cell, b: &Cell| ((a.east - b.east).powi(2) + (a.north - b.north).powi(2)).sqrt();
    let mut out: Vec<(Cell, CellScore)> = Vec::new();
    for c in &v {
        if out.len() >= keep / 2 {
            break;
        }
        if !c.1.matched.is_empty() && out.iter().all(|o| sep(&o.0, &c.0) >= MIRROR_MIN_M) {
            out.push(c.clone());
        }
    }
    for c in v {
        if out.len() >= keep {
            break;
        }
        if !out
            .iter()
            .any(|o| o.0.east == c.0.east && o.0.north == c.0.north)
        {
            out.push(c);
        }
    }
    out
}

/// The cell mirrored across the matched aircraft's ground track at CPA (a single straight pass
/// can't tell the two sides apart).
fn reflect_across_track(
    inp: &Inputs,
    frame: &LocalFrame,
    best: &Cell,
    score: &CellScore,
    ground: &dyn Fn(f64, f64) -> f64,
) -> Option<Cell> {
    let (ei, m) = score.matched.first()?;
    let a = inp.aircraft.get(m.aircraft)?;
    let ev = inp.events.get(*ei)?;
    let t = ev.fit.tc + m.delta_s;
    let tr = a
        .tracks
        .iter()
        .find_map(|tr| tr.window(t - 180.0, t + 60.0))?;
    let c = inp.atmos.cpa(&tr, best.ecef)?;
    let (p, v) = tr.state_at(c.t_emit)?;
    let (pe, ve) = (frame.to_enu(p), frame.vec_to_enu(v));
    let speed = ve.x.hypot(ve.y);
    if speed < 1e-6 {
        return None;
    }
    let dir = (ve.x / speed, ve.y / speed);
    // Foot of the perpendicular from the best cell onto the ground track, then reflect.
    let along = (best.east - pe.x) * dir.0 + (best.north - pe.y) * dir.1;
    let (fx, fy) = (pe.x + along * dir.0, pe.y + along * dir.1);
    Some(cell_at(
        frame,
        2.0 * fx - best.east,
        2.0 * fy - best.north,
        ground,
    ))
}

pub fn search(inp: &Inputs, p: &SearchParams) -> Outcome {
    let (clat, clon) = inp.area.centre();
    let origin = Geodetic {
        lat_deg: clat,
        lon_deg: clon,
        h_m: 0.0,
    };
    let frame = LocalFrame::new(origin);
    let ground = |la: f64, lo: f64| (inp.ground)(la, lo);
    let coarse = score_all(
        inp,
        grid(inp.area, &frame, p.coarse_m, &ground),
        false,
        p.threads,
    );
    let probs = softmax(&coarse.iter().map(|(_, s)| s.ll).collect::<Vec<_>>());
    let heat: Vec<(f64, f64, f64)> = coarse
        .iter()
        .zip(&probs)
        .map(|((c, _), pr)| (c.east, c.north, *pr))
        .collect();
    let empty = |why: &str| Outcome {
        verdict: Verdict::Inconclusive(why.to_string()),
        best: None,
        top: vec![],
        mirror: None,
        delta_s: 0.0,
        matched: vec![],
        silent: vec![],
        region90_km2: 0.0,
        on_boundary: false,
        heat: heat.clone(),
        frame_origin: origin,
        wind_enu: None,
        atmosphere: inp.atmos.describe(),
    };
    if inp.events.is_empty() {
        return empty("no Doppler events in the recording");
    }
    if coarse.iter().all(|(_, s)| s.matched.is_empty()) {
        return empty("no aircraft match the events (check --when, --area and the ADS-B coverage)");
    }
    // Coarse to fine around the best cells, with exact-curve scoring.
    let mut kept = top(coarse, p.keep);
    let mut last_step = p.coarse_m;
    let mut level: Vec<(Cell, CellScore)> = Vec::new();
    for &step in &p.refine {
        let mut cells: Vec<Cell> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for (c, _) in &kept {
            for j in -2i32..=2 {
                for i in -2i32..=2 {
                    let (e, n) = (c.east + f64::from(i) * step, c.north + f64::from(j) * step);
                    let key = ((e / step).round() as i64, (n / step).round() as i64);
                    if seen.insert(key) {
                        let cell = cell_at(&frame, e, n, &ground);
                        if inp.area.contains(cell.geo.lat_deg, cell.geo.lon_deg) {
                            cells.push(cell);
                        }
                    }
                }
            }
        }
        level = score_all(inp, cells, true, p.threads);
        kept = top(level.clone(), p.keep);
        last_step = step;
    }
    if level.is_empty() {
        level = kept.clone();
    }
    // Final stage: one horizontal wind per cell as a nuisance, at the best cells that match at
    // least two aircraft (one pass plus two wind parameters can fit anything).
    let mut winds: Vec<Option<(f64, f64)>> = vec![None; level.len()];
    let base = p.final_atmos.as_ref().unwrap_or(inp.atmos);
    if let Some(sigma) = p.wind_sigma {
        let cand = wind_candidates(&level, p.keep);
        let fits = par_map(&cand, p.threads, |&i| {
            fit_wind(inp, base, &frame, &level[i].0, sigma)
        });
        for (&i, fit) in cand.iter().zip(fits) {
            if let Some((s, w)) = fit {
                level[i].1 = s;
                winds[i] = Some(w);
            }
        }
        // Re-centre: a patch at the last step around the new best, under its wind.
        let bi = (0..level.len()).max_by(|&a, &b| level[a].1.ll.total_cmp(&level[b].1.ll));
        if let Some(bi) = bi
            && let Some((we, wn)) = winds[bi]
            && let Some(atm) = base.with_wind_offset(&frame, we, wn)
        {
            let sub = Inputs {
                atmos: &atm,
                ..*inp
            };
            let bc = level[bi].0;
            let mut cells = Vec::new();
            for j in -2i32..=2 {
                for i in -2i32..=2 {
                    let (e, n) = (
                        bc.east + f64::from(i) * last_step,
                        bc.north + f64::from(j) * last_step,
                    );
                    let dup = level
                        .iter()
                        .any(|(c, _)| (c.east - e).abs() < 1e-6 && (c.north - n).abs() < 1e-6);
                    let cell = cell_at(&frame, e, n, &ground);
                    if !dup && inp.area.contains(cell.geo.lat_deg, cell.geo.lon_deg) {
                        cells.push(cell);
                    }
                }
            }
            let prior = 0.5 * (we * we + wn * wn) / (sigma * sigma);
            let scored = par_map(&cells, p.threads, |c| {
                let mut s = score_cell_refined(&sub, c);
                s.ll -= prior;
                (*c, s)
            });
            for cs in scored {
                level.push(cs);
                winds.push(Some((we, wn)));
            }
        }
    }
    let pool = final_pool(&winds);
    let level: Vec<(Cell, CellScore)> = pool.iter().map(|&i| level[i].clone()).collect();
    let winds: Vec<Option<(f64, f64)>> = pool.iter().map(|&i| winds[i]).collect();
    let probs = softmax(&level.iter().map(|(_, s)| s.ll).collect::<Vec<_>>());
    let mut ranked: Vec<RankedCell> = level
        .iter()
        .zip(&probs)
        .zip(&winds)
        .map(|(((c, s), pr), w)| {
            (
                Ranked {
                    geo: c.geo,
                    east: c.east,
                    north: c.north,
                    ll: s.ll,
                    prob: *pr,
                },
                s,
                *w,
            )
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.0.ll
            .total_cmp(&a.0.ll)
            .then(a.0.east.total_cmp(&b.0.east))
            .then(a.0.north.total_cmp(&b.0.north))
    });
    let (best, best_score, best_wind) = (ranked[0].0.clone(), ranked[0].1.clone(), ranked[0].2);
    let far = |r: &Ranked| {
        ((r.east - best.east).powi(2) + (r.north - best.north).powi(2)).sqrt() >= MIRROR_MIN_M
    };
    let mut mirror = ranked
        .iter()
        .map(|(r, _, _)| r)
        .find(|r| far(r) && r.prob >= MIRROR_FRAC * best.prob)
        .cloned();
    // One matched pass: score its reflection across the flight track explicitly.
    if mirror.is_none() && best_score.matched.len() == 1 {
        let best_cell = level
            .iter()
            .map(|(c, _)| *c)
            .find(|c| c.east == best.east && c.north == best.north);
        if let Some(bc) = best_cell
            && let Some(rc) = reflect_across_track(inp, &frame, &bc, &best_score, &ground)
            && inp.area.contains(rc.geo.lat_deg, rc.geo.lon_deg)
        {
            // Same atmosphere as the best cell (its fitted wind, if any).
            let rs = match best_wind.and_then(|(e, n)| base.with_wind_offset(&frame, e, n)) {
                Some(atm) => score_cell_refined(
                    &Inputs {
                        atmos: &atm,
                        ..*inp
                    },
                    &rc,
                ),
                None => score_cell_refined(inp, &rc),
            };
            let ratio = (rs.ll - best.ll).exp();
            if ratio >= MIRROR_FRAC {
                mirror = Some(Ranked {
                    geo: rc.geo,
                    east: rc.east,
                    north: rc.north,
                    ll: rs.ll,
                    prob: best.prob * ratio,
                });
            }
        }
    }
    // 90% region: smallest set of evaluated cells holding 90% of the probability.
    // 90% region over the whole area, from the coarse posterior (the refinement patch is far
    // too small to measure ambiguity).
    let mut coarse_p: Vec<f64> = heat.iter().map(|h| h.2).collect();
    coarse_p.sort_by(|a, b| b.total_cmp(a));
    let (mut acc, mut n90) = (0.0, 0usize);
    for pr in coarse_p {
        acc += pr;
        n90 += 1;
        if acc >= 0.9 {
            break;
        }
    }
    let region90_km2 = n90 as f64 * p.coarse_m * p.coarse_m / 1e6;
    let _ = last_step;
    let on_boundary = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        .iter()
        .any(|(dx, dy)| {
            let g = frame.ground_point(
                best.east + dx * p.coarse_m,
                best.north + dy * p.coarse_m,
                0.0,
            );
            !inp.area.contains(g.lat_deg, g.lon_deg)
        });
    let matched: Vec<(usize, String)> = best_score
        .matched
        .iter()
        .map(|(e, m)| (*e, m.icao.clone()))
        .collect();
    let distinct: std::collections::BTreeSet<usize> =
        best_score.matched.iter().map(|(_, m)| m.aircraft).collect();
    let margin = best.ll - inp.events.len() as f64 * crate::params::OUTLIER_LL;
    let verdict = if matched.is_empty() {
        Verdict::Inconclusive("no aircraft match the events at the best cell".into())
    } else if mirror.is_some() {
        Verdict::AmbiguousMirror
    } else if distinct.len() < 2 {
        Verdict::Inconclusive("only one aircraft matched: a single pass can't fix a point (mirror and along-track ambiguity)".into())
    } else if region90_km2 > MAX_REGION_KM2 {
        Verdict::Inconclusive(format!(
            "evidence too weak: the 90% region is {region90_km2:.1} km²"
        ))
    } else if margin < MIN_MARGIN_LL {
        Verdict::Inconclusive(format!(
            "evidence too weak: only {margin:.1} log-likelihood above explaining nothing"
        ))
    } else {
        Verdict::Located
    };
    Outcome {
        verdict,
        best: Some(best),
        top: ranked.iter().take(10).map(|(r, _, _)| r.clone()).collect(),
        mirror,
        delta_s: best_score.delta_s,
        matched,
        silent: best_score.silent.clone(),
        region90_km2,
        on_boundary,
        heat,
        frame_origin: origin,
        wind_enu: best_wind,
        atmosphere: if best_wind.is_some() { base } else { inp.atmos }.describe(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Fixture, fixture};
    use gm_core::geodesic_distance_m;
    use gm_sources::area::{Area, When};

    fn run(f: &Fixture, area: &Area, when: When) -> Outcome {
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        let inp = Inputs {
            events: &f.events,
            aircraft: &f.aircraft,
            area,
            when,
            duration_s: f.duration_s,
            ground: &g,
            atmos: &crate::atmos::Atmosphere::still(),
            noise: None,
            decoys: &[],
        };
        search(&inp, &SearchParams::default())
    }

    fn offset(f: &Fixture, east: f64, north: f64) -> (f64, f64) {
        let g = LocalFrame::new(f.truth.observer).ground_point(east, north, 0.0);
        (g.lat_deg, g.lon_deg)
    }

    fn dist(a: &Geodetic, b: &Geodetic) -> f64 {
        geodesic_distance_m(Geodetic { h_m: 0.0, ..*a }, Geodetic { h_m: 0.0, ..*b })
    }

    fn circle_offset(f: &Fixture, east: f64, north: f64) -> Area {
        let (lat, lon) = offset(f, east, north);
        Area::Circle {
            lat,
            lon,
            radius_m: 5000.0,
        }
    }

    #[test]
    fn locates_in_strong_wind() {
        // 15 m/s: the still-air curve is off by more than the outlier floor allows. A fitted
        // wind offset absorbs it; with f₀ free and the curve term capped, wind is nearly
        // degenerate with f₀ and geometry, so it's a nuisance, not a wind measurement.
        let f = fixture("windy.toml", |s| s);
        let o = run(&f, &circle_offset(&f, 1200.0, -800.0), f.when);
        assert_eq!(
            o.verdict,
            Verdict::Located,
            "{:?} {:?}",
            o.verdict,
            o.matched
        );
        assert_eq!(o.matched.len(), 2, "{:?}", o.matched);
        let d = dist(&o.best.as_ref().unwrap().geo, &f.truth.observer);
        assert!(d < 300.0, "{d} m");
        let (e, n) = o.wind_enu.expect("wind fitted");
        assert!(e.hypot(n) <= 27.0, "wind ({e}, {n})");
    }

    fn scored(east: f64, ll: f64) -> (Cell, CellScore) {
        let m = |aircraft: usize| crate::matching::Match {
            aircraft,
            icao: format!("a{aircraft}"),
            delta_s: 0.0,
            ll: 0.0,
            slant_m: 1000.0,
            speed_mps: 100.0,
            via: crate::matching::MatchVia::Gate,
        };
        (
            Cell {
                east,
                north: 0.0,
                geo: Geodetic {
                    lat_deg: 0.0,
                    lon_deg: 0.0,
                    h_m: 0.0,
                },
                ecef: gm_core::Vec3::ZERO,
            },
            CellScore {
                ll,
                delta_s: 0.0,
                matched: vec![(0, m(0)), (1, m(1))],
                silent: vec![],
            },
        )
    }

    #[test]
    fn wind_candidates_include_other_modes() {
        // 20 cells around the best plus a second mode 2 km away: the far mode must be fitted too,
        // or it keeps its unfitted score and the comparison is between different models.
        let mut level: Vec<(Cell, CellScore)> = (0..20)
            .map(|i| scored(f64::from(i) * 25.0, -10.0 - 0.01 * f64::from(i)))
            .collect();
        level.push(scored(2000.0, -11.0));
        let c = wind_candidates(&level, 20);
        assert!(c.contains(&20), "{c:?}");
        assert!(c.len() <= 20);
    }

    #[test]
    fn final_pool_keeps_one_model() {
        assert_eq!(
            final_pool(&[None, Some((1.0, 0.0)), None, Some((0.0, 0.0))]),
            vec![1, 3]
        );
        assert_eq!(final_pool(&[None, None]), vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "worker")]
    fn par_map_propagates_panics() {
        let _ = par_map(&[1, 2, 3, 4], 2, |x| {
            if *x == 3 {
                panic!("worker");
            }
            *x
        });
    }

    #[test]
    fn final_atmosphere_seeds_wind_fit() {
        // Weather goes into the final refinement only: given the true wind there, the fitted
        // correction stays small and the answer is still located.
        let f = fixture("windy.toml", |s| s);
        let area = circle_offset(&f, 1200.0, -800.0);
        let h = f.truth.observer.h_m;
        let g = move |_: f64, _: f64| h;
        let inp = Inputs {
            events: &f.events,
            aircraft: &f.aircraft,
            area: &area,
            when: f.when,
            duration_s: f.duration_s,
            ground: &g,
            atmos: &crate::atmos::STILL_AIR,
            noise: None,
            decoys: &[],
        };
        let wind = gm_doppler::physics::Medium::new(343.0, f.truth.wind_ecef).unwrap();
        let o = search(
            &inp,
            &SearchParams {
                final_atmos: Some(crate::atmos::Atmosphere::Uniform(wind)),
                wind_sigma: Some(crate::params::WIND_SIGMA_FORECAST),
                ..SearchParams::default()
            },
        );
        assert_eq!(o.verdict, Verdict::Located, "{:?}", o.verdict);
        assert!(o.atmosphere.contains("uniform"), "{}", o.atmosphere);
        let (e, n) = o.wind_enu.expect("fitted");
        assert!(e.hypot(n) < 4.0, "correction ({e}, {n})");
    }

    #[test]
    fn calm_wind_fit_stays_small() {
        let f = fixture("two_props.toml", |s| s);
        let o = run(&f, &circle_offset(&f, 1200.0, -800.0), f.when);
        assert_eq!(o.verdict, Verdict::Located);
        let d = dist(&o.best.as_ref().unwrap().geo, &f.truth.observer);
        assert!(d < 100.0, "{d} m");
        let (e, n) = o.wind_enu.expect("wind fitted");
        assert!(e.hypot(n) <= 8.0, "wind ({e}, {n})");
    }

    #[test]
    fn single_pass_never_fits_wind() {
        let f = fixture("one_pass.toml", |s| s);
        let o = run(&f, &circle_offset(&f, 800.0, 500.0), f.when);
        assert!(o.wind_enu.is_none(), "{:?}", o.wind_enu);
        assert_ne!(o.verdict, Verdict::Located);
    }

    #[test]
    fn locates_decelerating_pass() {
        let f = fixture("decel.toml", |s| s);
        let (lat, lon) = offset(&f, 1200.0, -800.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 5000.0,
            },
            f.when,
        );
        assert_eq!(o.verdict, Verdict::Located, "{o:#?}");
        assert_eq!(o.matched.len(), 2, "{:?}", o.matched);
        let d = dist(&o.best.as_ref().unwrap().geo, &f.truth.observer);
        assert!(d < 300.0, "{d} m");
    }

    #[test]
    fn locates_two_props() {
        let f = fixture("two_props.toml", |s| s);
        let (lat, lon) = offset(&f, 1200.0, -800.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 5000.0,
            },
            f.when,
        );
        assert_eq!(o.verdict, Verdict::Located, "{o:#?}");
        let best = o.best.as_ref().unwrap();
        assert!(
            dist(&best.geo, &f.truth.observer) < 200.0,
            "{} m",
            dist(&best.geo, &f.truth.observer)
        );
        assert!(o.delta_s.abs() < 1.0);
        assert_eq!(o.matched.len(), 2);
        assert!(!o.on_boundary);
    }

    #[test]
    fn locates_two_props_despite_model_error() {
        // Strong wind aloft and 3 harmonics: the still-air curve is systematically off on
        // thousands of points (as on real recordings); the match evidence must still win.
        let f = fixture("two_props.toml", |s| {
            s.replace(
                "wind_enu = [3.0, -2.0, 0.0]",
                "wind_enu = [18.0, -12.0, 0.0]",
            )
            .replace("harmonics = [1.0, 0.5]", "harmonics = [1.0, 0.8, 0.6]")
        });
        let (lat, lon) = offset(&f, 1200.0, -800.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 5000.0,
            },
            f.when,
        );
        let best = o.best.as_ref().unwrap();
        assert_eq!(o.matched.len(), 2, "{o:#?}");
        let top: Vec<String> = o
            .top
            .iter()
            .map(|r| format!("{:.0}m ll{:.2}", dist(&r.geo, &f.truth.observer), r.ll))
            .collect();
        assert!(
            dist(&best.geo, &f.truth.observer) < 400.0,
            "{} m; δ {:.2}; {:?}; mirror {:?}; top {top:?}; events {:?}",
            dist(&best.geo, &f.truth.observer),
            o.delta_s,
            o.verdict,
            o.mirror.as_ref().map(|m| dist(&m.geo, &f.truth.observer)),
            f.events
                .iter()
                .map(|e| (e.fit.tc, e.fit.v, e.fit.d))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn single_pass_is_mirror_ambiguous() {
        let f = fixture("one_pass.toml", |s| s);
        let (lat, lon) = offset(&f, 600.0, 400.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 5000.0,
            },
            f.when,
        );
        assert!(o.verdict != Verdict::Located, "{:?}", o.verdict);
        // One pass constrains a line parallel to the track on each side: across-track is the
        // mirror, along-track trades against the ±5 s clock (140 m/s × 5 s ≈ 700 m).
        let near = |r: &Ranked| dist(&r.geo, &f.truth.observer) < 800.0;
        assert!(
            o.top.iter().any(near) || o.mirror.as_ref().is_some_and(near),
            "{o:#?}"
        );
        assert!(
            o.mirror.is_some(),
            "a single straight pass must report its mirror"
        );
    }

    #[test]
    fn no_aircraft_inconclusive() {
        let mut f = fixture("two_props.toml", |s| s);
        f.aircraft.clear();
        let (lat, lon) = offset(&f, 0.0, 0.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 3000.0,
            },
            f.when,
        );
        assert!(
            matches!(&o.verdict, Verdict::Inconclusive(why) if why.contains("no aircraft")),
            "{:?}",
            o.verdict
        );
        assert!(o.best.is_none());
    }

    #[test]
    fn wrong_clock_not_located() {
        let mut f = fixture("two_props.toml", |s| s);
        let (lat, lon) = offset(&f, 0.0, 0.0);
        // A --when 10 min late: the pipeline re-times tracks relative to that centre.
        for a in &mut f.aircraft {
            for t in &mut a.tracks {
                for k in &mut t.knots {
                    k.t -= 600.0;
                }
            }
        }
        let when = When {
            center: f.when.center.plus_seconds(600.0),
            ..f.when
        };
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 3000.0,
            },
            when,
        );
        assert!(o.verdict != Verdict::Located, "{:?}", o.verdict);
    }

    #[test]
    fn truth_outside_area_flags_boundary() {
        let f = fixture("two_props.toml", |s| s);
        let (lat, lon) = offset(&f, 2300.0, 0.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 2000.0,
            },
            f.when,
        );
        assert!(o.on_boundary, "{o:#?}");
    }

    #[test]
    fn one_aircraft_is_never_located() {
        // Reflection outside the area (or anything else) must not turn one pass into a pin.
        let f = fixture("one_pass.toml", |s| s);
        for (e, n, r) in [
            (0.0, 0.0, 1200.0),
            (0.0, -900.0, 1200.0),
            (300.0, 1300.0, 900.0),
        ] {
            let (lat, lon) = offset(&f, e, n);
            let o = run(
                &f,
                &Area::Circle {
                    lat,
                    lon,
                    radius_m: r,
                },
                f.when,
            );
            assert!(
                o.verdict != Verdict::Located,
                "area ({e},{n},{r}): {:?}",
                o.verdict
            );
        }
    }

    #[test]
    fn split_events_of_one_aircraft_count_once() {
        let mut f = fixture("one_pass.toml", |s| s);
        let dup = f.events[0].clone();
        f.events.push(dup);
        let (lat, lon) = offset(&f, 300.0, 200.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 4000.0,
            },
            f.when,
        );
        let distinct: std::collections::BTreeSet<&str> =
            o.matched.iter().map(|(_, i)| i.as_str()).collect();
        assert_eq!(o.matched.len(), distinct.len(), "{:?}", o.matched);
        assert!(o.verdict != Verdict::Located, "{:?}", o.verdict);
    }

    #[test]
    fn region_is_measured_over_the_area() {
        // A single pass leaves a line-shaped ambiguity far larger than the refinement patch.
        let f = fixture("one_pass.toml", |s| s);
        let (lat, lon) = offset(&f, 300.0, 200.0);
        let o = run(
            &f,
            &Area::Circle {
                lat,
                lon,
                radius_m: 4000.0,
            },
            f.when,
        );
        assert!(o.region90_km2 > 0.4, "{}", o.region90_km2);
    }

    #[test]
    fn deterministic() {
        let f = fixture("two_props.toml", |s| s);
        let (lat, lon) = offset(&f, 500.0, 500.0);
        let area = Area::Circle {
            lat,
            lon,
            radius_m: 3000.0,
        };
        let a = run(&f, &area, f.when);
        let b = run(&f, &area, f.when);
        assert_eq!(a.best, b.best);
        assert_eq!(a.top, b.top);
    }
}
