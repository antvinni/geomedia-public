//! Stage 3: location from timing and loudness, for aircraft sounds with no usable curve.
//!
//! Each pass's loudness hump (mid-band excess over the background, 5-s blocks) is compared with
//! the hump an ADS-B aircraft would make at a candidate cell: received level ∝ −20 log r − α r at
//! emission time, with a free gain G (unknown source level and phone gain) and a compression
//! slope β (phone AGC), predicted as *excess* 10 log(1 + 10^((G + βL)/10)) so the hump's tails
//! saturate as the observed excess does. Passes share one clock offset; each pass is explained by
//! at most one aircraft (an aircraft may explain several passes). Significance comes from a null
//! that shifts every aircraft independently within ±3 h. The region is descriptive (cells within
//! a fixed margin of the best) and always reported with its p; only a significant, compact,
//! geometrically distinct answer is "located".

use gm_core::{Geodetic, LocalFrame, Vec3};
use gm_doppler::audio::Detection;
use gm_doppler::physics::{Medium, Track, emission_time};
use serde::Serialize;

use crate::grid::{Candidate, Cell, Inputs, grid};
use crate::trackfirst::null::{Rng, draw_shifts, global_p};

/// One pass's loudness: 5-s block means of the mid-band excess (dB) at block centres (s).
#[derive(Clone, Debug, Serialize)]
pub struct PassEnvelope {
    pub pass_id: usize,
    pub t: Vec<f64>,
    pub level_db: Vec<f64>,
    pub sigma_db: f64,
    /// The pass is cut by the recording start or end.
    pub cut: bool,
}

#[derive(Clone, Debug)]
pub struct EnvParams {
    pub cell_m: f64,
    pub delta_step_s: f64,
    pub replicates: usize,
    pub p_accept: f64,
    pub p_drop: f64,
    pub compact_m: f64,
    /// Region: cells within this log-likelihood margin of the best.
    pub margin_ll: f64,
    pub betas: Vec<f64>,
    /// Aircraft farther than this (slant) from a cell are not considered there.
    pub rmax_m: f64,
    pub max_area_km2: f64,
    /// Shifts in the null are at least this far from zero.
    pub min_shift_s: f64,
    pub seed: u64,
    pub threads: usize,
}

impl Default for EnvParams {
    fn default() -> Self {
        Self {
            cell_m: 1000.0,
            delta_step_s: 2.0,
            replicates: 100,
            p_accept: 0.01,
            p_drop: 0.05,
            compact_m: 1500.0,
            margin_ll: 3.0,
            betas: vec![0.5, 0.75, 1.0],
            rmax_m: 12_000.0,
            max_area_km2: 2500.0,
            min_shift_s: 600.0,
            seed: 0x5eed_e1e0,
            threads: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvVerdict {
    Located {
        lat: f64,
        lon: f64,
        radius_m: f64,
    },
    /// Conditional on the identification: a strip (or two mirror strips) of plausible positions.
    Band {
        lat: f64,
        lon: f64,
        length_m: f64,
        width_m: f64,
        axis_deg: f64,
        mirror: bool,
    },
    Ambiguous {
        regions: usize,
    },
    Inconclusive {
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct EnvOutcome {
    pub verdict: EnvVerdict,
    pub p: f64,
    /// p with the strongest pass dropped (None with a single pass).
    pub p_drop: Option<f64>,
    pub delta_s: f64,
    /// Per pass: its id, the aircraft explaining it at the best cell, and its score.
    pub per_pass: Vec<(usize, Option<String>, f64)>,
    /// The descriptive region (lat, lon of its cells).
    pub region: Vec<(f64, f64)>,
    pub best: Option<(f64, f64)>,
    pub notes: Vec<String>,
}

/// Mean ISO 9613-1 attenuation (dB/m) of the four mid octaves (centres 141–1131 Hz), standard air.
pub fn mid_band_alpha() -> f64 {
    use gm_doppler::physics::absorption::{STD_P_KPA, STD_RH_PCT, STD_TEMP_C, alpha_db_per_m};
    let fc = [141.4, 282.8, 565.7, 1131.4];
    fc.iter()
        .map(|&f| alpha_db_per_m(f, STD_TEMP_C, STD_RH_PCT, STD_P_KPA))
        .sum::<f64>()
        / 4.0
}

/// Received level (dB, relative) of `track` at `obs` for receive time `t_rx`: −20 log₁₀(r/1 km) −
/// α r, r the slant range at emission time. None if the sound left outside the track.
pub fn predicted_level_db(
    track: &Track,
    obs: Vec3,
    t_rx: f64,
    m: Medium,
    alpha_db_per_m: f64,
) -> Option<f64> {
    let tau = emission_time(track, obs, t_rx, m, None)?;
    let (pos, _) = track.state_at(tau)?;
    let r = (pos - obs).norm().max(1.0);
    Some(-20.0 * (r / 1000.0).log10() - alpha_db_per_m * r)
}

/// The envelopes of the passes `ids` of `det` (5-s blocks over the pass ± 30 s).
pub fn pass_envelopes(det: &Detection, ids: &[usize]) -> Vec<PassEnvelope> {
    const BLOCK_S: f64 = 5.0;
    const MARGIN_S: f64 = 30.0;
    let env = &det.envelope;
    if env.raw_mid_db.is_empty() || env.dt <= 0.0 {
        return vec![];
    }
    let t_of = |k: usize| env.t0 + k as f64 * env.dt;
    let last = t_of(env.raw_mid_db.len() - 1);
    // Robust per-frame noise: raw minus its running median over a block, MAD-scaled.
    let per_block = (BLOCK_S / env.dt).round().max(1.0) as usize;
    // Frames overlap when longer than the hop: a block holds BLOCK_S / frame_s independent ones.
    let independent = (BLOCK_S / env.frame_s.max(env.dt)).max(1.0);
    let raw: Vec<f64> = env.raw_mid_db.iter().map(|v| f64::from(*v)).collect();
    let mut res: Vec<f64> = (0..raw.len())
        .map(|k| {
            let (a, b) = (
                k.saturating_sub(per_block / 2),
                (k + per_block / 2 + 1).min(raw.len()),
            );
            let mut w = raw[a..b].to_vec();
            w.sort_by(f64::total_cmp);
            (raw[k] - w[w.len() / 2]).abs()
        })
        .collect();
    res.sort_by(f64::total_cmp);
    let mad = res.get(res.len() / 2).copied().unwrap_or(1.0);
    let sigma = (1.4826 * mad / independent.sqrt()).max(0.5);
    det.passes
        .iter()
        .filter(|p| ids.contains(&p.id))
        .map(|p| {
            let (w0, w1) = (
                (p.t_start - MARGIN_S).max(env.t0),
                (p.t_end + MARGIN_S).min(last),
            );
            let mut t = Vec::new();
            let mut y = Vec::new();
            let mut b0 = w0;
            while b0 + BLOCK_S <= w1 + 1e-9 {
                let v: Vec<f64> = (0..raw.len())
                    .filter(|&k| {
                        let tk = t_of(k);
                        tk >= b0 && tk < b0 + BLOCK_S
                    })
                    .map(|k| raw[k])
                    .collect();
                if !v.is_empty() {
                    t.push(b0 + BLOCK_S / 2.0);
                    y.push(v.iter().sum::<f64>() / v.len() as f64);
                }
                b0 += BLOCK_S;
            }
            PassEnvelope {
                pass_id: p.id,
                t,
                level_db: y,
                sigma_db: sigma,
                cut: p
                    .flags
                    .iter()
                    .any(|f| f == "cut_at_start" || f == "cut_at_end"),
            }
        })
        .collect()
}

/// Received level along receive time (s, relative to `when.center`) for one aircraft at one
/// cell, on a regular grid; NaN where the aircraft is beyond `rmax`.
struct Lut {
    t0: f64,
    dt: f64,
    vals: Vec<f32>,
}

impl Lut {
    fn at(&self, t: f64) -> Option<f64> {
        let x = (t - self.t0) / self.dt;
        if x < 0.0 {
            return None;
        }
        let i = x.floor() as usize;
        if i + 1 >= self.vals.len() {
            return None;
        }
        let (a, b) = (self.vals[i], self.vals[i + 1]);
        if !(a.is_finite() && b.is_finite()) {
            return None;
        }
        let f = x - i as f64;
        Some(f64::from(a) * (1.0 - f) + f64::from(b) * f)
    }
}

/// Best (score, β, G) of one pass against levels `x` (NaN-free; far = very low).
fn fit_pass(y: &[f64], x: &[f64], sigma: f64, betas: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 {
        return 0.0;
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let chi_flat: f64 = y.iter().map(|v| ((v - mean) / sigma).powi(2)).sum();
    // The hump's core: blocks well above the background (at least the top two).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| y[b].total_cmp(&y[a]));
    let core: Vec<usize> = order
        .iter()
        .copied()
        .enumerate()
        .filter(|&(rank, i)| rank < 2 || y[i] >= 3.0)
        .map(|(_, i)| i)
        .collect();
    let chi = |g: f64, beta: f64| -> f64 {
        y.iter()
            .zip(x)
            .map(|(&yv, &xv)| {
                let pred = 10.0 * (1.0 + 10f64.powf((g + beta * xv) / 10.0)).log10();
                ((yv - pred) / sigma).powi(2)
            })
            .sum()
    };
    let mut best = f64::INFINITY;
    for &beta in betas {
        let g0 = core.iter().map(|&i| y[i] - beta * x[i]).sum::<f64>() / core.len() as f64;
        let (mut g, mut c) = (g0, chi(g0, beta));
        for step in [4.0, 2.0, 1.0, 0.5, 0.25] {
            for dg in [-step, step] {
                let cc = chi(g + dg, beta);
                if cc < c {
                    (g, c) = (g + dg, cc);
                }
            }
        }
        best = best.min(c);
    }
    0.5 * (chi_flat - best)
}

fn inconclusive(reason: &str) -> EnvOutcome {
    EnvOutcome {
        verdict: EnvVerdict::Inconclusive {
            reason: reason.into(),
        },
        p: 1.0,
        p_drop: None,
        delta_s: 0.0,
        per_pass: vec![],
        region: vec![],
        best: None,
        notes: vec![],
    }
}

/// Per cell: (aircraft index, lookup) for every aircraft that comes within `rmax` of it.
fn build_luts(
    cells: &[Cell],
    frame: &LocalFrame,
    traffic: &[Candidate],
    m: Medium,
    p: &EnvParams,
    alpha: f64,
) -> Vec<Vec<(usize, Lut)>> {
    use std::collections::HashMap;
    let step = p.cell_m;
    let key = |e: f64, n: f64| ((e / step).round() as i64, (n / step).round() as i64);
    let index: HashMap<(i64, i64), usize> = cells
        .iter()
        .enumerate()
        .map(|(i, c)| (key(c.east, c.north), i))
        .collect();
    let reach = (p.rmax_m / step).ceil() as i64;
    // (cell, aircraft) → receive-time interval when the aircraft is within rmax.
    let mut near: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    for (ai, a) in traffic.iter().enumerate() {
        for tr in &a.tracks {
            for k in &tr.knots {
                let enu = frame.to_enu(k.pos);
                if enu.z.abs() > p.rmax_m {
                    continue;
                }
                let (ce, cn) = key(enu.x, enu.y);
                for di in -reach..=reach {
                    for dj in -reach..=reach {
                        if let Some(&ci) = index.get(&(ce + di, cn + dj)) {
                            let c = &cells[ci];
                            let d2 = (enu.x - c.east).powi(2)
                                + (enu.y - c.north).powi(2)
                                + enu.z.powi(2);
                            if d2 <= p.rmax_m * p.rmax_m {
                                let e = near.entry((ci, ai)).or_insert((k.t, k.t));
                                e.0 = e.0.min(k.t);
                                e.1 = e.1.max(k.t);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut out: Vec<Vec<(usize, Lut)>> = (0..cells.len()).map(|_| Vec::new()).collect();
    let dt = 2.0;
    for ((ci, ai), (t0, t1)) in near {
        let obs = cells[ci].ecef;
        let (r0, r1) = (t0, t1 + p.rmax_m / m.c);
        let n = ((r1 - r0) / dt).ceil() as usize + 2;
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let t = r0 + i as f64 * dt;
            let v = traffic[ai]
                .tracks
                .iter()
                .filter_map(|tr| predicted_level_db(tr, obs, t, m, alpha))
                .fold(f64::NEG_INFINITY, f64::max);
            let r_ok = v > -20.0 * (p.rmax_m / 1000.0).log10() - alpha * p.rmax_m;
            vals.push(if v.is_finite() && r_ok {
                v as f32
            } else {
                f32::NAN
            });
        }
        out[ci].push((ai, Lut { t0: r0, dt, vals }));
    }
    for v in &mut out {
        v.sort_by_key(|x| x.0);
    }
    out
}

/// Per pass: its best score and aircraft.
type PassRow = Vec<(f64, Option<usize>)>;

/// Per (cell, δ): the passes' best scores (one aircraft each) under aircraft shifts `u`.
fn cell_scores(
    luts: &[(usize, Lut)],
    passes: &[PassEnvelope],
    deltas: &[f64],
    u: &[f64],
    betas: &[f64],
) -> Vec<PassRow> {
    deltas
        .iter()
        .map(|&d| {
            passes
                .iter()
                .map(|ps| {
                    let peak =
                        ps.t.iter()
                            .zip(&ps.level_db)
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map_or(0.0, |x| *x.0);
                    let mut best = (0.0, None);
                    for (ai, lut) in luts {
                        let shift = u[*ai];
                        if lut.at(peak + d - shift).is_none() {
                            continue;
                        }
                        let x: Vec<f64> =
                            ps.t.iter()
                                .map(|&t| lut.at(t + d - shift).unwrap_or(-300.0))
                                .collect();
                        let s = fit_pass(&ps.level_db, &x, ps.sigma_db, betas);
                        if s > best.0 {
                            best = (s, Some(*ai));
                        }
                    }
                    best
                })
                .collect()
        })
        .collect()
}

/// (sum, sum without the strongest pass) of one (cell, δ) row.
fn sums(row: &[(f64, Option<usize>)]) -> (f64, f64) {
    let sum: f64 = row.iter().map(|x| x.0).sum();
    let max = row.iter().map(|x| x.0).fold(0.0, f64::max);
    (sum, if row.len() >= 2 { sum - max } else { 0.0 })
}

fn par<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(4, |n| n.get())
    } else {
        threads
    };
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        let hs: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|a| scope.spawn(move || (a..(a + chunk).min(n)).map(f).collect::<Vec<T>>()))
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().expect("envelope worker panicked"))
            .collect()
    })
}

/// Score the passes against `traffic` (tracks timed relative to `inp.when.center`) over the area.
pub fn envelope_search(
    inp: &Inputs,
    traffic: &[Candidate],
    passes: &[PassEnvelope],
    p: &EnvParams,
) -> EnvOutcome {
    let passes: Vec<PassEnvelope> = passes.iter().filter(|x| x.t.len() >= 3).cloned().collect();
    if passes.is_empty() {
        return inconclusive("no aircraft sound long enough for timing and loudness");
    }
    if traffic.is_empty() {
        return inconclusive("no ADS-B aircraft in the traffic window");
    }
    let inp_traffic_len = traffic.len();
    // Aircraft the null can't shift (present for most of the traffic span) are left out of
    // both the observed fit and the null, as track-first does: they would count only in one.
    let extent_of = |a: &Candidate| {
        a.tracks
            .iter()
            .flat_map(|t| t.knots.iter().map(|k| k.t))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, t| {
                (acc.0.min(t), acc.1.max(t))
            })
    };
    let span = traffic
        .iter()
        .map(extent_of)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, e| {
            (acc.0.min(e.0), acc.1.max(e.1))
        });
    let traffic: Vec<Candidate> = traffic
        .iter()
        .filter(|a| crate::trackfirst::null::shiftable(extent_of(a), span, p.min_shift_s))
        .cloned()
        .collect();
    let left_out = inp_traffic_len - traffic.len();
    if traffic.is_empty() {
        return inconclusive(&format!(
            "no ADS-B aircraft the null can shift ({left_out} present for most of the traffic span)"
        ));
    }
    let traffic = &traffic[..];
    let (s, w, n, e) = inp.area.bbox();
    let mid_lat = (s + n) / 2.0;
    let km2 = (n - s) * 111.2 * (e - w) * 111.2 * mid_lat.to_radians().cos();
    if km2 > p.max_area_km2 {
        return inconclusive(&format!(
            "area too large for the timing-and-loudness leg ({km2:.0} km² > {:.0})",
            p.max_area_km2
        ));
    }
    let m = match inp.atmos {
        crate::atmos::Atmosphere::Uniform(m) => *m,
        crate::atmos::Atmosphere::Layered(_) => {
            Medium::still(crate::params::SEARCH_C).unwrap_or(Medium {
                c: 343.0,
                wind: Vec3::ZERO,
            })
        }
    };
    let centre = Geodetic {
        lat_deg: mid_lat,
        lon_deg: (w + e) / 2.0,
        h_m: 0.0,
    };
    let frame = LocalFrame::new(centre);
    let cells = grid(inp.area, &frame, p.cell_m, inp.ground);
    if cells.is_empty() {
        return inconclusive("no cells in the area");
    }
    let alpha = mid_band_alpha();
    let luts = build_luts(&cells, &frame, traffic, m, p, alpha);
    let mut deltas = Vec::new();
    let mut d = -inp.when.minus_s;
    while d <= inp.when.plus_s + 1e-9 {
        deltas.push(d);
        d += p.delta_step_s;
    }
    if deltas.is_empty() {
        deltas.push(0.0);
    }
    let zero = vec![0.0; traffic.len()];
    // Observed.
    let obs: Vec<Vec<PassRow>> = par(cells.len(), p.threads, |ci| {
        cell_scores(&luts[ci], &passes, &deltas, &zero, &p.betas)
    });
    let mut best = (f64::NEG_INFINITY, 0usize, 0usize);
    let mut best_drop = f64::NEG_INFINITY;
    let mut cell_best = vec![f64::NEG_INFINITY; cells.len()];
    for (ci, rows) in obs.iter().enumerate() {
        for (di, row) in rows.iter().enumerate() {
            let (sum, drop) = sums(row);
            cell_best[ci] = cell_best[ci].max(sum);
            best_drop = best_drop.max(drop);
            if sum > best.0 {
                best = (sum, ci, di);
            }
        }
    }
    let (t_obs, bci, bdi) = best;
    if t_obs <= 0.0 {
        return inconclusive(
            "no ADS-B aircraft explains the aircraft sound(s) better than a flat level",
        );
    }
    // Null: every aircraft shifted independently.
    let extents: Vec<(f64, f64)> = traffic.iter().map(extent_of).collect();
    let reps: Vec<(f64, f64)> = par(p.replicates, p.threads, |r| {
        let mut rng = Rng::new(p.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let u: Vec<f64> = draw_shifts(&mut rng, &extents, span, p.min_shift_s)
            .into_iter()
            .map(|x| x.unwrap_or(1e9))
            .collect();
        let mut mx = (0.0f64, 0.0f64);
        for l in &luts {
            for row in cell_scores(l, &passes, &deltas, &u, &p.betas) {
                let (sum, drop) = sums(&row);
                mx = (mx.0.max(sum), mx.1.max(drop));
            }
        }
        mx
    });
    let null: Vec<f64> = reps.iter().map(|x| x.0).collect();
    let empty = null.iter().filter(|&&x| x <= 0.0).count();
    if 2 * empty > null.len() {
        return inconclusive(&format!(
            "traffic too sparse for a null: {empty} of {} shifted replicates explain nothing",
            null.len()
        ));
    }
    let pval = global_p(t_obs, &null);
    let p_drop = (passes.len() >= 2).then(|| {
        let null_d: Vec<f64> = reps.iter().map(|x| x.1).collect();
        global_p(best_drop, &null_d)
    });
    let row = &obs[bci][bdi];
    let per_pass: Vec<(usize, Option<String>, f64)> = passes
        .iter()
        .zip(row)
        .map(|(ps, (sc, ai))| (ps.pass_id, ai.map(|i| traffic[i].icao.clone()), *sc))
        .collect();
    // Descriptive region and its components.
    let region_idx: Vec<usize> = (0..cells.len())
        .filter(|&ci| cell_best[ci] >= t_obs - p.margin_ll)
        .collect();
    let comps = components(&cells, &region_idx, p.cell_m);
    let mut notes = vec![format!(
        "{} cells, {} aircraft ({left_out} unshiftable left out), {} null replicates (p ≥ {:.4}); best score {t_obs:.1}",
        cells.len(),
        traffic.len(),
        p.replicates,
        1.0 / (p.replicates + 1) as f64
    )];
    if deltas.len() > 1 && (bdi == 0 || bdi + 1 == deltas.len()) {
        notes.push(format!(
            "best clock offset at the edge of the clock window (δ = {:+.0} s): the true offset may lie outside it",
            deltas[bdi]
        ));
    }
    // Distinct geometry: the matched aircraft's headings at their passes (as lines).
    let headings: Vec<f64> = passes
        .iter()
        .zip(row)
        .filter_map(|(ps, (_, ai))| {
            let a = &traffic[(*ai)?];
            let peak =
                ps.t.iter()
                    .zip(&ps.level_db)
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|x| *x.0)?;
            let t = peak + deltas[bdi];
            let (_, v) = a.tracks.iter().find_map(|tr| tr.state_at(t))?;
            let enu = frame.vec_to_enu(v);
            Some(enu.x.atan2(enu.y).to_degrees())
        })
        .collect();
    let distinct = headings.len() >= 2
        && headings.iter().enumerate().any(|(i, &a)| {
            headings[i + 1..]
                .iter()
                .any(|&b| crate::trackfirst::null::different_geometry(a, b))
        });
    let geo = |ci: usize| (cells[ci].geo.lat_deg, cells[ci].geo.lon_deg);
    let region: Vec<(f64, f64)> = region_idx.iter().map(|&ci| geo(ci)).collect();
    // Side of the strongest pass's aircraft track for each region cell: one track explains at
    // most two strips (both sides of it); pieces on the same side are one strip.
    let strongest = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let side_of = |ci: usize| -> Option<i8> {
        let ai = row[strongest].1?;
        let ps = &passes[strongest];
        let peak =
            ps.t.iter()
                .zip(&ps.level_db)
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|x| *x.0)?;
        let (pos, vel) = traffic[ai]
            .tracks
            .iter()
            .find_map(|tr| tr.state_at(peak + deltas[bdi]))?;
        let (pe, ve) = (frame.to_enu(pos), frame.vec_to_enu(vel));
        let c = &cells[ci];
        let cross = ve.x * (c.north - pe.y) - ve.y * (c.east - pe.x);
        Some(if cross >= 0.0 { 1 } else { -1 })
    };
    let mut sides: Vec<i8> = comps
        .iter()
        .filter_map(|comp| {
            let tot: i32 = comp
                .iter()
                .filter_map(|&ci| side_of(ci))
                .map(i32::from)
                .sum();
            (tot != 0).then_some(tot.signum() as i8)
        })
        .collect();
    sides.sort_unstable();
    sides.dedup();
    let best_comp: Vec<usize> = comps
        .iter()
        .find(|c| c.contains(&bci))
        .cloned()
        .unwrap_or_else(|| vec![bci]);
    let (len, wid, axis, _, _) = extent(&cells, &best_comp, p.cell_m);
    let band = |mirror: bool| EnvVerdict::Band {
        lat: geo(bci).0,
        lon: geo(bci).1,
        length_m: len,
        width_m: wid,
        axis_deg: axis,
        mirror,
    };
    let significant = pval <= p.p_accept && p_drop.is_some_and(|x| x <= p.p_drop);
    let verdict = if comps.len() == 1 && len <= p.compact_m && significant && distinct {
        EnvVerdict::Located {
            lat: geo(bci).0,
            lon: geo(bci).1,
            radius_m: len / 2.0,
        }
    } else if comps.len() == 1 || (sides.len() <= 1 && passes.len() == 1) {
        if !distinct && passes.len() >= 2 {
            notes.push("the matched aircraft lie on one stream (headings within 30°)".into());
        }
        band(false)
    } else if sides.len() == 2 && (passes.len() == 1 || comps.len() == 2) {
        band(true)
    } else {
        EnvVerdict::Ambiguous {
            regions: comps.len(),
        }
    };
    EnvOutcome {
        verdict,
        p: pval,
        p_drop,
        delta_s: deltas[bdi],
        per_pass,
        region,
        best: Some(geo(bci)),
        notes,
    }
}

/// Combine the tonal answer with the timing-and-loudness one. A located tonal answer stands
/// (the envelope is a cross-check: a conflict is noted); two located answers more than 1 km apart
/// are "methods disagree"; otherwise a located envelope answer becomes the answer, and a band or
/// ambiguity is reported as inconclusive with its region in the heat map. `envelope_only` (dev)
/// takes the envelope's answer whatever the tonal one says.
pub fn combine_with_envelope(
    tonal: crate::search::Outcome,
    env: &EnvOutcome,
    envelope_only: bool,
) -> (crate::search::Outcome, String) {
    use crate::search::{Ranked, Verdict};
    let frame = LocalFrame::new(tonal.frame_origin);
    let ranked = |(lat, lon): (f64, f64), prob: f64| -> Ranked {
        let geo = Geodetic {
            lat_deg: lat,
            lon_deg: lon,
            h_m: 0.0,
        };
        let enu = frame.to_enu(gm_core::geodetic_to_ecef(geo));
        Ranked {
            geo,
            east: enu.x,
            north: enu.y,
            ll: 0.0,
            prob,
        }
    };
    let dist = |a: (f64, f64), b: (f64, f64)| {
        gm_core::geodesic_distance_m(
            Geodetic {
                lat_deg: a.0,
                lon_deg: a.1,
                h_m: 0.0,
            },
            Geodetic {
                lat_deg: b.0,
                lon_deg: b.1,
                h_m: 0.0,
            },
        )
    };
    let heat: Vec<(f64, f64, f64)> = env
        .region
        .iter()
        .map(|&c| {
            let r = ranked(c, 1.0 / env.region.len().max(1) as f64);
            (r.east, r.north, r.prob)
        })
        .collect();
    let region_km2 = env.region.len() as f64; // 1-km cells
    let describe = |v: &EnvVerdict| match v {
        EnvVerdict::Located { radius_m, .. } => {
            format!("located (radius {radius_m:.0} m, p {:.3})", env.p)
        }
        EnvVerdict::Band {
            length_m,
            width_m,
            axis_deg,
            mirror,
            ..
        } => format!(
            "band {:.1} × {:.1} km along {axis_deg:.0}°{} (conditional on the identification, p {:.3})",
            length_m / 1000.0,
            width_m / 1000.0,
            if *mirror {
                ", with its mirror across the track"
            } else {
                ""
            },
            env.p
        ),
        EnvVerdict::Ambiguous { regions } => {
            format!("ambiguous: {regions} separate regions (p {:.3})", env.p)
        }
        EnvVerdict::Inconclusive { reason } => format!("inconclusive: {reason}"),
    };
    let text = describe(&env.verdict);
    let tonal_best = tonal.best.as_ref().map(|b| (b.geo.lat_deg, b.geo.lon_deg));
    if tonal.verdict == Verdict::Located && !envelope_only {
        let tb = tonal_best.unwrap_or((0.0, 0.0));
        if let EnvVerdict::Located { lat, lon, .. } = env.verdict
            && dist(tb, (lat, lon)) > 1000.0
        {
            let km = dist(tb, (lat, lon)) / 1000.0;
            let o = crate::search::Outcome {
                verdict: Verdict::Inconclusive(format!(
                    "methods disagree: tonal and timing-and-loudness answers {km:.1} km apart"
                )),
                ..tonal
            };
            return (
                o,
                format!("timing and loudness: {text}; disagrees with the tonal answer"),
            );
        }
        let excluded = !env.region.is_empty() && env.region.iter().all(|&c| dist(c, tb) > 1000.0);
        let note = if excluded {
            format!(
                "timing and loudness: {text}; its region excludes the tonal answer (conflict noted)"
            )
        } else {
            format!("timing and loudness: {text}; consistent with the tonal answer")
        };
        return (tonal, note);
    }
    match &env.verdict {
        EnvVerdict::Located { lat, lon, .. } => {
            let best = ranked((*lat, *lon), 1.0);
            let o = crate::search::Outcome {
                verdict: Verdict::Located,
                best: Some(best.clone()),
                top: vec![best],
                mirror: None,
                region90_km2: region_km2,
                heat,
                delta_s: env.delta_s,
                ..tonal
            };
            (o, format!("answer from timing and loudness: {text}"))
        }
        EnvVerdict::Inconclusive { .. } => (tonal, format!("timing and loudness: {text}")),
        v => {
            let o = crate::search::Outcome {
                verdict: Verdict::Inconclusive(describe(v)),
                best: env
                    .best
                    .map(|b| ranked(b, 1.0 / env.region.len().max(1) as f64)),
                heat,
                region90_km2: region_km2,
                ..tonal
            };
            (o, format!("timing and loudness: {text}"))
        }
    }
}

/// Connected components (8-neighbour on the grid) of the region cells.
fn components(cells: &[Cell], idx: &[usize], step: f64) -> Vec<Vec<usize>> {
    use std::collections::{HashMap, HashSet};
    let key = |c: &Cell| {
        (
            (c.east / step).round() as i64,
            (c.north / step).round() as i64,
        )
    };
    let by_key: HashMap<(i64, i64), usize> = idx.iter().map(|&i| (key(&cells[i]), i)).collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &i in idx {
        if !seen.insert(i) {
            continue;
        }
        let mut comp = vec![i];
        let mut stack = vec![i];
        while let Some(j) = stack.pop() {
            let (a, b) = key(&cells[j]);
            for da in -1..=1 {
                for db in -1..=1 {
                    if let Some(&k) = by_key.get(&(a + da, b + db))
                        && seen.insert(k)
                    {
                        comp.push(k);
                        stack.push(k);
                    }
                }
            }
        }
        out.push(comp);
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.len()));
    out
}

/// (length, width, axis° from north, centre east, centre north) of a component.
fn extent(cells: &[Cell], comp: &[usize], step: f64) -> (f64, f64, f64, f64, f64) {
    let n = comp.len() as f64;
    let (me, mn) = comp.iter().fold((0.0, 0.0), |a, &i| {
        (a.0 + cells[i].east / n, a.1 + cells[i].north / n)
    });
    let (mut cee, mut cnn, mut cen) = (0.0, 0.0, 0.0);
    for &i in comp {
        let (de, dn) = (cells[i].east - me, cells[i].north - mn);
        cee += de * de / n;
        cnn += dn * dn / n;
        cen += de * dn / n;
    }
    let tr = cee + cnn;
    let det = cee * cnn - cen * cen;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let (l1, l2) = (tr / 2.0 + disc, (tr / 2.0 - disc).max(0.0));
    let axis = (0.5 * (2.0 * cen).atan2(cnn - cee))
        .to_degrees()
        .rem_euclid(180.0);
    (
        (12.0 * l1).sqrt() + step,
        (12.0 * l2).sqrt() + step,
        axis,
        me,
        mn,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use gm_core::{Epoch, geodetic_to_ecef};
    use gm_doppler::physics::cv::StraightLine;
    use gm_sources::area::{Area, When};

    const OBS: Geodetic = Geodetic {
        lat_deg: 40.8765,
        lon_deg: -73.5250,
        h_m: 100.0 + crate::params::MIC_H,
    };

    fn line(heading: f64, v: f64, alt: f64, offset: f64, cpa_t: f64) -> StraightLine {
        StraightLine {
            frame_origin: OBS,
            heading_deg: heading,
            speed_mps: v,
            alt_m: alt,
            cpa_offset_m: offset,
            cpa_time_s: cpa_t,
            half_span_s: 200.0,
            dt_s: 1.0,
        }
    }

    fn cand(id: &str, l: StraightLine) -> Candidate {
        Candidate {
            icao: id.into(),
            type_code: Some("B738".into()),
            class: crate::grid::AircraftClass::Jet,
            tracks: vec![l.track(id)],
        }
    }

    /// Other aircraft over ±3 h through the area, none passing during the recording window.
    fn traffic_noise() -> Vec<Candidate> {
        (0..60)
            .filter(|&k| !(-900.0..=1500.0).contains(&(-10_800.0 + 360.0 * k as f64)))
            .map(|k| {
                let t = -10_800.0 + 360.0 * k as f64;
                let h = (37.0 * k as f64) % 360.0;
                let off = [-6000.0, -2500.0, 1500.0, 4000.0, 7000.0][k % 5];
                let alt = [900.0, 1500.0, 2500.0][k % 3];
                let v = [90.0, 120.0, 150.0, 200.0][k % 4];
                cand(&format!("tr{k:02}"), line(h, v, alt, off, t))
            })
            .collect()
    }

    /// A pass envelope at the observer from `c` (gain G, slope β), cut to [t0, t1], plus noise.
    fn envelope_of(c: &Candidate, t0: f64, t1: f64, g: f64, beta: f64, id: usize) -> PassEnvelope {
        let obs = geodetic_to_ecef(OBS);
        let m = Medium::still(343.0).unwrap();
        let a = mid_band_alpha();
        let mut t = Vec::new();
        let mut y = Vec::new();
        let mut k = 0u64;
        let mut tb = t0 + 2.5;
        while tb < t1 {
            let l = predicted_level_db(&c.tracks[0], obs, tb, m, a).unwrap_or(-300.0);
            let ex = 10.0 * (1.0 + 10f64.powf((g + beta * l) / 10.0)).log10();
            // Deterministic ±0.5 dB jitter.
            k = k
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((k >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.5;
            t.push(tb);
            y.push(ex + noise);
            tb += 5.0;
        }
        PassEnvelope {
            pass_id: id,
            t,
            level_db: y,
            sigma_db: 0.5,
            cut: t0 <= 0.0,
        }
    }

    fn inputs<'a>(area: &'a Area, atmos: &'a crate::atmos::Atmosphere) -> Inputs<'a> {
        Inputs {
            events: &[],
            aircraft: &[],
            area,
            when: When {
                center: Epoch::from_rfc3339("2026-09-08T20:10:00Z").unwrap(),
                minus_s: 60.0,
                plus_s: 60.0,
            },
            duration_s: 600.0,
            ground: &|_, _| 100.0,
            atmos,
            noise: None,
            decoys: &[],
        }
    }

    fn area() -> Area {
        Area::Circle {
            lat: OBS.lat_deg,
            lon: OBS.lon_deg,
            radius_m: 12_000.0,
        }
    }

    fn dist_m(a: (f64, f64)) -> f64 {
        gm_core::geodesic_distance_m(
            Geodetic {
                lat_deg: a.0,
                lon_deg: a.1,
                h_m: 0.0,
            },
            Geodetic { h_m: 0.0, ..OBS },
        )
    }

    fn near_truth(o: &EnvOutcome, tol_m: f64) -> bool {
        o.region.iter().any(|&c| dist_m(c) <= tol_m)
    }

    #[test]
    fn predicted_level_peaks_at_cpa_arrival() {
        let c = cand("jet", line(45.0, 120.0, 1000.0, 3000.0, 100.0));
        let obs = geodetic_to_ecef(OBS);
        let m = Medium::still(343.0).unwrap();
        let a = mid_band_alpha();
        let lv: Vec<(f64, f64)> = (0..400)
            .map(|i| {
                let t = i as f64 * 0.5;
                (t, predicted_level_db(&c.tracks[0], obs, t, m, a).unwrap())
            })
            .collect();
        let peak = lv.iter().max_by(|x, y| x.1.total_cmp(&y.1)).unwrap().0;
        let slant = (1000.0f64.powi(2) + 3000.0f64.powi(2)).sqrt();
        let arrival = 100.0 + slant / 343.0;
        assert!(
            (peak - arrival).abs() <= 1.0,
            "peak {peak} vs arrival {arrival}"
        );
    }

    #[test]
    fn single_pass_gives_a_band_containing_truth_not_a_pin() {
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let mut traffic = traffic_noise();
        traffic.push(real.clone());
        let env = envelope_of(&real, 0.0, 150.0, 25.0, 1.0, 1);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(&inputs(&ar, &at), &traffic, &[env], &EnvParams::default());
        assert!(
            !matches!(o.verdict, EnvVerdict::Located { .. }),
            "one pass must not pin: {:?}",
            o.verdict
        );
        assert!(
            matches!(o.verdict, EnvVerdict::Band { .. }),
            "{:?}",
            o.verdict
        );
        assert!(near_truth(&o, 1000.0), "truth outside the region");
        assert_eq!(o.per_pass[0].1.as_deref(), Some("a1b2c3"));
    }

    #[test]
    fn two_crossing_passes_pin_a_compact_region_at_truth() {
        let a = cand("aaaaaa", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let b = cand("bbbbbb", line(150.0, 90.0, 900.0, -1800.0, 330.0));
        let mut traffic = traffic_noise();
        traffic.push(a.clone());
        traffic.push(b.clone());
        let ea = envelope_of(&a, 0.0, 160.0, 25.0, 1.0, 1);
        let eb = envelope_of(&b, 250.0, 420.0, 22.0, 1.0, 2);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(
            &inputs(&ar, &at),
            &traffic,
            &[ea, eb],
            &EnvParams::default(),
        );
        assert!(o.p <= 0.01, "p {}", o.p);
        assert!(near_truth(&o, 1000.0), "{:?}", o.verdict);
        let best = o.best.unwrap();
        assert!(dist_m(best) <= 1000.0, "best cell {} m off", dist_m(best));
    }

    #[test]
    fn three_crossing_passes_locate_within_1km() {
        let a = cand("aaaaaa", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let b = cand("bbbbbb", line(150.0, 90.0, 900.0, -1800.0, 300.0));
        let c = cand("cccccc", line(270.0, 110.0, 1500.0, 2200.0, 520.0));
        let mut traffic = traffic_noise();
        traffic.extend([a.clone(), b.clone(), c.clone()]);
        let ea = envelope_of(&a, 0.0, 160.0, 25.0, 1.0, 1);
        let eb = envelope_of(&b, 230.0, 390.0, 22.0, 1.0, 2);
        let ec = envelope_of(&c, 450.0, 600.0, 26.0, 1.0, 3);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(
            &inputs(&ar, &at),
            &traffic,
            &[ea, eb, ec],
            &EnvParams::default(),
        );
        match o.verdict {
            EnvVerdict::Located { lat, lon, .. } => {
                assert!(dist_m((lat, lon)) <= 1000.0, "{} m off", dist_m((lat, lon)))
            }
            v => panic!("not located: {v:?} (p {}, p_drop {:?})", o.p, o.p_drop),
        }
    }

    #[test]
    fn wrong_time_traffic_is_inconclusive() {
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let env = envelope_of(&real, 0.0, 150.0, 25.0, 1.0, 1);
        // The real aircraft is missing from the traffic: only other times' aircraft.
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(
            &inputs(&ar, &at),
            &traffic_noise(),
            &[env],
            &EnvParams::default(),
        );
        assert!(
            !matches!(o.verdict, EnvVerdict::Located { .. }),
            "{:?}",
            o.verdict
        );
    }

    #[test]
    fn no_fitting_aircraft_is_inconclusive_with_reason() {
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let env = envelope_of(&real, 0.0, 150.0, 25.0, 1.0, 1);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(&inputs(&ar, &at), &[], &[env], &EnvParams::default());
        match o.verdict {
            EnvVerdict::Inconclusive { reason } => {
                assert!(reason.contains("no ADS-B aircraft"), "{reason}")
            }
            v => panic!("{v:?}"),
        }
    }

    #[test]
    fn each_pass_has_at_most_one_aircraft() {
        // A circuit: the same aircraft passes twice; both passes may name it, each names one.
        let first = line(45.0, 70.0, 600.0, 1500.0, 60.0);
        let second = line(225.0, 70.0, 600.0, -1500.0, 360.0);
        let mut t1 = first.track("cccccc");
        let t2 = second.track("cccccc");
        let last = t1.knots.last().unwrap().t;
        t1.knots.extend(t2.knots.into_iter().filter(|k| k.t > last));
        let circuit = Candidate {
            icao: "cccccc".into(),
            type_code: Some("C172".into()),
            class: crate::grid::AircraftClass::Prop,
            tracks: vec![t1],
        };
        let mut traffic = traffic_noise();
        traffic.push(circuit.clone());
        let e1 = envelope_of(&circuit, 0.0, 150.0, 20.0, 1.0, 1);
        let e2 = envelope_of(&circuit, 300.0, 450.0, 20.0, 1.0, 2);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(
            &inputs(&ar, &at),
            &traffic,
            &[e1, e2],
            &EnvParams::default(),
        );
        assert_eq!(o.per_pass.len(), 2);
        assert!(
            o.per_pass.iter().all(|p| p.1.as_deref() == Some("cccccc")),
            "{:?}",
            o.per_pass
        );
    }

    #[test]
    fn truncated_pass_scores_on_its_recorded_part() {
        // The recording opens near the loudest point (CPA arrival ≈ 1 s).
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, -7.5));
        let mut traffic = traffic_noise();
        traffic.push(real.clone());
        let env = envelope_of(&real, 0.0, 90.0, 25.0, 1.0, 1);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(&inputs(&ar, &at), &traffic, &[env], &EnvParams::default());
        assert!(!matches!(o.verdict, EnvVerdict::Located { .. }));
        assert!(
            near_truth(&o, 1000.0),
            "truth outside the region: {:?}",
            o.verdict
        );
    }

    #[test]
    fn tails_of_a_hump_are_fitted_as_excess_not_level() {
        // Peak excess ≈ 10 dB: the tails saturate toward 0.
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let mut traffic = traffic_noise();
        traffic.push(real.clone());
        let env = envelope_of(&real, 0.0, 150.0, 17.0, 1.0, 1);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(&inputs(&ar, &at), &traffic, &[env], &EnvParams::default());
        assert!(
            near_truth(&o, 1000.0),
            "truth outside the region: {:?}",
            o.verdict
        );
    }

    #[test]
    fn compressed_envelope_gives_the_same_region() {
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let mut traffic = traffic_noise();
        traffic.push(real.clone());
        let env = envelope_of(&real, 0.0, 150.0, 25.0, 0.5, 1);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(&inputs(&ar, &at), &traffic, &[env], &EnvParams::default());
        assert!(!matches!(o.verdict, EnvVerdict::Located { .. }));
        assert!(
            near_truth(&o, 1000.0),
            "truth outside the region: {:?}",
            o.verdict
        );
    }

    fn outcome(v: crate::search::Verdict, best: Option<(f64, f64)>) -> crate::search::Outcome {
        let o = OBS;
        crate::search::Outcome {
            verdict: v,
            best: best.map(|(lat, lon)| crate::search::Ranked {
                geo: Geodetic {
                    lat_deg: lat,
                    lon_deg: lon,
                    h_m: 0.0,
                },
                east: 0.0,
                north: 0.0,
                ll: 0.0,
                prob: 1.0,
            }),
            top: vec![],
            mirror: None,
            delta_s: 0.0,
            matched: vec![],
            silent: vec![],
            region90_km2: 1.0,
            on_boundary: false,
            heat: vec![],
            frame_origin: o,
            wind_enu: None,
            atmosphere: String::new(),
        }
    }

    fn env(v: EnvVerdict, best: (f64, f64), region: Vec<(f64, f64)>) -> EnvOutcome {
        EnvOutcome {
            verdict: v,
            p: 0.001,
            p_drop: Some(0.01),
            delta_s: 0.0,
            per_pass: vec![],
            region,
            best: Some(best),
            notes: vec![],
        }
    }

    #[test]
    fn tonal_located_answer_stands_over_envelope() {
        use crate::search::Verdict;
        let t = outcome(Verdict::Located, Some((40.87, -73.52)));
        let band = EnvVerdict::Band {
            lat: 40.92,
            lon: -73.44,
            length_m: 8000.0,
            width_m: 1000.0,
            axis_deg: 45.0,
            mirror: true,
        };
        let e = env(band, (40.92, -73.44), vec![(40.92, -73.44)]);
        let (o, note) = combine_with_envelope(t, &e, false);
        assert_eq!(o.verdict, Verdict::Located);
        assert!((o.best.unwrap().geo.lat_deg - 40.87).abs() < 1e-9);
        assert!(note.contains("excludes"), "{note}");
    }

    #[test]
    fn methods_disagreeing_is_not_located() {
        use crate::search::Verdict;
        let t = outcome(Verdict::Located, Some((40.87, -73.52)));
        let e = env(
            EnvVerdict::Located {
                lat: 40.92,
                lon: -73.52,
                radius_m: 500.0,
            },
            (40.92, -73.52),
            vec![(40.92, -73.52)],
        );
        let (o, note) = combine_with_envelope(t, &e, false);
        assert!(
            matches!(o.verdict, Verdict::Inconclusive(ref r) if r.contains("methods disagree")),
            "{:?} {note}",
            o.verdict
        );
    }

    #[test]
    fn envelope_located_becomes_the_answer() {
        use crate::search::Verdict;
        let t = outcome(Verdict::Inconclusive("no events".into()), None);
        let e = env(
            EnvVerdict::Located {
                lat: 40.89,
                lon: -73.53,
                radius_m: 600.0,
            },
            (40.89, -73.53),
            vec![(40.89, -73.53), (40.891, -73.53)],
        );
        let (o, _) = combine_with_envelope(t, &e, false);
        assert_eq!(o.verdict, Verdict::Located);
        let b = o.best.unwrap();
        assert!((b.geo.lat_deg - 40.89).abs() < 1e-9 && (b.geo.lon_deg + 73.53).abs() < 1e-9);
        assert!(
            o.region90_km2 > 0.0 && o.region90_km2 <= 2.0,
            "{}",
            o.region90_km2
        );
    }

    #[test]
    fn envelope_band_is_reported_as_inconclusive_band() {
        use crate::search::Verdict;
        let t = outcome(Verdict::Inconclusive("no events".into()), None);
        let band = EnvVerdict::Band {
            lat: 40.92,
            lon: -73.44,
            length_m: 8000.0,
            width_m: 1000.0,
            axis_deg: 45.0,
            mirror: true,
        };
        let e = env(
            band,
            (40.92, -73.44),
            vec![(40.92, -73.44), (40.93, -73.43)],
        );
        let (o, _) = combine_with_envelope(t, &e, false);
        assert!(
            matches!(o.verdict, Verdict::Inconclusive(ref r) if r.starts_with("band")),
            "{:?}",
            o.verdict
        );
        assert_eq!(o.heat.len(), 2);
    }

    #[test]
    fn an_unshiftable_aircraft_cannot_earn_significance() {
        // Only the real aircraft: the null can't move it, so it can't be scored either.
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let env = envelope_of(&real, 0.0, 150.0, 25.0, 1.0, 1);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(&inputs(&ar, &at), &[real], &[env], &EnvParams::default());
        assert!(
            matches!(o.verdict, EnvVerdict::Inconclusive { .. }),
            "{:?} p {}",
            o.verdict,
            o.p
        );
    }

    #[test]
    fn sparse_traffic_null_is_inconclusive() {
        let real = cand("a1b2c3", line(45.0, 120.0, 1000.0, 3000.0, 60.0));
        let traffic = vec![
            real.clone(),
            cand("far001", line(90.0, 120.0, 1500.0, 3000.0, -7000.0)),
            cand("far002", line(200.0, 150.0, 2500.0, -4000.0, 7000.0)),
        ];
        let env = envelope_of(&real, 0.0, 150.0, 25.0, 1.0, 1);
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let o = envelope_search(&inputs(&ar, &at), &traffic, &[env], &EnvParams::default());
        match &o.verdict {
            EnvVerdict::Inconclusive { reason } => {
                assert!(reason.contains("too sparse"), "{reason}")
            }
            v => panic!("{v:?} p {}", o.p),
        }
    }

    #[test]
    fn sigma_counts_independent_frames() {
        // 1-s frames at a 0.5-s hop: each value repeated twice, so a 5-s block mean holds 5
        // independent samples, not 10.
        let mut k = 7u64;
        let mut gauss = || {
            let mut u = || {
                k = k
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((k >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        let raw: Vec<f32> = (0..400)
            .flat_map(|_| {
                let v = (3.0 * gauss()) as f32;
                [v, v]
            })
            .collect();
        let n = raw.len();
        let det = Detection {
            passes: vec![gm_doppler::audio::Pass {
                id: 1,
                t_start: 100.0,
                t_end: 300.0,
                t_peak: 200.0,
                excess_db: 6.0,
                score: 5.0,
                flags: vec![],
                hf_db: None,
            }],
            envelope: gm_doppler::audio::Envelope {
                t0: 0.5,
                dt: 0.5,
                frame_s: 1.0,
                mid_db: raw.clone(),
                raw_mid_db: raw,
                bands_db: vec![[0.0; 6]; n],
            },
            ..Default::default()
        };
        let e = &pass_envelopes(&det, &[1])[0];
        let want = 3.0 / 5f64.sqrt();
        assert!(
            (e.sigma_db / want - 1.0).abs() < 0.25,
            "sigma {} vs {want}",
            e.sigma_db
        );
    }

    #[test]
    fn par_propagates_a_worker_panic() {
        let r = std::panic::catch_unwind(|| par(8, 4, |i| if i == 5 { panic!("boom") } else { i }));
        assert!(r.is_err());
    }

    #[test]
    fn a_clock_at_the_window_edge_is_noted() {
        // The recording started 70 s after the prior's centre and the clock window is only ±10 s:
        // two crossing passes pin the clock, so the best δ sits on the window's edge.
        let a = cand("aaaaaa", line(45.0, 120.0, 1000.0, 3000.0, 130.0));
        let b = cand("bbbbbb", line(150.0, 90.0, 900.0, -1800.0, 400.0));
        let mut traffic = traffic_noise();
        traffic.extend([a.clone(), b.clone()]);
        let mut ea = envelope_of(&a, 70.0, 230.0, 25.0, 1.0, 1);
        let mut eb = envelope_of(&b, 320.0, 490.0, 22.0, 1.0, 2);
        for t in ea.t.iter_mut().chain(eb.t.iter_mut()) {
            *t -= 70.0;
        }
        ea.cut = false;
        let (ar, at) = (area(), crate::atmos::Atmosphere::still());
        let mut inp = inputs(&ar, &at);
        inp.when.minus_s = 10.0;
        inp.when.plus_s = 10.0;
        let o = envelope_search(&inp, &traffic, &[ea, eb], &EnvParams::default());
        assert!(
            o.notes
                .iter()
                .any(|n| n.contains("edge of the clock window")),
            "{:?} δ {}",
            o.notes,
            o.delta_s
        );
    }
}
