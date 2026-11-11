//! Tonal ridges: spectral peaks linked frame to frame, bridging short gaps, keeping only
//! falling, long-enough tracks (a passing source always lowers its received pitch).

use super::spectrogram::Spectrogram;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RidgePoint {
    pub t: f64,
    /// Parabolic-interpolated frequency.
    pub f: f64,
    pub snr_db: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ridge {
    pub points: Vec<RidgePoint>,
}

impl Ridge {
    pub fn t_start(&self) -> f64 {
        self.points.first().map_or(f64::NAN, |p| p.t)
    }
    pub fn t_end(&self) -> f64 {
        self.points.last().map_or(f64::NAN, |p| p.t)
    }
    pub fn mean_snr_db(&self) -> f32 {
        self.points.iter().map(|p| p.snr_db).sum::<f32>() / self.points.len().max(1) as f32
    }
}

#[derive(Clone, Debug)]
pub struct RidgeParams {
    pub min_snr_db: f32,
    /// Largest |df/dt| as a fraction of f per second.
    pub max_rel_slope_per_s: f64,
    pub max_gap_s: f64,
    pub min_duration_s: f64,
    /// (f_max − f_min) / f_mean must exceed this (rejects stationary lines and hum).
    pub min_drop_frac: f64,
    pub max_peaks_per_frame: usize,
    /// A ridge's mean whitened level must reach this.
    pub min_mean_snr_db: f32,
    /// Allowed miss from the predicted frequency, beyond 1.5 bins, as a fraction of f per second of gap.
    pub miss_rel_per_s: f64,
}

impl Default for RidgeParams {
    fn default() -> Self {
        Self {
            min_snr_db: 9.0,
            max_rel_slope_per_s: 0.06,
            max_gap_s: 2.0,
            min_duration_s: 8.0,
            min_drop_frac: 0.02,
            max_peaks_per_frame: 12,
            min_mean_snr_db: 12.0,
            miss_rel_per_s: 0.01,
        }
    }
}

/// Frame-to-frame linking into raw pieces.
pub(crate) fn link(s: &Spectrogram, p: &RidgeParams) -> Vec<Ridge> {
    let df = s.bin_hz();
    let mut active: Vec<Ridge> = Vec::new();
    let mut done: Vec<Ridge> = Vec::new();
    for (fi, &t) in s.times.iter().enumerate() {
        // Retire ridges whose gap has grown too long.
        let (keep, old): (Vec<Ridge>, Vec<Ridge>) = active
            .into_iter()
            .partition(|r| t - r.t_end() <= p.max_gap_s);
        active = keep;
        done.extend(old);
        let mut taken = vec![false; active.len()];
        for pk in peaks(s, fi, p) {
            let mut best: Option<(usize, f64)> = None;
            for (ri, r) in active.iter().enumerate() {
                if taken[ri] {
                    continue;
                }
                let last = r.points[r.points.len() - 1];
                let dt = t - last.t;
                let slope = match r.points.len() {
                    n if n >= 2 => {
                        let prev = r.points[n - 2];
                        let sl = (last.f - prev.f) / (last.t - prev.t);
                        sl.clamp(
                            -p.max_rel_slope_per_s * last.f,
                            p.max_rel_slope_per_s * last.f,
                        )
                    }
                    _ => 0.0,
                };
                let predicted = last.f + slope * dt;
                let tol = 1.5 * df + p.miss_rel_per_s * last.f * dt;
                let miss = (pk.f - predicted).abs();
                // Prefer close *and* recently updated tracks, so stale strays don't steal peaks.
                let cost = miss / tol + dt / p.max_gap_s;
                if miss <= tol && best.is_none_or(|(_, c)| cost < c) {
                    best = Some((ri, cost));
                }
            }
            match best {
                Some((ri, _)) => {
                    taken[ri] = true;
                    active[ri].points.push(pk);
                }
                None => {
                    active.push(Ridge { points: vec![pk] });
                    taken.push(true);
                }
            }
        }
    }
    done.extend(active);
    done
}

pub fn ridges(s: &Spectrogram, p: &RidgeParams) -> Vec<Ridge> {
    let df = s.bin_hz();
    let pieces: Vec<Ridge> = link(s, p)
        .into_iter()
        .filter(|r| r.points.len() >= 3)
        .collect();
    let merged = merge_pieces(pieces, p, df);
    let mut out: Vec<Ridge> = merged.into_iter().filter(|r| keep_ridge(r, p)).collect();
    out.sort_by(|a, b| a.t_start().total_cmp(&b.t_start()));
    out
}

/// A fragment needs this long to carry a slope worth fitting.
pub const FRAGMENT_MIN_S: f64 = 3.0;
/// And this many points: the fit's minimum.
pub const FRAGMENT_MIN_POINTS: usize = 8;

/// The pieces a pass breaks into when it fades: linked and merged like ridges, kept if long
/// and loud enough to measure, whatever their direction or length beyond that. A ridge with
/// at least [`FRAGMENT_MIN_POINTS`] points is a fragment that is also long and falling; the
/// others are the arms of a pass between fades and, when they rise, the material of pass
/// pursuit's mirrored null.
pub fn fragments(s: &Spectrogram, p: &RidgeParams) -> Vec<Ridge> {
    let df = s.bin_hz();
    let pieces: Vec<Ridge> = link(s, p)
        .into_iter()
        .filter(|r| r.points.len() >= 3)
        .collect();
    let mut out: Vec<Ridge> = merge_pieces(pieces, p, df)
        .into_iter()
        .filter(|r| {
            r.points.len() >= FRAGMENT_MIN_POINTS
                && r.t_end() - r.t_start() >= FRAGMENT_MIN_S
                && r.mean_snr_db() >= p.min_mean_snr_db
        })
        .collect();
    out.sort_by(|a, b| a.t_start().total_cmp(&b.t_start()));
    out
}

/// Frequency of a ridge at time `t` (nearest point).
fn f_near(r: &Ridge, t: f64) -> f64 {
    r.points
        .iter()
        .min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
        .map_or(f64::NAN, |q| q.f)
}

/// Join fragments of one track: B starts within [A.end − 3 s, A.end + max_gap] and agrees in
/// frequency where they meet. Duplicate frames keep the stronger point.
fn merge_pieces(mut pieces: Vec<Ridge>, p: &RidgeParams, df: f64) -> Vec<Ridge> {
    const OVERLAP_S: f64 = 3.0;
    pieces.sort_by(|a, b| a.t_start().total_cmp(&b.t_start()));
    // A merge keeps the earlier piece's start, so start times never change: each piece's join
    // candidates are the pieces starting within [t_end − OVERLAP_S, t_end + max_gap_s], found by
    // binary search instead of scanning every pair after every join. Passes repeat until stable,
    // so a piece that only becomes joinable once its partner has grown is still merged.
    let starts: Vec<f64> = pieces.iter().map(Ridge::t_start).collect();
    let mut alive = vec![true; pieces.len()];
    loop {
        let mut changed = false;
        for i in 0..pieces.len() {
            if !alive[i] {
                continue;
            }
            loop {
                let (a_start, a_end) = (pieces[i].t_start(), pieces[i].t_end());
                let lo_t = a_start.max(a_end - OVERLAP_S);
                let lo = starts.partition_point(|&s| s < lo_t);
                let hi = starts.partition_point(|&s| s <= a_end + p.max_gap_s);
                let found = (lo..hi).find(|&j| {
                    if j == i || !alive[j] {
                        return false;
                    }
                    let (a, b) = (&pieces[i], &pieces[j]);
                    let start = b.t_start();
                    if b.t_end() <= a.t_end() {
                        return false;
                    }
                    let tol =
                        4.0 * df + p.miss_rel_per_s * b.points[0].f * (start - a.t_end()).abs();
                    (f_near(a, start) - b.points[0].f).abs() <= tol
                });
                let Some(j) = found else { break };
                alive[j] = false;
                changed = true;
                let b_points = std::mem::take(&mut pieces[j].points);
                let a = &mut pieces[i];
                for q in b_points {
                    match a.points.iter_mut().find(|x| (x.t - q.t).abs() < 1e-9) {
                        Some(x) if x.snr_db >= q.snr_db => {}
                        Some(x) => *x = q,
                        None => a.points.push(q),
                    }
                }
                a.points.sort_by(|x, y| x.t.total_cmp(&y.t));
            }
        }
        if !changed {
            break;
        }
    }
    pieces
        .into_iter()
        .zip(alive)
        .filter_map(|(r, keep)| keep.then_some(r))
        .collect()
}

/// Local maxima of the whitened frame, strongest first, interpolated in frequency.
fn peaks(s: &Spectrogram, fi: usize, p: &RidgeParams) -> Vec<RidgePoint> {
    let w = &s.whitened[fi];
    let pw = &s.power_db[fi];
    let df = s.bin_hz();
    let mut out: Vec<RidgePoint> = (1..w.len().saturating_sub(1))
        .filter(|&b| !s.mask.get(b).copied().unwrap_or(false))
        .filter(|&b| w[b] >= p.min_snr_db && w[b] > w[b - 1] && w[b] >= w[b + 1])
        .map(|b| {
            let (a, c, d) = (f64::from(pw[b - 1]), f64::from(pw[b]), f64::from(pw[b + 1]));
            let denom = a - 2.0 * c + d;
            let delta = if denom.abs() > 1e-12 {
                (0.5 * (a - d) / denom).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            RidgePoint {
                t: s.times[fi],
                f: s.freqs[b] + delta * df,
                snr_db: w[b],
            }
        })
        .collect();
    out.sort_by(|a, b| b.snr_db.total_cmp(&a.snr_db));
    out.truncate(p.max_peaks_per_frame);
    out
}

fn keep_ridge(r: &Ridge, p: &RidgeParams) -> bool {
    ridge_verdict(r, p).is_ok()
}

/// Why a merged ridge piece is kept (`Ok`) or dropped — the rejection log behind `keep_ridge`.
pub fn ridge_verdict(r: &Ridge, p: &RidgeParams) -> Result<(), &'static str> {
    if r.points.len() < 3 {
        return Err("fewer than 3 points");
    }
    if r.t_end() - r.t_start() < p.min_duration_s {
        return Err("shorter than min_duration_s");
    }
    if r.mean_snr_db() < p.min_mean_snr_db {
        return Err("mean SNR below min_mean_snr_db");
    }
    let (lo, hi, sum) = r
        .points
        .iter()
        .fold((f64::MAX, f64::MIN, 0.0), |(lo, hi, s), q| {
            (lo.min(q.f), hi.max(q.f), s + q.f)
        });
    let mean = sum / r.points.len() as f64;
    // Compare the first and last fifths so a noisy end point doesn't decide the direction.
    let k = (r.points.len() / 5).max(1);
    let head = r.points[..k].iter().map(|q| q.f).sum::<f64>() / k as f64;
    let tail = r.points[r.points.len() - k..]
        .iter()
        .map(|q| q.f)
        .sum::<f64>()
        / k as f64;
    if (hi - lo) / mean < p.min_drop_frac {
        return Err("frequency drop below min_drop_frac");
    }
    if head <= tail {
        return Err("not falling (head <= tail)");
    }
    Ok(())
}

/// Every merged piece (≥ 3 points) with its verdict: the rejection log of the ridge stage.
pub fn ridges_explained(
    s: &Spectrogram,
    p: &RidgeParams,
) -> Vec<(Ridge, Result<(), &'static str>)> {
    let df = s.bin_hz();
    let pieces: Vec<Ridge> = link(s, p)
        .into_iter()
        .filter(|r| r.points.len() >= 3)
        .collect();
    let mut v: Vec<(Ridge, Result<(), &'static str>)> = merge_pieces(pieces, p, df)
        .into_iter()
        .map(|r| {
            let verdict = ridge_verdict(&r, p);
            (r, verdict)
        })
        .collect();
    v.sort_by(|a, b| a.0.t_start().total_cmp(&b.0.t_start()));
    v
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::audio::{ANALYSIS_RATE, Audio, SpecParams, decimate, spectrogram};
    use std::path::PathBuf;

    pub(crate) fn sim(name: &str, edit: impl Fn(String) -> String) -> (Audio, gm_sim::Truth) {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sim")
            .join(name);
        let s = gm_sim::parse(&edit(std::fs::read_to_string(p).unwrap())).unwrap();
        let r = gm_sim::render(&s).unwrap();
        (
            Audio {
                sample_rate: s.sample_rate,
                samples: r.samples,
                warnings: Vec::new(),
            },
            r.truth,
        )
    }

    fn summary(r: &[Ridge]) -> Vec<(f64, f64, f64, f64, f32)> {
        r.iter()
            .map(|x| {
                (
                    x.t_start(),
                    x.t_end(),
                    x.points[0].f,
                    x.points[x.points.len() - 1].f,
                    x.mean_snr_db(),
                )
            })
            .collect()
    }

    fn run(a: &Audio) -> Vec<Ridge> {
        let d = decimate(a, ANALYSIS_RATE);
        ridges(
            &spectrogram(&d, &SpecParams::default()),
            &RidgeParams::default(),
        )
    }

    /// Fraction of 1 s slots in [t0, t1] containing at least one ridge point.
    fn coverage(r: &[Ridge], t0: f64, t1: f64) -> f64 {
        let slots: Vec<f64> = (0..((t1 - t0) as usize))
            .map(|i| t0 + i as f64 + 0.5)
            .collect();
        let hit = slots
            .iter()
            .filter(|&&c| {
                r.iter()
                    .any(|x| x.points.iter().any(|q| (q.t - c).abs() <= 0.5))
            })
            .count();
        hit as f64 / slots.len() as f64
    }

    /// The original (cubic) merge, kept as the reference the fast one must agree with.
    fn merge_pieces_reference(mut pieces: Vec<Ridge>, p: &RidgeParams, df: f64) -> Vec<Ridge> {
        const OVERLAP_S: f64 = 3.0;
        pieces.sort_by(|a, b| a.t_start().total_cmp(&b.t_start()));
        loop {
            let mut joined = None;
            'outer: for i in 0..pieces.len() {
                for j in 0..pieces.len() {
                    if i == j {
                        continue;
                    }
                    let (a, b) = (&pieces[i], &pieces[j]);
                    let start = b.t_start();
                    if start < a.t_start()
                        || start < a.t_end() - OVERLAP_S
                        || start > a.t_end() + p.max_gap_s
                        || b.t_end() <= a.t_end()
                    {
                        continue;
                    }
                    let tol =
                        4.0 * df + p.miss_rel_per_s * b.points[0].f * (start - a.t_end()).abs();
                    if (f_near(a, start) - b.points[0].f).abs() <= tol {
                        joined = Some((i, j));
                        break 'outer;
                    }
                }
            }
            let Some((i, j)) = joined else { return pieces };
            let b = pieces.remove(j);
            let i = if j < i { i - 1 } else { i };
            let a = &mut pieces[i];
            for q in b.points {
                match a.points.iter_mut().find(|x| (x.t - q.t).abs() < 1e-9) {
                    Some(x) if x.snr_db >= q.snr_db => {}
                    Some(x) => *x = q,
                    None => a.points.push(q),
                }
            }
            a.points.sort_by(|x, y| x.t.total_cmp(&y.t));
        }
    }

    /// Deterministic synthetic pieces: short falling fragments scattered over `secs`.
    fn synthetic_pieces(n: usize, secs: f64, seed: u64) -> Vec<Ridge> {
        let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut u = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let t0 = u() * secs;
                let f0 = 60.0 + u() * 400.0;
                let len = 3 + (u() * 12.0) as usize;
                Ridge {
                    points: (0..len)
                        .map(|k| RidgePoint {
                            t: t0 + 0.25 * k as f64,
                            f: f0 - 0.05 * k as f64,
                            snr_db: 10.0 + (u() * 5.0) as f32,
                        })
                        .collect(),
                }
            })
            .collect()
    }

    /// A tone gliding from `f_a` to `f_b` Hz over `secs`, in light noise.
    fn glide(f_a: f64, f_b: f64, secs: f64) -> Audio {
        let rate = 8000u32;
        let mut st = 0x2545_F491_4F6C_DD1Du64;
        let mut phase = 0.0f64;
        let n = (secs * f64::from(rate)) as usize;
        let samples = (0..n)
            .map(|i| {
                let f = f_a + (f_b - f_a) * i as f64 / n as f64;
                phase += std::f64::consts::TAU * f / f64::from(rate);
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                let noise = (st >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                (0.3 * phase.sin() + 0.05 * noise) as f32
            })
            .collect();
        Audio {
            sample_rate: rate,
            samples,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn fragments_keep_rising_and_short_pieces() {
        let p = RidgeParams::default();
        let spec = |a: &Audio| spectrogram(&decimate(a, ANALYSIS_RATE), &SpecParams::default());
        // A rising tone is no ridge (nothing that passes by rises), but it is a fragment: the
        // mirrored null of pass pursuit is built from such pieces.
        let rising = spec(&glide(200.0, 212.0, 20.0));
        assert!(ridges(&rising, &p).is_empty());
        // (A tone this clean also shows its window sidelobes as parallel fragments.)
        let f = fragments(&rising, &p);
        assert!(!f.is_empty(), "{:?}", summary(&f));
        for x in &f {
            assert!(x.points[0].f < x.points[x.points.len() - 1].f);
        }
        // So is a falling piece too short to be a ridge: an arm of a pass between two fades.
        let short = spec(&glide(212.0, 208.0, 5.0));
        assert!(ridges(&short, &p).is_empty());
        assert!(!fragments(&short, &p).is_empty());
        // Every ridge is a fragment.
        let (a, _) = sim("one_pass.toml", |s| s);
        let s = spec(&a);
        let (rs, fs) = (ridges(&s, &p), fragments(&s, &p));
        assert!(!rs.is_empty());
        for r in &rs {
            assert!(fs.contains(r), "{:?}", summary(std::slice::from_ref(r)));
        }
        for f in &fs {
            assert!(
                f.t_end() - f.t_start() >= FRAGMENT_MIN_S
                    && f.points.len() >= FRAGMENT_MIN_POINTS
                    && f.mean_snr_db() >= p.min_mean_snr_db,
                "{:?}",
                summary(std::slice::from_ref(f))
            );
        }
    }

    #[test]
    fn merge_pieces_matches_reference() {
        let p = RidgeParams::default();
        for seed in 1..=5 {
            let pieces = synthetic_pieces(400, 300.0, seed);
            let fast = merge_pieces(pieces.clone(), &p, 0.244);
            let slow = merge_pieces_reference(pieces, &p, 0.244);
            let key = |v: &[Ridge]| {
                let mut k: Vec<(i64, usize)> = v
                    .iter()
                    .map(|r| ((r.t_start() * 1e6) as i64, r.points.len()))
                    .collect();
                k.sort();
                k
            };
            assert_eq!(key(&fast), key(&slow), "seed {seed}");
        }
    }

    #[test]
    fn merge_pieces_scales() {
        // Robust whitening leaves thousands of short noise pieces in a long recording; merging
        // them must stay near-linear (the original rescanned every pair after each join).
        let pieces = synthetic_pieces(6000, 1000.0, 7);
        let t = std::time::Instant::now();
        let _ = merge_pieces(pieces, &RidgeParams::default(), 0.244);
        assert!(t.elapsed().as_secs_f64() < 1.0, "{:?}", t.elapsed());
    }

    #[test]
    fn burst_does_not_ridge() {
        use crate::audio::spectrogram::{SpecParams, spectrogram, tests::noisy};
        let a = noisy(4000, 60.0, None, &[(10.0, 0.5), (25.0, 0.5), (40.0, 0.5)]);
        let s = spectrogram(&a, &SpecParams::default());
        let r = ridges(&s, &RidgeParams::default());
        assert!(r.is_empty(), "{} ridges", r.len());
    }

    #[test]
    fn one_pass_ridges_cover_the_pass() {
        let (a, truth) = sim("one_pass.toml", |s| s);
        let r = run(&a);
        let tc = truth.events[0].cpa_arrival_audio_s;
        assert!(!r.is_empty() && r.len() <= 3, "{:?}", summary(&r));
        assert!(
            coverage(&r, tc - 20.0, tc + 20.0) >= 0.95,
            "{:?}",
            summary(&r)
        );
        assert!(
            r.iter()
                .all(|x| x.points.first().unwrap().f > x.points.last().unwrap().f)
        );
    }

    #[test]
    fn hum_not_a_ridge() {
        let (mut a, truth) = sim("one_pass.toml", |s| s);
        for (i, v) in a.samples.iter_mut().enumerate() {
            let t = i as f64 / f64::from(a.sample_rate);
            *v += (0.3 * (std::f64::consts::TAU * 120.0 * t).sin()
                + 0.1 * (std::f64::consts::TAU * 240.0 * t).sin()) as f32;
        }
        let r = run(&a);
        let tc = truth.events[0].cpa_arrival_audio_s;
        assert!(
            coverage(&r, tc - 20.0, tc + 20.0) >= 0.9,
            "{:?}",
            summary(&r)
        );
        for x in &r {
            for band in [118.0..122.0, 238.0..242.0] {
                let inside = x.points.iter().filter(|p| band.contains(&p.f)).count();
                assert!(
                    inside < x.points.len() / 4,
                    "hum ridge {:?}",
                    summary(std::slice::from_ref(x))
                );
            }
        }
    }

    #[test]
    fn gap_bridged() {
        let (mut a, truth) = sim("one_pass.toml", |s| s);
        let tc = truth.events[0].cpa_arrival_audio_s;
        let rate = f64::from(a.sample_rate);
        let (i0, i1) = (((tc - 0.75) * rate) as usize, ((tc + 0.75) * rate) as usize);
        for v in &mut a.samples[i0..i1] {
            *v = 0.0;
        }
        let r = run(&a);
        assert!(!r.is_empty() && r.len() <= 3, "{:?}", summary(&r));
        assert!(
            coverage(&r, tc - 20.0, tc - 2.0) >= 0.95 && coverage(&r, tc + 2.0, tc + 20.0) >= 0.95,
            "{:?}",
            summary(&r)
        );
    }
}
