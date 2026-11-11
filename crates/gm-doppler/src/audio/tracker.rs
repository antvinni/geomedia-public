//! A model-free path tracker: colsto's "trace a very faint curve" step. On the log-frequency
//! guide image it finds the falling paths of greatest total clipped level, without assuming the
//! pass's exact shape, so partial and one-sided passes that no full template fits still show.
//!
//! It is a local-alignment Viterbi: each cell's score is its clipped level minus a bias (so a
//! path through noise drifts down), a path may start and end anywhere, and from one frame to the
//! next it may hold its bin or fall up to `max_fall` bins (the steep sweep at CPA), never rise.
//! Viterbi always returns paths, even in pure noise: the tracker only proposes. Its candidates
//! are fitted (`fit_cv`, `plausible`, ≥ 32 measured points) and must clear the template stage's
//! bar (the recording's rising-template null) with their fitted curve's f₀-shift z.
//!
//! That bar is not the tracker's own null: a path is chosen *because* its cells were high, so its
//! curve sum is biased up against the f₀ shifts. A null of the tracker's own (Phase 1's
//! calibrated acceptance) was planned but rejected on real audio, so the false rate is measured
//! instead: pure white noise with no pass, 16 × 600 s, gave 0 template/tracker events
//! (`discover-bench --class piston --no-pass`, 2026-11-05).

use super::guide::GuideImage;

#[derive(Clone, Debug)]
pub struct TrackParams {
    /// Per-cell clip (σ), as the template sums use.
    pub clip: (f32, f32),
    /// Subtracted from every cell, per bin: the image's own mean best step gain at that bin
    /// (the best reachable cell, after fall costs, averaged over all frames: noise-dominated,
    /// as passes are sparse) plus this margin, so a path through noise loses score at every
    /// frequency, whatever the bin correlation there.
    pub bias_margin: f32,
    /// Largest fall per frame, in log bins (at most 127: stored as i8).
    pub max_fall: usize,
    /// Cost per bin of fall beyond the first (penalises jumps between unrelated cells). With the
    /// best net gain per cell ≈ 1 (clip 3 minus the bias), sustained falls of ≥ 3 bins/frame
    /// (≈ 4 %/s at 0.25 s frames) run at a loss: a steep CPA section (100 m/s under ~700 m)
    /// is only bridged on the score its flanks have built, so the tracker favours far and
    /// partial passes; near passes are the ridge and template stages'.
    pub fall_cost: f32,
    /// A path is proposed when its score reaches this.
    pub min_score: f32,
    /// ...and it lasts at least this long.
    pub min_duration_s: f64,
    /// At most this many paths.
    pub max_paths: usize,
}

impl Default for TrackParams {
    fn default() -> Self {
        Self {
            clip: (-1.0, 3.0),
            bias_margin: 0.5,
            max_fall: 10,
            fall_cost: 0.6,
            min_score: 20.0,
            min_duration_s: 8.0,
            max_paths: 200,
        }
    }
}

/// A tracked path: (frame, log bin) per frame, in time order, and its score.
#[derive(Clone, Debug)]
pub struct Path {
    pub cells: Vec<(usize, usize)>,
    pub score: f32,
}

const START: i8 = i8::MIN;

/// The strongest non-overlapping falling paths in bins `q_lo..=q_hi` of `img`'s score image.
pub fn track_paths(img: &GuideImage, q_lo: usize, q_hi: usize, p: &TrackParams) -> Vec<Path> {
    let n = img.times.len();
    let nb = img.n_bins();
    if n < 2 || q_lo >= q_hi || q_hi >= nb {
        return Vec::new();
    }
    let w = q_hi - q_lo + 1;
    debug_assert!(p.max_fall <= 127, "max_fall is stored as i8");
    let k = p.max_fall as i64;
    // back[j][q]: d where the path came from bin q + d at frame j − 1, or START.
    let mut back: Vec<Vec<i8>> = Vec::with_capacity(n);
    let mut prev = vec![0f32; w];
    let mut cur = vec![0f32; w];
    // For each path start (frame, bin), the best (score, end frame, end bin) reached.
    let mut best_by_start: std::collections::HashMap<(u32, u16), (f32, usize, usize)> =
        std::collections::HashMap::new();
    let cell = |j: usize, i: usize| img.score_row(j)[q_lo + i].clamp(p.clip.0, p.clip.1);
    let mut bias = vec![0f32; w];
    for j in 0..n {
        for (i, b) in bias.iter_mut().enumerate() {
            // Always max_fall + 1 cells (the window slides inside the band at its top edge):
            // a truncated max there would under-bias the edge and seed paths on noise.
            let a = i.min(w.saturating_sub(k as usize + 1));
            let best = (0..=k as usize)
                .take_while(|d| a + d < w)
                .map(|d| cell(j, a + d) - p.fall_cost * d.saturating_sub(1) as f32)
                .fold(f32::NEG_INFINITY, f32::max);
            *b += best;
        }
    }
    for b in bias.iter_mut() {
        *b = *b / n as f32 + p.bias_margin;
    }
    let mut start_prev = vec![(0u32, 0u16); w];
    let mut start_cur = vec![(0u32, 0u16); w];
    for j in 0..n {
        let mut bj = vec![START; w];
        for i in 0..w {
            let x = cell(j, i) - bias[i];
            let mut best = 0f32;
            let mut from = START;
            if j > 0 {
                for d in 0..=k {
                    let s = i as i64 + d;
                    if s >= w as i64 {
                        break;
                    }
                    let c = prev[s as usize] - p.fall_cost * (d - 1).max(0) as f32;
                    if c > best {
                        best = c;
                        from = d as i8;
                    }
                }
            }
            let v = best + x;
            if v <= 0.0 {
                cur[i] = 0.0;
                bj[i] = START;
                start_cur[i] = (j as u32, i as u16);
                continue;
            }
            cur[i] = v;
            bj[i] = from;
            start_cur[i] = if from == START {
                (j as u32, i as u16)
            } else {
                start_prev[(i as i64 + i64::from(from)) as usize]
            };
            if v >= p.min_score {
                let e = best_by_start.entry(start_cur[i]).or_insert((0.0, j, i));
                if v > e.0 {
                    *e = (v, j, i);
                }
            }
        }
        back.push(bj);
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut start_prev, &mut start_cur);
    }
    let mut ends: Vec<(f32, usize, usize)> = best_by_start.into_values().collect();
    ends.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut taken: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut out = Vec::new();
    for (score, j_end, i_end) in ends {
        if out.len() >= p.max_paths {
            break;
        }
        let mut cells = Vec::new();
        let (mut j, mut i) = (j_end, i_end);
        loop {
            cells.push((j, q_lo + i));
            let d = back[j][i];
            if d == START || j == 0 {
                break;
            }
            i = (i as i64 + i64::from(d)) as usize;
            j -= 1;
        }
        cells.reverse();
        let span = img.times[j_end] - img.times[cells[0].0];
        if span < p.min_duration_s {
            continue;
        }
        // Skip paths mostly on cells an earlier (stronger) path took.
        let overlap = |t: &Vec<(usize, usize)>| {
            cells
                .iter()
                .filter(|&&(cj, cq)| {
                    let a = t.partition_point(|x| x.0 < cj);
                    t.get(a).is_some_and(|x| x.0 == cj && x.1.abs_diff(cq) <= 3)
                })
                .count()
        };
        if taken.iter().any(|t| overlap(t) * 3 >= cells.len()) {
            continue;
        }
        taken.push(cells.clone());
        out.push(Path { cells, score });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A standardised noise image (0 ± 1) with `curve(j) -> bin` lit at `level` for frames in
    /// `frames`.
    fn image(n: usize, nb: usize, seed: u64) -> GuideImage {
        let mut x = seed;
        let level = (0..n)
            .map(|_| {
                (0..nb)
                    .map(|_| {
                        // Sum of 4 uniforms: near-Gaussian, unit variance.
                        let mut s = 0f32;
                        for _ in 0..4 {
                            x = x
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            s += (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        }
                        s * 3f32.sqrt()
                    })
                    .collect()
            })
            .collect();
        GuideImage {
            times: (0..n).map(|j| 0.5 + j as f64 * 0.25).collect(),
            hop_s: 0.25,
            f_lo: 30.0,
            bpo: crate::audio::guide::GUIDE_BPO,
            score: Vec::new(),
            level,
        }
    }

    #[test]
    fn tracker_follows_a_faint_one_sided_sweep() {
        // A sweep at +2.5σ per cell for 40 s that falls 120 bins and ends at its steepest
        // (a pass cut off at CPA): the strongest path follows it.
        let mut img = image(800, 400, 11);
        let bin = |j: usize| 300.0 - 120.0 * ((j - 200) as f64 / 160.0).powi(2);
        for j in 200..360 {
            img.level[j][bin(j).round() as usize] += 2.5;
        }
        let paths = track_paths(&img, 10, 390, &TrackParams::default());
        let best = &paths[0];
        let on = best
            .cells
            .iter()
            .filter(|&&(j, q)| (200..360).contains(&j) && (q as f64 - bin(j)).abs() <= 2.0)
            .count();
        assert!(
            on >= 120,
            "{on} of 160 cells on the sweep: {:?}",
            &best.cells[..5]
        );
    }

    #[test]
    fn tracker_noise_paths_stay_short_and_weak() {
        // Pure noise: whatever paths reach the proposal score stay far below a real sweep's.
        let img = image(800, 400, 5);
        let paths = track_paths(&img, 10, 390, &TrackParams::default());
        assert!(
            paths.iter().all(|p| p.score < 60.0),
            "{:?}",
            paths.first().map(|p| p.score)
        );
    }
}
