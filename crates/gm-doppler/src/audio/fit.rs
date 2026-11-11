//! Constant-velocity fit of a ridge by variable projection: Levenberg–Marquardt over
//! (v, ln d, tc), with the rest frequency f₀ solved in closed form at every step.
//!
//! A pass is rarely one unbroken ridge in a real recording: it fades, and what remains are
//! stretches on different partials. Every partial of one aircraft follows the same curve, so
//! [`fit_joint`] fits several tracks at once: one (v, d, tc), a rest frequency per track.
//! [`fit_cv`] is the one-track case.

use serde::Serialize;

use crate::physics::cv::{best_f0, shape};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct CvFit {
    pub v: f64,
    pub d: f64,
    pub tc: f64,
    pub f0: f64,
    pub rms_hz: f64,
    pub sigma_v: f64,
    pub sigma_d: f64,
    pub sigma_tc: f64,
    pub n: usize,
}

/// Several tracks of one pass fitted together.
#[derive(Clone, Debug, PartialEq)]
pub struct JointFit {
    /// The shared geometry. `f0` is the reference track's rest frequency and `rms_hz` the
    /// weighted residual on that track's scale.
    pub fit: CvFit,
    /// Rest frequency of each track, in input order.
    pub f0: Vec<f64>,
    /// Residual rms of each track in its own Hz, over the points the robust trim kept.
    pub rms_hz: Vec<f64>,
}

/// Expected frequency noise of a ridge point at frequency `f`: a floor set by the
/// spectrogram's resolution plus a share that grows with frequency (the source's own wobble,
/// which every partial carries in proportion). It sets the weights between tracks in a joint
/// fit: a high partial is neither treated as exact in relative terms nor drowned in Hz.
pub fn track_noise_hz(f: f64) -> f64 {
    f64::hypot(TRACK_NOISE_FLOOR_HZ, TRACK_NOISE_REL * f)
}

const TRACK_NOISE_FLOOR_HZ: f64 = 0.5;
const TRACK_NOISE_REL: f64 = 0.002;

/// Fit (t, f) points; None if the points show no Doppler drop or the fit fails.
pub fn fit_cv(points: &[(f64, f64)], c: f64) -> Option<CvFit> {
    let n = points.len();
    if n < 8
        || !(c.is_finite() && c > 0.0)
        || points
            .iter()
            .any(|(t, f)| !(t.is_finite() && f.is_finite()))
    {
        return None;
    }
    let bounds = Bounds::around(points[0].0, points[n - 1].0, c);
    let init = initial_guess(points, c)?;
    // Coarse grid around the initial guess, then LM from the best cell.
    let mut best = (f64::INFINITY, init);
    for fv in [0.6, 0.8, 1.0, 1.25, 1.5] {
        for fd in [0.4, 0.7, 1.0, 1.4, 2.5] {
            for dt in [-4.0, -2.0, 0.0, 2.0, 4.0] {
                let th = bounds.clamp([init[0] * fv, init[1] + f64::ln(fd), init[2] + dt]);
                let e = sse(&[points], &[1.0], th, c);
                if e < best.0 {
                    best = (e, th);
                }
            }
        }
    }
    solve(&[points], &[1.0], 0, best.1, c, &bounds).map(|j| j.fit)
}

/// Fit several tracks of one pass together, starting from `start` = [v, d, tc].
///
/// Each track is a time-sorted run of (t, f) points on one partial; the partials may sit at any
/// ratio to one another. Tracks are weighted by [`track_noise_hz`] at their mean frequency, and
/// residuals are expressed on the `reference` track's scale, so one track gives what [`fit_cv`]
/// gives from the same start. None on bad input or a degenerate fit.
pub fn fit_joint(
    tracks: &[&[(f64, f64)]],
    reference: usize,
    start: [f64; 3],
    c: f64,
) -> Option<JointFit> {
    let n: usize = tracks.iter().map(|t| t.len()).sum();
    if reference >= tracks.len()
        || n < 8
        || !(c.is_finite() && c > 0.0)
        || !(start.iter().all(|x| x.is_finite()) && start[1] > 0.0)
        || tracks
            .iter()
            .any(|t| t.is_empty() || t.iter().any(|(t, f)| !(t.is_finite() && f.is_finite())))
    {
        return None;
    }
    let times = || tracks.iter().flat_map(|t| t.iter().map(|p| p.0));
    let bounds = Bounds::around(
        times().fold(f64::INFINITY, f64::min),
        times().fold(f64::NEG_INFINITY, f64::max),
        c,
    );
    let noise: Vec<f64> = tracks
        .iter()
        .map(|t| track_noise_hz(t.iter().map(|p| p.1).sum::<f64>() / t.len() as f64))
        .collect();
    let w: Vec<f64> = noise.iter().map(|s| noise[reference] / s).collect();
    let start = bounds.clamp([start[0], start[1].ln(), start[2]]);
    solve(tracks, &w, reference, start, c, &bounds)
}

/// `tracks` without the points whose residual (`r`, in the tracks' order) lies far beyond the
/// spread: that of all the points together, or with `own` that of each track's own points.
fn shed(tracks: &[Vec<(f64, f64)>], r: &[f64], own: bool) -> Vec<Vec<(f64, f64)>> {
    let cut = |r: &[f64]| {
        let mut abs: Vec<f64> = r.iter().map(|x| x.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let mad = abs[abs.len() / 2] * 1.4826;
        (4.5 * mad).max(0.5)
    };
    let shared = cut(r);
    let mut at = 0;
    tracks
        .iter()
        .map(|t| {
            let r = &r[at..at + t.len()];
            at += t.len();
            let cut = if own { cut(r) } else { shared };
            t.iter()
                .zip(r)
                .filter(|(_, x)| x.abs() <= cut)
                .map(|(q, _)| *q)
                .collect()
        })
        .collect()
}

/// Levenberg–Marquardt from `start`, a robust trim, then the uncertainties.
fn solve(
    tracks: &[&[(f64, f64)]],
    w: &[f64],
    reference: usize,
    start: [f64; 3],
    c: f64,
    bounds: &Bounds,
) -> Option<JointFit> {
    let n: usize = tracks.iter().map(|t| t.len()).sum();
    let mut kept: Vec<Vec<(f64, f64)>> = tracks.iter().map(|t| t.to_vec()).collect();
    let mut theta = levenberg_marquardt(&kept, w, start, c, bounds);
    // Robust refinement: drop points far beyond the residual spread (contamination from hum,
    // crossing tracks, codec artefacts) and refit, at most twice; keep at least 70% of points
    // and never empty a track.
    let count = |ts: &[Vec<(f64, f64)>]| ts.iter().map(Vec::len).sum::<usize>();
    let may = |next: &[Vec<(f64, f64)>]| {
        count(next) >= (n * 7).div_ceil(10).max(8)
            && next
                .iter()
                .zip(tracks)
                .all(|(a, b)| a.len() >= b.len().min(2))
    };
    for _ in 0..2 {
        let r = residuals(&kept, w, theta, c);
        let before = count(&kept);
        let mut next = shed(&kept, &r, false);
        if count(&next) == before {
            break;
        }
        if !may(&next) {
            // More lies beyond the cut than may go. That is what strays do to a track: they
            // pull its rest frequency, and with it every one of its points, off the curve the
            // other tracks sit on. Held against its own spread, such a track sheds its strays.
            next = shed(&kept, &r, true);
            if count(&next) == before || !may(&next) {
                break;
            }
        }
        kept = next;
        theta = levenberg_marquardt(&kept, w, theta, c, bounds);
    }
    let n: usize = kept.iter().map(Vec::len).sum();
    let (v, d, tc) = (theta[0], theta[1].exp(), theta[2]);
    let f0: Vec<f64> = kept
        .iter()
        .map(|t| best_f0(t, v, d, tc, c))
        .collect::<Option<_>>()?;
    let rms_hz: Vec<f64> = kept
        .iter()
        .zip(&f0)
        .map(|(t, f0)| {
            let e: f64 = t
                .iter()
                .map(|&(t, f)| (f - f0 * shape(t, v, d, tc, c)).powi(2))
                .sum();
            (e / t.len() as f64).sqrt()
        })
        .collect();
    let e = sse(&kept, w, theta, c);
    // Three shared parameters and a rest frequency per track.
    let dof = (n as f64 - 3.0 - kept.len() as f64).max(1.0);
    let rms = (e / n as f64).sqrt();
    let jac = jacobian(&kept, w, theta, c);
    let cov = inverse3(normal_matrix(&jac))?;
    let s2 = e / dof;
    let sd = |i: usize| (cov[i][i].max(0.0) * s2).sqrt();
    Some(JointFit {
        fit: CvFit {
            v,
            d,
            tc,
            f0: f0[reference],
            rms_hz: rms,
            sigma_v: sd(0),
            sigma_d: d * sd(1),
            sigma_tc: sd(2),
            n,
        },
        f0,
        rms_hz,
    })
}

struct Bounds {
    v: (f64, f64),
    ln_d: (f64, f64),
    tc: (f64, f64),
}

impl Bounds {
    /// The physical box for points spanning `t_lo..t_hi`.
    fn around(t_lo: f64, t_hi: f64, c: f64) -> Self {
        Self {
            v: (20.0, 0.8 * c),
            ln_d: (50f64.ln(), 20_000f64.ln()),
            tc: (t_lo - 30.0, t_hi + 30.0),
        }
    }

    fn clamp(&self, th: [f64; 3]) -> [f64; 3] {
        [
            th[0].clamp(self.v.0, self.v.1),
            th[1].clamp(self.ln_d.0, self.ln_d.1),
            th[2].clamp(self.tc.0, self.tc.1),
        ]
    }
}

/// Start from the steepest descent (tc), the asymptote ratio (v) and the slope at tc (d).
fn initial_guess(points: &[(f64, f64)], c: f64) -> Option<[f64; 3]> {
    let n = points.len();
    let k = (n / 10).max(2);
    let f_early = points[..k].iter().map(|p| p.1).sum::<f64>() / k as f64;
    let f_late = points[n - k..].iter().map(|p| p.1).sum::<f64>() / k as f64;
    let r = f_early / f_late;
    if r.is_nan() || r <= 1.002 {
        return None;
    }
    let v = (c * (r - 1.0) / (r + 1.0)).clamp(20.0, 0.8 * c);
    let f0 = 2.0 * f_early * f_late / (f_early + f_late);
    // Smoothed slope over ±w points.
    let w = (n / 40).max(2);
    let mut steepest = (0.0f64, points[n / 2].0);
    for i in w..n - w {
        let (a, b) = (points[i - w], points[i + w]);
        let slope = (b.1 - a.1) / (b.0 - a.0);
        if slope < steepest.0 {
            steepest = (slope, points[i].0);
        }
    }
    let d = if steepest.0 < 0.0 {
        (-f0 * v * v / (c * steepest.0)).clamp(60.0, 19_000.0)
    } else {
        2000.0
    };
    Some([v, d.ln(), steepest.1])
}

/// Residuals of every track against the shared curve, each track with its own closed-form f₀
/// and its weight, concatenated in track order.
fn residuals<T: AsRef<[(f64, f64)]>>(tracks: &[T], w: &[f64], th: [f64; 3], c: f64) -> Vec<f64> {
    let (v, d, tc) = (th[0], th[1].exp(), th[2]);
    let mut out = Vec::with_capacity(tracks.iter().map(|t| t.as_ref().len()).sum());
    for (points, wk) in tracks.iter().zip(w) {
        let points = points.as_ref();
        let f0 = best_f0(points, v, d, tc, c).unwrap_or(0.0);
        out.extend(
            points
                .iter()
                .map(|&(t, f)| (f - f0 * shape(t, v, d, tc, c)) * wk),
        );
    }
    out
}

fn sse<T: AsRef<[(f64, f64)]>>(tracks: &[T], w: &[f64], th: [f64; 3], c: f64) -> f64 {
    let e: f64 = residuals(tracks, w, th, c).iter().map(|r| r * r).sum();
    if e.is_finite() { e } else { f64::INFINITY }
}

/// Central-difference Jacobian of the residuals (f₀ re-projected at each evaluation).
fn jacobian<T: AsRef<[(f64, f64)]>>(
    tracks: &[T],
    w: &[f64],
    th: [f64; 3],
    c: f64,
) -> Vec<[f64; 3]> {
    let steps = [1e-4 * th[0].abs().max(1.0), 1e-5, 1e-4];
    let mut cols: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (j, col) in cols.iter_mut().enumerate() {
        let (mut a, mut b) = (th, th);
        a[j] += steps[j];
        b[j] -= steps[j];
        let (ra, rb) = (residuals(tracks, w, a, c), residuals(tracks, w, b, c));
        *col = ra
            .iter()
            .zip(&rb)
            .map(|(x, y)| (x - y) / (2.0 * steps[j]))
            .collect();
    }
    (0..cols[0].len())
        .map(|i| [cols[0][i], cols[1][i], cols[2][i]])
        .collect()
}

fn normal_matrix(jac: &[[f64; 3]]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for row in jac {
        for a in 0..3 {
            for b in 0..3 {
                m[a][b] += row[a] * row[b];
            }
        }
    }
    m
}

fn inverse3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() < 1e-300 {
        return None;
    }
    let inv = 1.0 / det;
    let c = |a: usize, b: usize, x: usize, y: usize| m[a][b] * m[x][y] - m[a][y] * m[x][b];
    Some([
        [
            c(1, 1, 2, 2) * inv,
            -c(0, 1, 2, 2) * inv,
            c(0, 1, 1, 2) * inv,
        ],
        [
            -c(1, 0, 2, 2) * inv,
            c(0, 0, 2, 2) * inv,
            -c(0, 0, 1, 2) * inv,
        ],
        [
            c(1, 0, 2, 1) * inv,
            -c(0, 0, 2, 1) * inv,
            c(0, 0, 1, 1) * inv,
        ],
    ])
}

fn levenberg_marquardt<T: AsRef<[(f64, f64)]>>(
    tracks: &[T],
    w: &[f64],
    start: [f64; 3],
    c: f64,
    b: &Bounds,
) -> [f64; 3] {
    let mut th = start;
    let mut e = sse(tracks, w, th, c);
    let mut lambda = 1e-3;
    for _ in 0..200 {
        let jac = jacobian(tracks, w, th, c);
        let r = residuals(tracks, w, th, c);
        let mut g = [0.0; 3];
        for (row, ri) in jac.iter().zip(&r) {
            for k in 0..3 {
                g[k] += row[k] * ri;
            }
        }
        let jtj = normal_matrix(&jac);
        let mut improved = false;
        for _ in 0..12 {
            let mut a = jtj;
            for (k, row) in a.iter_mut().enumerate() {
                row[k] += lambda * jtj[k][k].max(1e-12);
            }
            let Some(inv) = inverse3(a) else {
                lambda *= 10.0;
                continue;
            };
            let delta: [f64; 3] =
                std::array::from_fn(|i| -(inv[i][0] * g[0] + inv[i][1] * g[1] + inv[i][2] * g[2]));
            let cand = b.clamp([th[0] + delta[0], th[1] + delta[1], th[2] + delta[2]]);
            let ec = sse(tracks, w, cand, c);
            if ec < e {
                let rel = (e - ec) / e.max(1e-300);
                th = cand;
                e = ec;
                lambda = (lambda * 0.3).max(1e-12);
                improved = true;
                if rel < 1e-12 {
                    return th;
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    th
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            (-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()
        }
    }

    fn pass(v: f64, d: f64, tc: f64, f0: f64, sigma: f64, seed: u64) -> Vec<(f64, f64)> {
        let mut n = noise(seed);
        (0..400)
            .map(|i| {
                let t = tc - 50.0 + f64::from(i) * 0.25;
                (t, f0 * shape(t, v, d, tc, 343.0) + sigma * n())
            })
            .collect()
    }

    #[test]
    fn recovers_noise_free() {
        let f = fit_cv(&pass(140.0, 2300.0, 41.7, 87.0, 0.0, 1), 343.0).unwrap();
        for (got, want) in [(f.v, 140.0), (f.d, 2300.0), (f.tc, 41.7), (f.f0, 87.0)] {
            assert!(((got - want) / want).abs() < 1e-4, "{f:?}");
        }
        assert_eq!(f.n, 400);
    }

    #[test]
    fn recovers_noisy() {
        let mut ev = Vec::new();
        let mut sigmas = Vec::new();
        for seed in 1..=20 {
            let f = fit_cv(&pass(140.0, 2300.0, 41.7, 87.0, 0.15, seed), 343.0).unwrap();
            assert!((f.v - 140.0).abs() / 140.0 < 0.02, "{f:?}");
            assert!((f.d - 2300.0).abs() / 2300.0 < 0.05, "{f:?}");
            assert!((f.tc - 41.7).abs() < 0.2, "{f:?}");
            ev.push(f.v);
            sigmas.push(f.sigma_v);
        }
        let mean = ev.iter().sum::<f64>() / 20.0;
        let spread = (ev.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 19.0).sqrt();
        let sig = sigmas.iter().sum::<f64>() / 20.0;
        assert!(
            sig > 0.3 * spread && sig < 3.0 * spread,
            "sigma {sig} vs spread {spread}"
        );
    }

    #[test]
    fn rejects_flat_ridge() {
        let flat: Vec<(f64, f64)> = (0..200).map(|i| (f64::from(i) * 0.25, 120.0)).collect();
        match fit_cv(&flat, 343.0) {
            None => {}
            Some(f) => assert!(f.sigma_d > f.d, "{f:?}"),
        }
        assert!(fit_cv(&[], 343.0).is_none());
        assert!(fit_cv(&[(0.0, 100.0), (1.0, 99.0)], 343.0).is_none());
    }

    /// One stretch of a pass on one partial: `f0 * shape` over [t0, t1], with noise.
    fn stretch(
        geom: [f64; 3],
        f0: f64,
        span: (f64, f64),
        sigma: f64,
        seed: u64,
    ) -> Vec<(f64, f64)> {
        let mut n = noise(seed);
        let steps = ((span.1 - span.0) / 0.25) as u32;
        (0..=steps)
            .map(|i| {
                let t = span.0 + f64::from(i) * 0.25;
                (
                    t,
                    f0 * shape(t, geom[0], geom[1], geom[2], 343.0) + sigma * n(),
                )
            })
            .collect()
    }

    /// A pass that fades between partials: the approach arm on one partial, the middle on
    /// another, the receding arm on a third at a non-integer ratio. No piece shows the whole
    /// curve; together they pin it.
    #[test]
    fn joint_fit_recovers_pass_from_fragments() {
        let geom = [86.0, 907.0, 32.5];
        let truth_f0 = [108.0, 84.98, 743.4];
        let tracks = [
            stretch(geom, truth_f0[0], (1.0, 25.0), 0.3, 11),
            stretch(geom, truth_f0[1], (26.0, 38.0), 0.3, 12),
            stretch(geom, truth_f0[2], (36.0, 59.0), 1.0, 13),
        ];
        let views: Vec<&[(f64, f64)]> = tracks.iter().map(Vec::as_slice).collect();
        let j = fit_joint(&views, 1, [60.0, 500.0, 28.0], 343.0).unwrap();
        assert!((j.fit.v - 86.0).abs() / 86.0 < 0.03, "{j:?}");
        assert!((j.fit.d - 907.0).abs() / 907.0 < 0.05, "{j:?}");
        assert!((j.fit.tc - 32.5).abs() < 0.3, "{j:?}");
        assert_eq!(j.f0.len(), 3);
        for (got, want) in j.f0.iter().zip(truth_f0) {
            assert!((got - want).abs() / want < 0.003, "{j:?}");
        }
        assert_eq!(
            j.fit.f0, j.f0[1],
            "the fit is on the reference track's scale"
        );
        for (got, sigma) in j.rms_hz.iter().zip([0.3, 0.3, 1.0]) {
            assert!(*got > 0.5 * sigma && *got < 1.6 * sigma, "{j:?}");
        }
        assert!(
            j.fit.sigma_v < 0.2 * j.fit.v && j.fit.sigma_d < j.fit.d,
            "{j:?}"
        );
    }

    #[test]
    fn joint_fit_of_one_track_is_fit_cv() {
        let pts = pass(140.0, 2300.0, 41.7, 87.0, 0.15, 3);
        let single = fit_cv(&pts, 343.0).unwrap();
        let j = fit_joint(&[&pts], 0, [120.0, 1800.0, 40.0], 343.0).unwrap();
        for (got, want) in [
            (j.fit.v, single.v),
            (j.fit.d, single.d),
            (j.fit.tc, single.tc),
            (j.fit.f0, single.f0),
            (j.fit.rms_hz, single.rms_hz),
            (j.fit.sigma_v, single.sigma_v),
            (j.fit.sigma_d, single.sigma_d),
        ] {
            assert!(((got - want) / want).abs() < 1e-3, "{j:?} vs {single:?}");
        }
        assert_eq!(j.fit.n, single.n);
    }

    /// A noisy low partial must not drown a clean high one: tracks are weighted by the noise
    /// expected at their frequency, not equally in Hz and not equally in relative terms. The
    /// fit's residual is that weighted one, on the reference track's scale.
    #[test]
    fn joint_fit_weights_tracks_by_expected_noise() {
        assert!((track_noise_hz(0.0) - 0.5).abs() < 1e-12);
        assert!(track_noise_hz(800.0) > 3.0 * track_noise_hz(80.0) / 2.0);
        assert!(track_noise_hz(800.0) < 10.0 * track_noise_hz(80.0) / 2.0);

        let geom = [86.0, 907.0, 32.5];
        let low = stretch(geom, 84.98, (5.0, 60.0), 0.3, 21);
        let high = stretch(geom, 743.4, (5.0, 60.0), 1.5, 22);
        let start = [80.0, 800.0, 30.0];
        // As many points on each; the high track's scatter counts by the ratio of the noises
        // expected at the tracks' mean frequencies.
        let mean = |t: &[(f64, f64)]| t.iter().map(|p| p.1).sum::<f64>() / t.len() as f64;
        let ratio = track_noise_hz(mean(&low)) / track_noise_hz(mean(&high));
        let want = ((0.3f64.powi(2) + (ratio * 1.5).powi(2)) / 2.0).sqrt();
        let on_low = fit_joint(&[&low, &high], 0, start, 343.0).unwrap();
        assert!(
            (on_low.fit.rms_hz / want - 1.0).abs() < 0.1,
            "{on_low:?} against {want}"
        );
        // The same fit on the high track's scale: every residual grows by that ratio.
        let on_high = fit_joint(&[&low, &high], 1, start, 343.0).unwrap();
        assert!(
            (on_high.fit.rms_hz * ratio / on_low.fit.rms_hz - 1.0).abs() < 0.02,
            "{on_high:?} against {on_low:?}"
        );
        assert!(
            (on_high.fit.v / on_low.fit.v - 1.0).abs() < 1e-3,
            "{on_high:?} against {on_low:?}"
        );
    }

    /// Stray points at the head of one track (a chain of noise peaks the linker led into the
    /// tone) pull that track's rest frequency, so all of its points sit some hertz off the
    /// curve. The other track lies on the curve to a fraction of a hertz; against a spread
    /// pooled over both, most of the pulled track would lie beyond the cut, more than the trim
    /// may shed, and it would shed nothing. Against its own spread only the strays go.
    #[test]
    fn joint_fit_sheds_strays_that_bend_one_track() {
        let geom = [80.0, 780.0, 300.0];
        let mut approach = stretch(geom, 1800.0, (198.0, 306.5), 0.1, 31);
        // The other track is the same tone in a band that ends at 1600 Hz: until the tone
        // comes down into the band, its points sit at the band's ceiling.
        let receding: Vec<(f64, f64)> = stretch(geom, 1800.0, (304.5, 485.0), 0.1, 32)
            .into_iter()
            .map(|(t, f)| (t, f.min(1595.0)))
            .collect();
        // The first 13 s: a point a second, from 460 Hz above the tone down to 90 Hz above it.
        approach = approach
            .into_iter()
            .enumerate()
            .filter(|&(i, (t, _))| t >= 211.0 || i % 4 == 0)
            .map(|(_, (t, f))| {
                let stray = if t < 211.0 {
                    460.0 - (t - 198.0) / 13.0 * 370.0
                } else {
                    0.0
                };
                (t, f + stray)
            })
            .collect();
        let j = fit_joint(&[&approach, &receding], 1, [76.0, 759.0, 298.0], 343.0).unwrap();
        assert!((j.fit.v - 80.0).abs() / 80.0 < 0.01, "{j:?}");
        assert!((j.fit.d - 780.0).abs() / 780.0 < 0.02, "{j:?}");
        assert!((j.fit.tc - 300.0).abs() < 0.1, "{j:?}");
        for rms in &j.rms_hz {
            assert!(*rms < 0.3, "{j:?}");
        }
    }

    /// A weak partial tracked through noise, its points several hertz about the curve, beside a
    /// clean one. By its own spread none of the weak track's points is far; by the spread all
    /// the points share most of them are, and they may go. Where the shared cut may be applied
    /// it is the one applied, so the fit rests on the clean track.
    #[test]
    fn joint_fit_holds_a_loose_track_to_the_shared_spread() {
        let geom = [86.0, 907.0, 32.5];
        let clean = stretch(geom, 108.0, (1.0, 59.0), 0.1, 21);
        let loose = stretch(geom, 85.0, (25.0, 40.0), 4.0, 22);
        let j = fit_joint(&[&clean, &loose], 0, [60.0, 500.0, 28.0], 343.0).unwrap();
        assert!(j.fit.n < clean.len() + loose.len() / 2, "{j:?}");
        assert!((j.fit.v - 86.0).abs() / 86.0 < 0.01, "{j:?}");
        assert!((j.fit.d - 907.0).abs() / 907.0 < 0.02, "{j:?}");
        assert!((j.fit.tc - 32.5).abs() < 0.1, "{j:?}");
    }

    #[test]
    fn joint_fit_rejects_bad_input() {
        let pts = pass(140.0, 2300.0, 41.7, 87.0, 0.0, 1);
        let start = [140.0, 2300.0, 41.7];
        assert!(fit_joint(&[], 0, start, 343.0).is_none());
        assert!(fit_joint(&[&pts], 1, start, 343.0).is_none());
        assert!(fit_joint(&[&pts[..5]], 0, start, 343.0).is_none());
        assert!(fit_joint(&[&pts, &[]], 0, start, 343.0).is_none());
        assert!(fit_joint(&[&pts], 0, [140.0, -1.0, 41.7], 343.0).is_none());
        assert!(fit_joint(&[&pts], 0, [f64::NAN, 2300.0, 41.7], 343.0).is_none());
    }
}
