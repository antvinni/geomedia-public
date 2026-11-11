//! Pass pursuit: Doppler passes from ridge fragments.
//!
//! A real pass fades in and out, so what the ridge linker returns is a handful of stretches
//! on different partials, none showing the whole curve; fitted one at a time they come out
//! slow and close. Pursuit treats them as what they are: every partial of one aircraft follows
//! the same curve, whatever the ratio between the partials. From a clearly falling fragment
//! (the seed) it scans a bank of curves for the one that the most frequency movement in the
//! neighbourhood agrees with, fits all agreeing fragments together ([`fit_joint`]), and
//! collects again around the fitted curve until the set settles, [`REFINE_ROUNDS`] times at
//! most.
//!
//! A pass scores what its curve explains of the fragments' movement, less what any curve gets
//! for nothing: its three numbers can be spent to match the slopes of three straight pieces,
//! so those count for nothing ([`FREE_SLOPES`]) and a pass must show more ([`SCORE_FLOOR`]).
//!
//! The linker joins a tone across short silences, and what it joins need not be one tone: the
//! steady harmonics of a voice, joined from syllable to syllable, step down as a pass falls.
//! So the curve must also explain movement within the stretches no silence breaks
//! ([`WITHIN_FLOOR`]), where no join is in play.
//!
//! What it finds is tested against the recording itself: the same pursuit runs on the
//! fragments mirrored in time, where an aircraft's falling tones rise and cannot make a pass.
//! Whatever the mirror finds measures what music, speech and engines offer by chance, and a
//! pass must beat it.

use super::events::{
    Event, EventParams, PASS_SPAN_D_OVER_V, PASS_SPAN_PAD_S, harmonic_ratio, inflate_sigma,
    plausible, vehicle_like,
};
use super::fit::{JointFit, fit_cv, fit_joint, track_noise_hz};
use super::ridges::{FRAGMENT_MIN_POINTS, Ridge};
use super::templates::logspace;
use crate::physics::cv::shape;

/// The bank of curves a seed is scanned against: Mach number and τ = d/v, log-spaced
/// (min, max, steps), with the closest approach stepped at τ/4.
pub const BANK_MACH: (f64, f64, usize) = (0.08, 0.75, 12);
pub const BANK_TAU_S: (f64, f64, usize) = (3.0, 150.0, 14);
const BANK_TC_STEP_MIN_S: f64 = 0.5;
/// Bank curves at distances where a fit counts as pinned to its bounds (`plausible`) are not
/// tried.
const BANK_D_M: (f64, f64) = (60.0, 19_000.0);
/// Points per fragment in the bank scan (evenly spaced; the fit uses all of them).
const VOTE_POINTS: usize = 32;
/// A fragment votes for a bank curve that explains this share of its frequency movement.
const VOTE_GATE: f64 = 0.75;
/// A fragment moves (and may vote) when its spread reaches this many times its expected noise.
const VOTER_SIGMAS: f64 = 1.0;
/// A seed falls by at least this many times its noise between its first and last fifth.
const SEED_FALL_SIGMAS: f64 = 2.0;
/// Fit, collect, fit again: at most this many times.
const REFINE_ROUNDS: usize = 4;
/// On the fitted curve: residual within this many times the expected noise …
const MEMBER_RMS_SIGMAS: f64 = 2.0;
/// … and, to count as evidence, this share of the fragment's movement explained.
const MEMBER_GATE: f64 = 0.5;
/// A fragment the curve does not explain (flat, or lost in its noise) joins only on a partial
/// that already carries evidence: rest frequencies within this.
const SAME_PARTIAL_REL: f64 = 0.005;
/// Residuals are judged over the points left after dropping this share of the worst: the
/// linker wanders at a fragment's ends.
const TRIM_FRAC: f64 = 0.1;
/// A new pass whose every fragment lies this close to a curve already found is that pass
/// again, not a second aircraft.
const RIDER_RMS_SIGMAS: f64 = 3.0;
/// A member moves to another pass when its residual there is at most this share of its
/// residual here.
const SETTLE_RMS_RATIO: f64 = 0.75;
/// The closest approach lies within the fragments' span, give or take this.
const TC_SLACK_S: f64 = 15.0;
/// A pass is heard without long silences: its fragments make a chain from the seed with no
/// gap in time longer than this. A piece further off may lie on the curve, but nothing ties
/// it to the pass, and a far, slow curve runs near any slowly falling piece.
const CHAIN_GAP_S: f64 = 20.0;
/// A curve has three numbers to spend (v, d, tc): enough to match the slopes of any three
/// straight pieces, each at its own pitch. So the three largest straight-line falls among a
/// pass's pieces earn nothing; what counts is the bend the curve explains within a piece and
/// the fall of every further piece whose slope it foretells.
const FREE_SLOPES: usize = 3;
/// Movement is counted in units of the noise a track is allowed. A pass whose tracks lie
/// closer to its curve than that is counted in units of their own scatter, down to this share
/// of the allowed noise: on a clean track a slight bend is plain.
const OWN_SCATTER_MIN: f64 = 0.5;
/// The least a pass scores. Chains of unrelated pieces in quiet recordings, where no mirrored
/// pass lies near enough to judge them by, score up to about 130 (the mirrored passes of the
/// development recordings); nine real passes in ten score above 300.
const SCORE_FLOOR: f64 = 150.0;
/// A fragment is unbroken where consecutive points lie closer than this, the length of the
/// analysis window. Points further apart share no samples: that the tone after the silence
/// is the tone before it is the linker's guess.
const BREAK_S: f64 = 1.0;
/// The least a pass's curve explains within the unbroken stretches of its evidence, in noise
/// units. Where nothing moves within the stretches, a curve gains there only what its three
/// numbers fit of the noise: under 16.3 in 999 cases of 1000 (χ² with three degrees of
/// freedom). The false passes on speech in the development recordings come to −18 and −10,
/// the lowest real pass to 45.
const WITHIN_FLOOR: f64 = 16.0;
/// A pass must score more than this times the best mirrored pass near it.
const NULL_MARGIN: f64 = 1.0;
const NULL_PAD_S: f64 = 30.0;

/// Events among ridge fragments (ids unset, unsorted). `windows` (seconds) restricts the
/// search, empty = everything.
pub fn events_from_fragments(
    fragments: Vec<Ridge>,
    p: &EventParams,
    windows: &[(f64, f64)],
) -> Vec<Event> {
    run(fragments, p, windows, &mut Vec::new())
}

/// What pursuit makes of each seed among `fragments`, one line per step: the vote, the fit,
/// the verdict, and the mirrored passes each survivor was held against. Only the lines about
/// the stretch `span` (seconds) of the recording.
pub fn explain_fragments(
    fragments: Vec<Ridge>,
    p: &EventParams,
    windows: &[(f64, f64)],
    span: (f64, f64),
) -> Vec<String> {
    let mut log = Vec::new();
    run(fragments, p, windows, &mut log);
    log.into_iter()
        .filter(|&(t0, t1, _)| t1 >= span.0 && t0 <= span.1)
        .map(|(_, _, line)| line)
        .collect()
}

/// The step log: each line with the stretch of the recording (seconds) it is about.
type Log = Vec<(f64, f64, String)>;

fn run(
    fragments: Vec<Ridge>,
    p: &EventParams,
    windows: &[(f64, f64)],
    log: &mut Log,
) -> Vec<Event> {
    let in_window =
        |t: f64| windows.is_empty() || windows.iter().any(|&(lo, hi)| t >= lo && t <= hi);
    let frags: Vec<Frag> = fragments
        .into_iter()
        .filter_map(|mut r| {
            r.points.retain(|q| in_window(q.t));
            (r.points.len() >= FRAGMENT_MIN_POINTS).then(|| Frag::of(&r))
        })
        .collect();
    let mirror: Vec<Frag> = frags.iter().map(Frag::mirrored).collect();
    let mut mirror_log = Vec::new();
    let null = pursue(&mirror, p, true, &mut mirror_log);
    log.extend(
        mirror_log
            .into_iter()
            .map(|(t0, t1, l)| (t0, t1, format!("mirror: {l}"))),
    );
    let corr = (p.spec.window_s / p.spec.hop_s).max(1.0).sqrt();
    let passes = pursue(&frags, p, false, log);
    let mut out = Vec::new();
    for q in passes {
        let worst = null_score(&q, &null);
        let kept = q.score > NULL_MARGIN * worst;
        log.push((
            q.t0,
            q.t1,
            format!(
                "pass tc {:.1}: score {:.0} against the mirror's {:.0}: {}",
                q.fit.fit.tc,
                q.score,
                worst,
                if kept { "event" } else { "no more than chance" }
            ),
        ));
        if kept {
            out.push(event(&frags, &q, p.c, corr));
        }
    }
    out
}

/// The best score among mirrored passes near `q` (their times are negated).
fn null_score(q: &Pass, null: &[Pass]) -> f64 {
    null.iter()
        .filter(|a| -a.t0 >= q.t0 - NULL_PAD_S && -a.t1 <= q.t1 + NULL_PAD_S)
        .map(|a| a.score)
        .fold(0.0, f64::max)
}

/// One fragment, with what the scan needs worked out once.
struct Frag {
    pts: Vec<(f64, f64)>,
    snr_sum: f32,
    t0: f64,
    t1: f64,
    /// Expected frequency noise of a point at this fragment's pitch.
    noise: f64,
    mean_f: f64,
    /// Σ(f − mean)² over all points.
    sse_const: f64,
    /// For each point, which unbroken stretch of the fragment it lies in ([`BREAK_S`]).
    stretch: Vec<usize>,
    voter: bool,
    falls: bool,
    sub: Vec<(f64, f64)>,
    sub_sff: f64,
    sub_sse_const: f64,
}

impl Frag {
    fn of(r: &Ridge) -> Self {
        Self::new(
            r.points.iter().map(|q| (q.t, q.f)).collect(),
            r.points.iter().map(|q| q.snr_db).sum(),
        )
    }

    fn new(pts: Vec<(f64, f64)>, snr_sum: f32) -> Self {
        let n = pts.len();
        let mean = |q: &[(f64, f64)]| q.iter().map(|p| p.1).sum::<f64>() / q.len() as f64;
        let spread = |q: &[(f64, f64)]| {
            let m = mean(q);
            q.iter().map(|p| (p.1 - m).powi(2)).sum::<f64>()
        };
        let noise = track_noise_hz(mean(&pts));
        let sse_const = spread(&pts);
        let fifth = (n / 5).max(1);
        let mut k = 0;
        let stretch: Vec<usize> = (0..n)
            .map(|i| {
                if i > 0 && pts[i].0 - pts[i - 1].0 >= BREAK_S {
                    k += 1;
                }
                k
            })
            .collect();
        let sub: Vec<(f64, f64)> = if n <= VOTE_POINTS {
            pts.clone()
        } else {
            (0..VOTE_POINTS)
                .map(|i| pts[i * (n - 1) / (VOTE_POINTS - 1)])
                .collect()
        };
        Self {
            snr_sum,
            t0: pts[0].0,
            t1: pts[n - 1].0,
            noise,
            mean_f: mean(&pts),
            sse_const,
            stretch,
            voter: sse_const / n as f64 >= (VOTER_SIGMAS * noise).powi(2),
            falls: mean(&pts[..fifth]) - mean(&pts[n - fifth..]) >= SEED_FALL_SIGMAS * noise,
            sub_sff: sub.iter().map(|p| p.1 * p.1).sum(),
            sub_sse_const: spread(&sub),
            sub,
            pts,
        }
    }

    /// The same fragment played backwards.
    fn mirrored(&self) -> Self {
        Self::new(
            self.pts.iter().rev().map(|&(t, f)| (-t, f)).collect(),
            self.snr_sum,
        )
    }

    /// How much of this fragment's movement the curve `h` = [v, d, tc] explains, in noise
    /// units, if it explains enough to vote at all.
    fn vote(&self, h: [f64; 3], c: f64) -> Option<f64> {
        let (mut sfs, mut sss) = (0.0, 0.0);
        for &(t, f) in &self.sub {
            let s = shape(t, h[0], h[1], h[2], c);
            sfs += f * s;
            sss += s * s;
        }
        let sse = self.sub_sff - sfs * sfs / sss;
        (sse <= (1.0 - VOTE_GATE) * self.sub_sse_const).then(|| {
            (self.sub_sse_const - sse) * (self.pts.len() as f64 / self.sub.len() as f64)
                / self.noise.powi(2)
        })
    }

    /// How this fragment sits on the curve `h`, over the points left after the trim.
    fn on_curve(&self, h: [f64; 3], c: f64) -> Option<OnCurve> {
        struct Row {
            f: f64,
            /// The curve's shape at the point's time.
            s: f64,
            /// Squared residual, before the trim.
            sq: f64,
            t: f64,
            stretch: usize,
        }
        let rest = |rows: &[Row]| {
            let (sfs, sss) = rows
                .iter()
                .fold((0.0, 0.0), |(a, b), r| (a + r.f * r.s, b + r.s * r.s));
            (sss > 0.0 && sfs > 0.0).then(|| sfs / sss)
        };
        let mut rows: Vec<Row> = self
            .pts
            .iter()
            .zip(&self.stretch)
            .map(|(&(t, f), &stretch)| Row {
                f,
                s: shape(t, h[0], h[1], h[2], c),
                sq: 0.0,
                t,
                stretch,
            })
            .collect();
        let f0 = rest(&rows)?;
        for r in &mut rows {
            r.sq = (r.f - f0 * r.s).powi(2);
        }
        let keep = rows.len() - (TRIM_FRAC * rows.len() as f64) as usize;
        if keep < rows.len() {
            rows.select_nth_unstable_by(keep - 1, |a, b| a.sq.total_cmp(&b.sq));
            rows.truncate(keep);
        }
        let f0 = rest(&rows)?;
        let n = rows.len() as f64;
        let mean = rows.iter().map(|r| r.f).sum::<f64>() / n;
        let sse = rows.iter().map(|r| (r.f - f0 * r.s).powi(2)).sum::<f64>();
        let sse_const = rows.iter().map(|r| (r.f - mean).powi(2)).sum::<f64>();
        // The same points about their own straight line.
        let mean_t = rows.iter().map(|r| r.t).sum::<f64>() / n;
        let (stt, stf) = rows.iter().fold((0.0, 0.0), |(a, b), r| {
            (
                a + (r.t - mean_t).powi(2),
                b + (r.t - mean_t) * (r.f - mean),
            )
        });
        let sse_line = if stt > 0.0 {
            sse_const - stf * stf / stt
        } else {
            sse_const
        };
        // Stretch by stretch, the points and the curve each about their own mean there: what
        // the curve explains of the movement inside the stretch, whatever its level.
        // Per stretch: n, Σf, Σg, Σfg, Σg² (g the curve; both less the fragment's mean).
        let mut sums = vec![[0.0f64; 5]; self.stretch.last().map_or(0, |&k| k + 1)];
        for r in &rows {
            let (f, g) = (r.f - mean, f0 * r.s - mean);
            let a = &mut sums[r.stretch];
            *a = [a[0] + 1.0, a[1] + f, a[2] + g, a[3] + f * g, a[4] + g * g];
        }
        let within: f64 = sums
            .iter()
            .filter(|a| a[0] > 0.0)
            .map(|a| 2.0 * (a[3] - a[1] * a[2] / a[0]) - (a[4] - a[2] * a[2] / a[0]))
            .sum();
        Some(OnCurve {
            f0,
            rms: (sse / n).sqrt(),
            gain: (sse_const - sse) / self.noise.powi(2),
            line: (sse_const - sse_line) / self.noise.powi(2),
            within: within / self.noise.powi(2),
            left: sse / self.noise.powi(2),
            n,
            explained: if sse_const > 0.0 {
                1.0 - sse / sse_const
            } else {
                0.0
            },
        })
    }
}

struct OnCurve {
    /// Rest frequency of the fragment's partial on this curve.
    f0: f64,
    rms: f64,
    /// Movement the curve explains, in noise units.
    gain: f64,
    /// Movement the fragment's own straight line explains, in noise units: its slope and no
    /// more.
    line: f64,
    /// Movement the curve explains within the fragment's unbroken stretches, in noise units:
    /// the jumps from one stretch to the next count for nothing.
    within: f64,
    /// What the curve leaves unexplained, in noise units, and over how many points.
    left: f64,
    n: f64,
    explained: f64,
}

/// A group of fragments on one curve.
struct Pass {
    /// The fragment it grew from.
    seed: usize,
    fit: JointFit,
    /// Fragment indices, in the order of the fit's tracks.
    members: Vec<usize>,
    /// Frequency movement the curve explains over its members, less the [`FREE_SLOPES`]
    /// largest straight-line falls among them; in units of the noise a track is allowed, or
    /// of the members' own scatter about the curve where that is less ([`OWN_SCATTER_MIN`]).
    score: f64,
    /// Movement the curve explains within the unbroken stretches of its evidence, in noise
    /// units.
    within: f64,
    /// How many members are evidence: fragments whose movement the curve explains.
    carriers: usize,
    t0: f64,
    t1: f64,
}

impl Pass {
    /// The fragments fitted together from `start` = [v, d, tc]. None if the fit fails.
    fn fit(
        frags: &[Frag],
        seed: usize,
        members: Vec<usize>,
        start: [f64; 3],
        c: f64,
    ) -> Option<Self> {
        // The reference track: the best measured one.
        let reference = (0..members.len()).max_by(|&a, &b| {
            frags[members[a]]
                .snr_sum
                .total_cmp(&frags[members[b]].snr_sum)
        })?;
        let tracks: Vec<&[(f64, f64)]> = members.iter().map(|&i| &frags[i].pts[..]).collect();
        let fit = fit_joint(&tracks, reference, start, c)?;
        let h = [fit.fit.v, fit.fit.d, fit.fit.tc];
        let evidence: Vec<OnCurve> = members
            .iter()
            .filter(|&&i| frags[i].voter)
            .filter_map(|&i| frags[i].on_curve(h, c))
            .filter(|o| o.explained >= MEMBER_GATE)
            .collect();
        let mut lines: Vec<f64> = evidence.iter().map(|o| o.line).collect();
        lines.sort_by(|a, b| b.total_cmp(a));
        let free: f64 = lines.iter().take(FREE_SLOPES).sum();
        // The members' own scatter about the curve, squared, as a share of the allowed noise.
        let (left, n) = evidence
            .iter()
            .fold((0.0, 0.0), |(a, b), o| (a + o.left, b + o.n));
        let unit = if n > 0.0 {
            (left / n).clamp(OWN_SCATTER_MIN.powi(2), 1.0)
        } else {
            1.0
        };
        Some(Self {
            seed,
            score: (evidence.iter().map(|o| o.gain).sum::<f64>() - free) / unit,
            within: evidence.iter().map(|o| o.within).sum(),
            carriers: evidence.len(),
            t0: members
                .iter()
                .map(|&i| frags[i].t0)
                .fold(f64::INFINITY, f64::min),
            t1: members
                .iter()
                .map(|&i| frags[i].t1)
                .fold(f64::NEG_INFINITY, f64::max),
            fit,
            members,
        })
    }

    /// [v, d, tc].
    fn h(&self) -> [f64; 3] {
        [self.fit.fit.v, self.fit.fit.d, self.fit.fit.tc]
    }

    /// Does `g` reach into the time this pass is heard?
    fn hears(&self, g: &Frag) -> bool {
        let f = &self.fit.fit;
        let half = PASS_SPAN_D_OVER_V * f.d / f.v.max(1.0) + PASS_SPAN_PAD_S;
        g.t1 >= f.tc - half && g.t0 <= f.tc + half
    }

    /// Does `g` lie near this pass's curve: one more piece of the same aircraft?
    fn carries(&self, g: &Frag, c: f64) -> bool {
        self.hears(g)
            && g.on_curve(self.h(), c)
                .is_some_and(|o| o.rms <= RIDER_RMS_SIGMAS * g.noise)
    }
}

/// Every pass among `frags` that stands on its own (the mirrored null comes after).
/// `mirrored` only turns the log's times back to the recording's.
fn pursue(frags: &[Frag], p: &EventParams, mirrored: bool, log: &mut Log) -> Vec<Pass> {
    let between = |t0: f64, t1: f64| if mirrored { (-t1, -t0) } else { (t0, t1) };
    let span = |g: &Frag| between(g.t0, g.t1);
    let when = |t: f64| if mirrored { -t } else { t };
    let mut free = vec![true; frags.len()];
    let mut seeds: Vec<usize> = (0..frags.len())
        .filter(|&i| {
            let g = &frags[i];
            g.voter && g.falls && g.t1 - g.t0 >= p.ridge.min_duration_s
        })
        .collect();
    // Strongest first: the fragment with the most movement to explain.
    let strength = |i: usize| frags[i].sse_const / frags[i].noise.powi(2);
    seeds.sort_by(|&a, &b| strength(b).total_cmp(&strength(a)));
    let mut out: Vec<Pass> = Vec::new();
    for seed in seeds {
        if !free[seed] {
            continue;
        }
        let g = &frags[seed];
        let name = format!(
            "seed {:.1} Hz {:.1}..{:.1} s",
            g.mean_f,
            span(g).0,
            span(g).1
        );
        let mut note = |(t0, t1): (f64, f64), line: String| log.push((t0, t1, line));
        if vehicle_alone(g, p.c) {
            note(
                span(g),
                format!("{name}: a road vehicle's sweep on its own, no seed"),
            );
            continue;
        }
        let Some((h, first)) = vote(frags, &free, seed, p.c) else {
            note(span(g), format!("{name}: no bank curve explains it"));
            continue;
        };
        let voted = format!(
            "vote v {:.0} d {:.0} tc {:.1} with {} fragments",
            h[0],
            h[1],
            when(h[2]),
            first.len()
        );
        let Some((pass, alone)) = refine(frags, &free, seed, h, first, p.c) else {
            note(
                span(g),
                format!("{name}: {voted}; the fit fails or leaves the seed"),
            );
            continue;
        };
        let voted = if alone {
            format!("{voted}; grown from the seed alone")
        } else {
            voted
        };
        let verdict = verdict(frags, &pass, p);
        // The same aircraft again: every piece already lies near a curve that was found.
        let again = verdict.is_ok()
            && out
                .iter()
                .any(|a| pass.members.iter().all(|&i| a.carries(&frags[i], p.c)));
        let f = &pass.fit.fit;
        let around = between(pass.t0, pass.t1);
        note(
            around,
            format!(
                "{name}: {}; {voted}; fit v {:.1} d {:.0} tc {:.2} rms {:.2}, score {:.0}",
                match &verdict {
                    Ok(()) if again => "a pass already found",
                    Ok(()) => "pass",
                    Err(Reject::Vehicle) => "a road vehicle",
                    Err(Reject::Other(why)) => why,
                },
                f.v,
                f.d,
                when(f.tc),
                f.rms_hz,
                pass.score,
            ),
        );
        for (&i, (f0, rms)) in pass
            .members
            .iter()
            .zip(pass.fit.f0.iter().zip(&pass.fit.rms_hz))
        {
            let (a, b) = span(&frags[i]);
            note(
                around,
                format!("    track {f0:.1} Hz {a:.1}..{b:.1} s rms {rms:.2}"),
            );
        }
        // A vehicle's partials are spent too: none may seed again and drift past the veto.
        if matches!(verdict, Ok(()) | Err(Reject::Vehicle)) {
            for &i in &pass.members {
                free[i] = false;
            }
        }
        if verdict.is_ok() && !again {
            out.push(pass);
        }
    }
    for (tc, line) in settle(frags, &mut out, p) {
        let (t0, t1) = between(tc, tc);
        log.push((
            t0,
            t1,
            format!("settled: the pass at tc {:.1} {line}", when(tc)),
        ));
    }
    out
}

/// A fragment whose own fit is a road vehicle's does not seed a pass (it may still belong to
/// one: the middle of a pass fitted alone can look close and slow).
fn vehicle_alone(g: &Frag, c: f64) -> bool {
    fit_cv(&g.pts, c).is_some_and(|f| {
        plausible(&f, c)
            && f.tc >= g.t0 - TC_SLACK_S
            && f.tc <= g.t1 + TC_SLACK_S
            && vehicle_like(&f)
    })
}

/// Which of the stretches `spans` are tied to the stretch `from`, directly or through each
/// other, by gaps in time of at most [`CHAIN_GAP_S`].
fn chained(from: (f64, f64), spans: &[(f64, f64)]) -> Vec<bool> {
    let (mut lo, mut hi) = from;
    let mut on = vec![false; spans.len()];
    loop {
        let mut grew = false;
        for (k, &(a, b)) in spans.iter().enumerate() {
            if !on[k] && a <= hi + CHAIN_GAP_S && b >= lo - CHAIN_GAP_S {
                on[k] = true;
                (lo, hi) = (lo.min(a), hi.max(b));
                grew = true;
            }
        }
        if !grew {
            return on;
        }
    }
}

/// The bank curve [v, d, tc] that explains the seed and, with it, the most movement among the
/// free fragments chained to it; and the fragments that voted for it (seed included,
/// ascending).
fn vote(frags: &[Frag], free: &[bool], seed: usize, c: f64) -> Option<([f64; 3], Vec<usize>)> {
    let s = &frags[seed];
    let half = |tau: f64| PASS_SPAN_D_OVER_V * tau + PASS_SPAN_PAD_S;
    let reach = PASS_SPAN_D_OVER_V * BANK_TAU_S.1 + half(BANK_TAU_S.1);
    let near: Vec<usize> = (0..frags.len())
        .filter(|&i| {
            let g = &frags[i];
            i != seed && free[i] && g.voter && g.t1 >= s.t0 - reach && g.t0 <= s.t1 + reach
        })
        .collect();
    // The fragments that vote for `h` and hang together with the seed, each with its vote.
    let voters = |h: [f64; 3], tau: f64| -> Vec<(usize, f64)> {
        let all: Vec<(usize, f64)> = near
            .iter()
            .filter_map(|&i| {
                let g = &frags[i];
                (g.t1 >= h[2] - half(tau) && g.t0 <= h[2] + half(tau))
                    .then(|| g.vote(h, c))
                    .flatten()
                    .map(|x| (i, x))
            })
            .collect();
        let spans: Vec<(f64, f64)> = all
            .iter()
            .map(|&(i, _)| (frags[i].t0, frags[i].t1))
            .collect();
        let on = chained((s.t0, s.t1), &spans);
        all.into_iter()
            .zip(on)
            .filter_map(|(v, on)| on.then_some(v))
            .collect()
    };
    let mut best: Option<(f64, [f64; 3], f64)> = None;
    for mach in logspace(BANK_MACH) {
        let v = mach * c;
        for tau in logspace(BANK_TAU_S) {
            let d = tau * v;
            if d <= BANK_D_M.0 || d >= BANK_D_M.1 {
                continue;
            }
            let step = (tau / 4.0).max(BANK_TC_STEP_MIN_S);
            let mut tc = s.t0 - PASS_SPAN_D_OVER_V * tau;
            while tc <= s.t1 + PASS_SPAN_D_OVER_V * tau {
                let h = [v, d, tc];
                if let Some(own) = s.vote(h, c) {
                    let total = own + voters(h, tau).iter().map(|&(_, x)| x).sum::<f64>();
                    if best.is_none_or(|(b, _, _)| total > b) {
                        best = Some((total, h, tau));
                    }
                }
                tc += step;
            }
        }
    }
    let (_, h, tau) = best?;
    let mut first: Vec<usize> = voters(h, tau).into_iter().map(|(i, _)| i).collect();
    first.push(seed);
    first.sort_unstable();
    Some((h, first))
}

/// Does `g` belong on the curve `h`, alongside partials at rest frequencies `tones`? Either
/// the curve explains its movement, or it merely agrees with the curve on a partial that is
/// already there. Returns how it sits and whether it is evidence (a carrier).
fn belongs(g: &Frag, h: [f64; 3], tones: &[f64], c: f64) -> Option<(OnCurve, bool)> {
    let o = g
        .on_curve(h, c)
        .filter(|o| o.rms <= MEMBER_RMS_SIGMAS * g.noise)?;
    let carrier = g.voter && o.explained >= MEMBER_GATE;
    (carrier
        || tones
            .iter()
            .any(|t| (o.f0 / t - 1.0).abs() <= SAME_PARTIAL_REL))
    .then_some((o, carrier))
}

/// The free fragments on the curve `h` that hang together with `seed` (ascending), and
/// whether `seed` is evidence for the curve.
fn crew(frags: &[Frag], free: &[bool], seed: usize, h: [f64; 3], c: f64) -> (Vec<usize>, bool) {
    let half = PASS_SPAN_D_OVER_V * h[1] / h[0].max(1.0) + PASS_SPAN_PAD_S;
    let near: Vec<usize> = (0..frags.len())
        .filter(|&i| free[i] && frags[i].t1 >= h[2] - half && frags[i].t0 <= h[2] + half)
        .collect();
    let from = (frags[seed].t0, frags[seed].t1);
    let tied = |set: Vec<usize>| -> Vec<usize> {
        let spans: Vec<(f64, f64)> = set.iter().map(|&i| (frags[i].t0, frags[i].t1)).collect();
        let on = chained(from, &spans);
        set.into_iter()
            .zip(on)
            .filter_map(|(i, on)| on.then_some(i))
            .collect()
    };
    // Carriers first: they say which partials are there.
    let carriers: Vec<(usize, f64)> = near
        .iter()
        .filter_map(|&i| match belongs(&frags[i], h, &[], c) {
            Some((o, true)) => Some((i, o.f0)),
            _ => None,
        })
        .collect();
    let seeded = carriers.iter().any(|&(i, _)| i == seed);
    let evidence = tied(carriers.iter().map(|&(i, _)| i).collect());
    let tones: Vec<f64> = carriers
        .iter()
        .filter(|(i, _)| evidence.contains(i))
        .map(|&(_, f0)| f0)
        .collect();
    // Then whatever else sits on the curve, on those partials or as evidence a quiet piece
    // ties in.
    let mut all = tied(
        near.into_iter()
            .filter(|i| {
                carriers.iter().any(|(k, _)| k == i) || belongs(&frags[*i], h, &tones, c).is_some()
            })
            .collect(),
    );
    all.sort_unstable();
    (all, seeded)
}

/// The pass that grows from the voters and, where none does, the pass that grows from the seed
/// alone, marked `true`. An aircraft that changes speed while it is heard lies on no one
/// curve: its pieces fitted together leave each of them behind, and what gathers around one
/// piece's own curve is the stretch of the pass that curve holds for. Something must gather:
/// falling tones near each other that share no curve, each alone on its own, are clutter.
fn refine(
    frags: &[Frag],
    free: &[bool],
    seed: usize,
    start: [f64; 3],
    first: Vec<usize>,
    c: f64,
) -> Option<(Pass, bool)> {
    let alone = first.len() == 1;
    if let Some(pass) = grow(frags, free, seed, start, first, c) {
        return Some((pass, false));
    }
    if alone {
        return None;
    }
    grow(frags, free, seed, start, vec![seed], c)
        .filter(|pass| pass.carriers > 1)
        .map(|pass| (pass, true))
}

/// Fit `first` together, collect whatever lies on the fitted curve, and repeat until the set
/// settles or [`REFINE_ROUNDS`] fits are done. None when the fit fails or the curve leaves the
/// seed behind.
fn grow(
    frags: &[Frag],
    free: &[bool],
    seed: usize,
    start: [f64; 3],
    first: Vec<usize>,
    c: f64,
) -> Option<Pass> {
    let mut pass = Pass::fit(frags, seed, first, start, c)?;
    for round in 0.. {
        let (next, seeded) = crew(frags, free, seed, pass.h(), c);
        if !seeded {
            return None;
        }
        if next == pass.members || round + 1 == REFINE_ROUNDS {
            break;
        }
        pass = Pass::fit(frags, seed, next, pass.h(), c)?;
    }
    Some(pass)
}

/// Each fragment to the curve that explains it best. Passes are found one after another, and
/// the first takes what fits it well enough; where two aircraft overlap, an arm of the second
/// can sit acceptably on the first's curve. Once all passes are known, a member that lies
/// clearly closer to another pass's curve moves there (seeds stay), and the passes that
/// changed are fitted again. If a refit fails, nothing moves. Returns what changed, for the
/// log: each refitted pass's closest approach and a few words.
fn settle(frags: &[Frag], passes: &mut [Pass], p: &EventParams) -> Vec<(f64, String)> {
    let mut members: Vec<Vec<usize>> = passes.iter().map(|q| q.members.clone()).collect();
    let mut moved = vec![false; passes.len()];
    for (a, from) in passes.iter().enumerate() {
        for &i in from.members.iter().filter(|&&i| i != from.seed) {
            let g = &frags[i];
            let Some(own) = g.on_curve(from.h(), p.c) else {
                continue;
            };
            let better = passes
                .iter()
                .enumerate()
                .filter(|&(b, to)| {
                    b != a && g.t0 <= to.t1 + CHAIN_GAP_S && g.t1 >= to.t0 - CHAIN_GAP_S
                })
                .filter_map(|(b, to)| Some((b, belongs(g, to.h(), &to.fit.f0, p.c)?.0.rms)))
                .filter(|&(_, rms)| rms <= SETTLE_RMS_RATIO * own.rms)
                .min_by(|x, y| x.1.total_cmp(&y.1));
            if let Some((b, _)) = better {
                members[a].retain(|&m| m != i);
                members[b].push(i);
                (moved[a], moved[b]) = (true, true);
            }
        }
    }
    let mut refits = Vec::new();
    for (k, q) in passes.iter().enumerate().filter(|&(k, _)| moved[k]) {
        members[k].sort_unstable();
        let fit = Pass::fit(frags, q.seed, members[k].clone(), q.h(), p.c)
            .filter(|n| verdict(frags, n, p).is_ok());
        let Some(n) = fit else {
            return Vec::new();
        };
        refits.push((k, n));
    }
    refits
        .into_iter()
        .map(|(k, n)| {
            let line = format!(
                "now has {} fragments (was {}), score {:.0}",
                n.members.len(),
                passes[k].members.len(),
                n.score
            );
            passes[k] = n;
            (passes[k].fit.fit.tc, line)
        })
        .collect()
}

enum Reject {
    Vehicle,
    Other(&'static str),
}

/// Is this group of fragments an aircraft's pass (before the mirrored null)?
fn verdict(frags: &[Frag], q: &Pass, p: &EventParams) -> Result<(), Reject> {
    let f = &q.fit.fit;
    if !plausible(f, p.c) {
        return Err(Reject::Other("the joint fit is not a plausible pass"));
    }
    if f.tc < q.t0 - TC_SLACK_S || f.tc > q.t1 + TC_SLACK_S {
        return Err(Reject::Other("closest approach outside the fragments"));
    }
    if vehicle_like(f) {
        return Err(Reject::Vehicle);
    }
    // Time the fragments cover between them (overlaps counted once).
    let mut spans: Vec<(f64, f64)> = q
        .members
        .iter()
        .map(|&i| (frags[i].t0, frags[i].t1))
        .collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut covered, mut end) = (0.0, f64::NEG_INFINITY);
    for (a, b) in spans {
        covered += (b - a.max(end)).max(0.0);
        end = end.max(b);
    }
    if covered < p.ridge.min_duration_s {
        return Err(Reject::Other("too short"));
    }
    let at = |t: f64| shape(t, f.v, f.d, f.tc, p.c);
    if at(q.t0) / at(q.t1) - 1.0 < p.ridge.min_drop_frac {
        return Err(Reject::Other("too little of the fall is seen"));
    }
    if q.score < SCORE_FLOOR {
        return Err(Reject::Other(
            "no more than three straight pieces would show",
        ));
    }
    if q.within < WITHIN_FLOOR {
        return Err(Reject::Other(
            "only the jumps between pieces follow the curve",
        ));
    }
    Ok(())
}

/// The event of a pass: every member's points scaled onto the reference partial.
fn event(frags: &[Frag], q: &Pass, c: f64, corr: f64) -> Event {
    let mut fit = q.fit.fit;
    let mut pts: Vec<(f64, f64)> = Vec::new();
    let (mut snr_sum, mut snr_n) = (0.0f32, 0usize);
    // The scatter of the scaled points: each track's own, grown as its points are.
    let mut sq = 0.0;
    for ((&i, &tone), &rms) in q.members.iter().zip(&q.fit.f0).zip(&q.fit.rms_hz) {
        let g = &frags[i];
        let scale = fit.f0 / tone;
        pts.extend(g.pts.iter().map(|&(t, f)| (t, f * scale)));
        sq += g.pts.len() as f64 * (rms * scale).powi(2);
        snr_sum += g.snr_sum;
        snr_n += g.pts.len();
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    // The joint fit's residual is weighted by track; what follows discovery reads `rms_hz`
    // as the scatter of the event's points.
    fit.rms_hz = (sq / snr_n.max(1) as f64).sqrt();
    // Drop points far off the fit (what the robust fit trimmed).
    let cut = (5.0 * fit.rms_hz).max(1.0);
    pts.retain(|&(t, f)| (f - fit.f0 * shape(t, fit.v, fit.d, fit.tc, c)).abs() <= cut);
    // The fundamental: the lowest tone, divided down if the tones share an unseen lower base.
    let lowest = q.fit.f0.iter().copied().fold(f64::INFINITY, f64::min);
    let base = q
        .fit
        .f0
        .iter()
        .filter_map(|&t| harmonic_ratio(t, lowest))
        .map(|(_, m)| m)
        .max()
        .unwrap_or(1);
    let f0_fundamental = lowest / f64::from(base);
    let harmonic = (fit.f0 / f0_fundamental).round().max(1.0) as u32;
    inflate_sigma(&mut fit, corr);
    Event {
        id: 0,
        t_start: q.t0,
        t_end: q.t1,
        fit,
        harmonic,
        f0_fundamental,
        ridges: q.members.len(),
        mean_snr_db: snr_sum / snr_n.max(1) as f32,
        points: pts,
        guided: false,
        guide_icao: None,
        guide_z: None,
        guide_null_z: None,
        guide_null_n: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::ridges::RidgePoint;
    use crate::physics::cv::shape;

    const C: f64 = 343.0;

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

    /// A stretch of a pass on one partial, as the ridge linker would return it.
    fn arm(geom: [f64; 3], f0: f64, span: (f64, f64), sigma: f64, seed: u64) -> Ridge {
        let mut n = noise(seed);
        let steps = ((span.1 - span.0) / 0.25) as u32;
        Ridge {
            points: (0..=steps)
                .map(|i| {
                    let t = span.0 + f64::from(i) * 0.25;
                    RidgePoint {
                        t,
                        f: f0 * shape(t, geom[0], geom[1], geom[2], C) + sigma * n(),
                        snr_db: 20.0,
                    }
                })
                .collect(),
        }
    }

    /// The same stretch played backwards: a tone that rises as a pass would fall.
    fn rising(geom: [f64; 3], f0: f64, span: (f64, f64), sigma: f64, seed: u64) -> Ridge {
        let mut r = arm(geom, f0, span, sigma, seed);
        let fs: Vec<f64> = r.points.iter().rev().map(|q| q.f).collect();
        for (q, f) in r.points.iter_mut().zip(fs) {
            q.f = f;
        }
        r
    }

    /// A straight piece: the chord of the pass's curve over `span`. Short pieces of unrelated
    /// tones look like this when some curve happens to run along them.
    fn chord(geom: [f64; 3], f0: f64, span: (f64, f64), sigma: f64, seed: u64) -> Ridge {
        let mut n = noise(seed);
        let at = |t: f64| f0 * shape(t, geom[0], geom[1], geom[2], C);
        let (a, b) = (at(span.0), at(span.1));
        let steps = ((span.1 - span.0) / 0.25) as u32;
        Ridge {
            points: (0..=steps)
                .map(|i| {
                    let t = span.0 + f64::from(i) * 0.25;
                    RidgePoint {
                        t,
                        f: a + (b - a) * (t - span.0) / (span.1 - span.0) + sigma * n(),
                        snr_db: 20.0,
                    }
                })
                .collect(),
        }
    }

    /// One partial heard in separate stretches (seconds), joined into a single fragment as the
    /// linker joins them across short silences. `steady`: each stretch holds the pitch the curve
    /// has at its middle, so that only the jumps between stretches follow the curve.
    fn broken(
        geom: [f64; 3],
        f0: f64,
        stretches: &[(f64, f64)],
        steady: bool,
        sigma: f64,
        seed: u64,
    ) -> Ridge {
        let mut n = noise(seed);
        let at = |t: f64| f0 * shape(t, geom[0], geom[1], geom[2], C);
        let mut points = Vec::new();
        for &(a, b) in stretches {
            let steps = ((b - a) / 0.25) as u32;
            for i in 0..=steps {
                let t = a + f64::from(i) * 0.25;
                let f = if steady { at((a + b) / 2.0) } else { at(t) };
                points.push(RidgePoint {
                    t,
                    f: f + sigma * n(),
                    snr_db: 20.0,
                });
            }
        }
        Ridge { points }
    }

    fn hum(f: f64, span: (f64, f64), seed: u64) -> Ridge {
        let mut n = noise(seed);
        let steps = ((span.1 - span.0) / 0.25) as u32;
        Ridge {
            points: (0..=steps)
                .map(|i| RidgePoint {
                    t: span.0 + f64::from(i) * 0.25,
                    f: f + 0.2 * n(),
                    snr_db: 20.0,
                })
                .collect(),
        }
    }

    fn run(fragments: Vec<Ridge>) -> Vec<Event> {
        let mut ev = events_from_fragments(fragments, &EventParams::default(), &[]);
        ev.sort_by(|a, b| a.fit.tc.total_cmp(&b.fit.tc));
        ev
    }

    /// Events without their points, for failure messages.
    fn brief(ev: &[Event]) -> Vec<String> {
        ev.iter()
            .map(|e| {
                format!(
                    "v {:.1} d {:.0} tc {:.2} f0 {:.1} rms {:.2} ridges {} {:.1}..{:.1}",
                    e.fit.v,
                    e.fit.d,
                    e.fit.tc,
                    e.fit.f0,
                    e.fit.rms_hz,
                    e.ridges,
                    e.t_start,
                    e.t_end
                )
            })
            .collect()
    }

    /// The pursuit's own account, for failure messages.
    fn story(fragments: Vec<Ridge>) -> String {
        explain_fragments(
            fragments,
            &EventParams::default(),
            &[],
            (f64::NEG_INFINITY, f64::INFINITY),
        )
        .join("\n")
    }

    fn close(e: &Event, geom: [f64; 3]) -> bool {
        (e.fit.v - geom[0]).abs() / geom[0] < 0.05
            && (e.fit.d - geom[1]).abs() / geom[1] < 0.08
            && (e.fit.tc - geom[2]).abs() < 0.5
    }

    /// The fragments one real pass left in a 60 s clip: arms and stretches on six partials,
    /// one of them at 8.75 × the fundamental, none showing the whole curve.
    fn faded_pass(geom: [f64; 3]) -> Vec<Ridge> {
        let t = geom[2] - 32.5;
        vec![
            arm(geom, 108.0, (t + 1.0, t + 25.0), 0.3, 1),
            arm(geom, 84.98, (t + 26.0, t + 38.0), 0.3, 2),
            arm(geom, 743.4, (t + 36.0, t + 59.0), 1.2, 3),
            arm(geom, 169.9, (t + 36.0, t + 59.0), 0.4, 4),
            arm(geom, 215.0, (t + 19.0, t + 23.5), 0.4, 5),
            arm(geom, 215.0, (t + 39.0, t + 46.0), 0.4, 6),
        ]
    }

    #[test]
    fn faded_pass_is_one_event() {
        let geom = [86.0, 907.0, 32.5];
        let mut fr = faded_pass(geom);
        fr.push(hum(120.0, (0.0, 60.0), 7));
        fr.push(hum(50.0, (5.0, 40.0), 8));
        fr.push(rising([60.0, 400.0, 12.0], 300.0, (9.0, 15.0), 0.3, 9));
        let ev = run(fr);
        assert_eq!(ev.len(), 1, "{:?}", brief(&ev));
        let e = &ev[0];
        assert!(close(e, geom), "{:?}", e.fit);
        assert!(e.ridges >= 5, "{} ridges", e.ridges);
        assert!(
            e.t_start <= 1.5 && e.t_end >= 58.5,
            "{}-{}",
            e.t_start,
            e.t_end
        );
        // Every point lies on the fitted curve of the reference partial.
        for &(t, f) in &e.points {
            let m = e.fit.f0 * shape(t, e.fit.v, e.fit.d, e.fit.tc, C);
            assert!((f - m).abs() < 3.0, "{t} {f} vs {m}");
        }
    }

    /// What comes after discovery reads the event's `rms_hz` as the scatter of its points. A
    /// low partial scaled onto a high reference scatters by more than the joint fit's
    /// weighted residual says: the event must carry what its points show, and keep them.
    #[test]
    fn the_event_rms_is_the_scatter_of_its_points() {
        let geom = [86.0, 907.0, 40.0];
        let ev = run(vec![
            arm(geom, 743.4, (5.0, 75.0), 1.2, 1),
            arm(geom, 84.98, (10.0, 60.0), 0.5, 2),
        ]);
        assert_eq!(ev.len(), 1, "{:?}", brief(&ev));
        let e = &ev[0];
        assert!((e.fit.f0 / 743.4 - 1.0).abs() < 0.01, "{:?}", e.fit);
        assert!(e.points.len() >= 470, "{} of 482 points", e.points.len());
        let sq: f64 = e
            .points
            .iter()
            .map(|&(t, f)| (f - e.fit.f0 * shape(t, e.fit.v, e.fit.d, e.fit.tc, C)).powi(2))
            .sum();
        let scatter = (sq / e.points.len() as f64).sqrt();
        assert!(
            (e.fit.rms_hz / scatter - 1.0).abs() < 0.15,
            "rms {} against the points' {scatter}",
            e.fit.rms_hz
        );
    }

    #[test]
    fn one_whole_ridge_is_a_one_track_pass() {
        let geom = [140.0, 2300.0, 41.7];
        let r = arm(geom, 87.0, (0.0, 90.0), 0.15, 3);
        let pts: Vec<(f64, f64)> = r.points.iter().map(|q| (q.t, q.f)).collect();
        let alone = crate::audio::fit_cv(&pts, C).unwrap();
        let ev = run(vec![r]);
        assert_eq!(ev.len(), 1, "{:?}", brief(&ev));
        let f = &ev[0].fit;
        for (got, want) in [(f.v, alone.v), (f.d, alone.d), (f.f0, alone.f0)] {
            assert!(((got - want) / want).abs() < 1e-3, "{f:?} vs {alone:?}");
        }
        assert!((f.tc - alone.tc).abs() < 0.01, "{f:?} vs {alone:?}");
        assert_eq!((ev[0].ridges, ev[0].harmonic), (1, 1));
    }

    #[test]
    fn steady_and_rising_tones_make_no_event() {
        let geom = [86.0, 907.0, 32.5];
        let ev = run(vec![
            hum(120.0, (0.0, 60.0), 1),
            hum(240.0, (0.0, 60.0), 2),
            rising(geom, 108.0, (1.0, 25.0), 0.3, 3),
            rising(geom, 84.98, (26.0, 38.0), 0.3, 4),
            rising(geom, 169.9, (36.0, 59.0), 0.4, 5),
        ]);
        assert!(ev.is_empty(), "{:?}", brief(&ev));
    }

    /// Engines revving, music, sirens: tones that fall about as often and as much as they
    /// rise. Each falling piece alone would fit some pass, but the falls near each other share
    /// no curve and none finds a second piece on its own.
    #[test]
    fn clutter_that_rises_as_it_falls_makes_no_event() {
        let mut fr = Vec::new();
        for i in 0..6u32 {
            let t = 20.0 + 25.0 * f64::from(i);
            let g = [60.0 + 15.0 * f64::from(i), 500.0 + 150.0 * f64::from(i), t];
            fr.push(arm(
                g,
                150.0 + 40.0 * f64::from(i),
                (t - 8.0, t + 8.0),
                0.3,
                10 + u64::from(i),
            ));
            let u = t + 9.0;
            let h = [g[0], g[1], u];
            fr.push(rising(
                h,
                170.0 + 40.0 * f64::from(i),
                (u - 8.0, u + 8.0),
                0.3,
                30 + u64::from(i),
            ));
        }
        let ev = run(fr.clone());
        assert!(ev.is_empty(), "{:?}\n{}", brief(&ev), story(fr));
    }

    #[test]
    fn pass_among_clutter_survives_the_null() {
        let geom = [86.0, 907.0, 82.5];
        let mut fr = faded_pass(geom);
        let g = [70.0, 600.0, 70.0];
        fr.push(arm(g, 310.0, (66.0, 74.0), 0.3, 21));
        fr.push(rising([70.0, 600.0, 95.0], 330.0, (91.0, 99.0), 0.3, 22));
        let ev = run(fr);
        assert_eq!(ev.len(), 1, "{:?}", brief(&ev));
        assert!(close(&ev[0], geom), "{:?}", ev[0].fit);
    }

    #[test]
    fn two_passes_at_once_are_two_events() {
        let (a, b) = ([86.0, 907.0, 40.0], [150.0, 2500.0, 55.0]);
        let fr = vec![
            arm(a, 108.0, (5.0, 33.0), 0.3, 1),
            arm(a, 84.98, (34.0, 46.0), 0.3, 2),
            arm(a, 169.9, (44.0, 70.0), 0.4, 3),
            arm(b, 131.0, (10.0, 50.0), 0.3, 4),
            arm(b, 262.3, (52.0, 62.0), 0.4, 5),
            arm(b, 131.0, (64.0, 100.0), 0.3, 6),
        ];
        let ev = run(fr.clone());
        assert_eq!(ev.len(), 2, "{:?}\n{}", brief(&ev), story(fr));
        assert!(close(&ev[0], a), "{:?}", ev[0].fit);
        assert!(close(&ev[1], b), "{:?}", ev[1].fit);
        assert_eq!((ev[0].ridges, ev[1].ridges), (3, 3));
    }

    /// Short pieces a minute apart, each at its own pitch: a slow, far curve runs near all of
    /// them, and nothing else ties them together.
    #[test]
    fn pieces_a_minute_apart_are_not_joined() {
        let g = [126.0, 5085.0, 437.0];
        let fr = vec![
            arm(g, 564.7, (331.8, 341.8), 0.5, 1),
            arm(g, 494.2, (332.2, 338.0), 0.5, 2),
            arm(g, 776.6, (332.5, 340.0), 0.6, 3),
            arm(g, 78.8, (391.2, 399.5), 0.15, 4),
            arm(g, 152.6, (456.5, 461.2), 0.2, 5),
            arm(g, 152.3, (510.2, 516.5), 0.2, 6),
            arm(g, 1143.7, (511.2, 521.5), 0.8, 7),
        ];
        let ev = run(fr.clone());
        assert!(
            ev.iter().all(|e| e.t_end - e.t_start < 30.0),
            "{:?}\n{}",
            brief(&ev),
            story(fr)
        );
    }

    /// Three numbers (v, d, tc) can match the slopes of any three straight pieces, each at its
    /// own pitch: a chain of three says nothing about a pass, however far the pieces fall.
    #[test]
    fn three_straight_pieces_make_no_event() {
        let g = [77.0, 724.0, 30.0];
        let fr = vec![
            chord(g, 55.6, (10.0, 21.5), 0.15, 1),
            chord(g, 92.6, (20.0, 25.5), 0.2, 2),
            chord(g, 66.7, (27.5, 33.5), 0.15, 3),
        ];
        let ev = run(fr.clone());
        assert!(ev.is_empty(), "{:?}\n{}", brief(&ev), story(fr));
    }

    /// Speech and fades cut a pass into short pieces, none long enough to show its bend. Past
    /// the third, every piece whose slope the curve foretells is evidence.
    #[test]
    fn short_straight_pieces_on_one_curve_are_a_pass() {
        let g = [86.0, 907.0, 40.0];
        let fr = vec![
            chord(g, 108.0, (10.0, 18.5), 0.15, 1),
            chord(g, 215.0, (17.0, 22.0), 0.2, 2),
            chord(g, 84.98, (24.0, 29.0), 0.15, 3),
            chord(g, 169.9, (31.0, 36.0), 0.2, 4),
            chord(g, 254.9, (32.0, 37.0), 0.25, 5),
            chord(g, 131.0, (38.0, 43.0), 0.15, 6),
            chord(g, 262.3, (39.0, 44.0), 0.25, 7),
            chord(g, 84.98, (45.0, 50.0), 0.15, 8),
            chord(g, 108.0, (52.0, 60.0), 0.15, 9),
        ];
        let ev = run(fr.clone());
        assert_eq!(ev.len(), 1, "{:?}\n{}", brief(&ev), story(fr));
        let f = &ev[0].fit;
        assert!(
            (f.tc - g[2]).abs() < 2.0 && ev[0].ridges >= 8,
            "{f:?} {}",
            ev[0].ridges
        );
    }

    /// Three partials heard in short stretches around a pass at 30 s: the linker joins a
    /// partial's stretches across the silences between them.
    const TALK: [f64; 3] = [60.0, 388.0, 30.0];

    fn syllables(steady: bool) -> Vec<Ridge> {
        let high: Vec<(f64, f64)> = (0..11u32)
            .map(|i| 36.0 + 2.5 * f64::from(i))
            .map(|t| (t, t + 1.0))
            .collect();
        vec![
            broken(
                TALK,
                80.8,
                &[(24.75, 25.25), (26.5, 27.0), (28.75, 29.0)],
                steady,
                0.2,
                1,
            ),
            broken(
                TALK,
                127.3,
                &[(34.0, 34.75), (36.75, 37.75)],
                steady,
                0.2,
                2,
            ),
            broken(TALK, 1399.0, &high, steady, 1.0, 3),
        ]
    }

    /// A voice: a harmonic held steady, a silence, the next steady peak a little lower. Joined,
    /// the jumps between the pieces trace a pass's fall, and within each piece nothing moves.
    #[test]
    fn steady_pieces_joined_over_silences_make_no_event() {
        let ev = run(syllables(true));
        let log = story(syllables(true));
        assert!(ev.is_empty(), "{:?}\n{log}", brief(&ev));
        assert!(
            log.contains(": only the jumps between pieces follow the curve; vote"),
            "{log}"
        );
    }

    /// The same stretches of a real pass, heard between the words of someone talking over it:
    /// the tones move within each stretch as the curve says.
    #[test]
    fn a_pass_heard_in_stretches_is_an_event() {
        let ev = run(syllables(false));
        assert_eq!(ev.len(), 1, "{:?}\n{}", brief(&ev), story(syllables(false)));
        assert!((ev[0].fit.tc - TALK[2]).abs() < 2.0, "{:?}", ev[0].fit);
    }

    /// A slow, far pass heard on one partial bends little. Against the noise a track is
    /// allowed the bend is slight; against the scatter this track has, it is plain.
    #[test]
    fn slight_bend_on_a_clean_track_is_a_pass() {
        let g = [40.0, 3060.0, 64.0];
        let fr = vec![arm(g, 87.0, (1.0, 119.0), 0.06, 1)];
        let ev = run(fr.clone());
        assert_eq!(ev.len(), 1, "{:?}\n{}", brief(&ev), story(fr));
        assert!((ev[0].fit.tc - g[2]).abs() < 2.0, "{:?}", ev[0].fit);
    }

    /// When the sound of the slowing aircraft's closest approach arrives.
    const SLOWS_AT: f64 = 73.0;

    /// One partial of an aircraft that slows and descends as it passes 1500 m away: 137 m/s
    /// when first heard, 76 m/s when last. Each sound arrives late by its way to the listener.
    fn slowing(f0: f64, span: (f64, f64), sigma: f64, seed: u64) -> Ridge {
        let (v0, a, sink, side, up) = (95.0, -0.6, 6.0, 900.0, 1200.0);
        // Where it is, and how fast it closes in, `u` seconds after its closest approach.
        let at = |u: f64| {
            let x: f64 = v0 * u + 0.5 * a * u * u;
            let h = up - sink * u;
            let r = (x * x + side * side + h * h).sqrt();
            (r, (x * (v0 + a * u) - h * sink) / r)
        };
        let closest = SLOWS_AT - at(0.0).0 / C;
        let mut n = noise(seed);
        let steps = ((span.1 - span.0) / 0.25) as u32;
        Ridge {
            points: (0..=steps)
                .map(|i| {
                    let t = span.0 + f64::from(i) * 0.25;
                    // When the sound heard at `t` was made.
                    let mut u = t - closest;
                    for _ in 0..40 {
                        u = t - closest - at(u).0 / C;
                    }
                    RidgePoint {
                        t,
                        f: f0 / (1.0 + at(u).1 / C) + sigma * n(),
                        snr_db: 20.0,
                    }
                })
                .collect(),
        }
    }

    /// No constant-velocity curve runs along the whole of a pass that slows, and its long
    /// tracks sit on none. Its shorter pieces each lie on one: the pass is found on those.
    #[test]
    fn a_pass_that_slows_is_found_on_its_short_pieces() {
        let fr = vec![
            slowing(103.0, (0.0, 104.5), 0.15, 1),
            slowing(206.0, (34.75, 49.0), 0.25, 2),
            slowing(206.0, (49.0, 58.25), 0.25, 3),
            slowing(206.0, (58.0, 75.25), 0.25, 4),
            slowing(206.0, (71.0, 104.5), 0.25, 5),
        ];
        let ev = run(fr.clone());
        let log = story(fr);
        assert!(
            ev.iter().any(|e| (e.fit.tc - SLOWS_AT).abs() < 3.0),
            "{:?}\n{log}",
            brief(&ev)
        );
        // The log says which fits grew from the seed alone.
        assert!(
            log.contains(" fragments; grown from the seed alone; fit v "),
            "{log}"
        );
    }

    /// 30 m/s at 100 m: the whole sweep in some ten seconds.
    const CAR: [f64; 3] = [30.0, 100.0, 40.0];

    #[test]
    fn road_vehicle_heard_whole_starts_no_pass() {
        let fr = vec![arm(CAR, 95.0, (20.0, 60.0), 0.2, 1)];
        let ev = run(fr.clone());
        assert!(ev.is_empty(), "{:?}", brief(&ev));
        let log = story(fr);
        assert!(
            log.contains("a road vehicle's sweep on its own, no seed"),
            "{log}"
        );
    }

    #[test]
    fn road_vehicle_in_pieces_makes_no_event_and_is_spent() {
        // Approach on one partial, departure on another: neither alone is a vehicle's sweep,
        // together they are. The half that did not seed must not start a search of its own.
        let fr = vec![
            arm(CAR, 95.0, (22.0, 39.0), 0.2, 1),
            arm(CAR, 190.0, (41.0, 58.0), 0.3, 2),
        ];
        let ev = run(fr.clone());
        assert!(ev.is_empty(), "{:?}", brief(&ev));
        let log = story(fr);
        let seeds: Vec<&str> = log.lines().filter(|l| l.starts_with("seed ")).collect();
        assert_eq!(seeds.len(), 1, "{log}");
        assert!(seeds[0].contains(": a road vehicle; vote"), "{log}");
    }

    #[test]
    fn windows_restrict_the_fragments() {
        let geom = [86.0, 907.0, 32.5];
        let p = EventParams::default();
        assert_eq!(
            events_from_fragments(faded_pass(geom), &p, &[(0.0, 60.0)]).len(),
            1
        );
        assert!(events_from_fragments(faded_pass(geom), &p, &[(200.0, 260.0)]).is_empty());
        // Windows that cut the pass: what is found is made of the points inside them.
        let cut = events_from_fragments(faded_pass(geom), &p, &[(0.0, 24.0), (36.0, 60.0)]);
        assert_eq!(cut.len(), 1, "{:?}", brief(&cut));
        let whole = faded_pass(geom)
            .iter()
            .map(|r| r.points.len())
            .sum::<usize>();
        assert!(
            cut[0].points.len() > whole / 3,
            "{} of {whole}",
            cut[0].points.len()
        );
        assert!(
            cut[0].points.iter().all(|&(t, _)| t <= 24.0 || t >= 36.0),
            "{:?}",
            cut[0].points
        );
    }

    #[test]
    fn the_log_gives_each_seed_its_verdict_first() {
        // A terminal cuts long lines: the verdict comes before the numbers, tracks follow on
        // lines of their own, and only the lines about the asked stretch are returned.
        let p = EventParams::default();
        let log = explain_fragments(faded_pass([86.0, 907.0, 32.5]), &p, &[], (0.0, 60.0));
        let seeds: Vec<&String> = log.iter().filter(|l| l.starts_with("seed ")).collect();
        assert!(
            seeds.iter().any(|l| l.contains(" s: pass; vote v ")),
            "{log:#?}"
        );
        assert!(
            log.iter().filter(|l| l.starts_with("    track ")).count() >= 5,
            "{log:#?}"
        );
        assert!(
            log.iter()
                .any(|l| l.starts_with("pass tc 32.") && l.ends_with(": event")),
            "{log:#?}"
        );
        assert!(log.iter().all(|l| l.chars().count() <= 200), "{log:#?}");
        let elsewhere = explain_fragments(faded_pass([86.0, 907.0, 32.5]), &p, &[], (300.0, 400.0));
        assert!(elsewhere.is_empty(), "{elsewhere:#?}");
    }
}
