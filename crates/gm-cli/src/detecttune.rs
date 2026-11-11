//! The stage-1 tuning sweep (`eval --detect-tune`): each grid point judged against two fixed
//! reference rows, `original` (the PR #24 detector, every street-sound cue off) and `current`
//! (today's defaults), with the keep rule of
//! `docs/superpowers/specs/2026-11-09-street-sounds-design.md` (decision 2).

use std::fmt::Write as _;

use gm_doppler::audio::DetectParams;
use serde::Serialize;

use crate::detecteval::{DetectScore, RowCounts};

/// One street recording's figures (cases whose id starts with `street-`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StreetTotals {
    pub id: String,
    /// False passes in the PR #24 sense (no aircraft row under them).
    pub fp: usize,
    /// Those with a not-aircraft row under them.
    pub fp_strict: usize,
    /// Those on ignore rows only.
    pub on_ignored: usize,
    pub runs_found: usize,
    pub runs: usize,
}

/// One grid point's figures on a split.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TuneTotals {
    /// AeroSonicDB labelled hours: 5-s row F1, labelled runs, false passes (PR #24 and strict),
    /// silent seconds.
    pub aero_f1: f64,
    pub aero_runs_found: usize,
    pub aero_runs: usize,
    pub aero_fp: usize,
    pub aero_fp_strict: usize,
    pub aero_silence_s: f64,
    /// AeroSonicDB clips: a pass over the audible window.
    pub clip_hits: usize,
    pub clips: usize,
    pub street: Vec<StreetTotals>,
}

impl TuneTotals {
    /// Totals of one grid point: per case, its id, its score and whether it is labelled in
    /// 5-s rows (a continuous recording) rather than scored by its audible window (a clip).
    pub fn of(cases: &[(&str, &DetectScore, bool)]) -> Self {
        let mut t = Self::default();
        let mut rc = RowCounts::default();
        for &(id, s, labelled) in cases {
            if !labelled {
                if let Some(h) = s.clip_hit {
                    t.clips += 1;
                    t.clip_hits += usize::from(h);
                }
                continue;
            }
            let strict = s.false_passes - s.passes_on_ignored;
            if id.starts_with("street-") {
                t.street.push(StreetTotals {
                    id: id.to_string(),
                    fp: s.false_passes,
                    fp_strict: strict,
                    on_ignored: s.passes_on_ignored,
                    runs_found: s.runs_found,
                    runs: s.runs,
                });
                continue;
            }
            if let Some(r) = s.rows {
                rc.tp += r.tp;
                rc.fp += r.fp;
                rc.fn_ += r.fn_;
                rc.tn += r.tn;
            }
            t.aero_fp += s.false_passes;
            t.aero_fp_strict += strict;
            t.aero_silence_s += s.silence_s;
            t.aero_runs += s.runs;
            t.aero_runs_found += s.runs_found;
        }
        let prec = rc.tp as f64 / (rc.tp + rc.fp).max(1) as f64;
        let rec = rc.tp as f64 / (rc.tp + rc.fn_).max(1) as f64;
        t.aero_f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };
        t
    }

    /// Strict false passes on the split: AeroSonicDB hours plus every street recording.
    pub fn strict_fp(&self) -> usize {
        self.aero_fp_strict + self.street.iter().map(|s| s.fp_strict).sum::<usize>()
    }

    pub fn street(&self, id: &str) -> Option<&StreetTotals> {
        self.street.iter().find(|s| s.id == id)
    }

    /// AeroSonicDB false passes per silent hour, PR #24 measure.
    pub fn aero_fp_per_h(&self) -> f64 {
        if self.aero_silence_s > 0.0 {
            self.aero_fp as f64 / (self.aero_silence_s / 3600.0)
        } else {
            0.0
        }
    }
}

/// The grid: `original` (fixed), `current` (today's defaults), then each cue's variants built
/// on `current`.
pub fn tune_grid(current: &DetectParams) -> Vec<(String, DetectParams)> {
    let mut g = vec![
        ("original".into(), DetectParams::legacy()),
        ("current".into(), current.clone()),
    ];
    // Cues 1 and 2 are decided and live in `current` (sweeps of 2026-11-09: the 0.3 quantile
    // of 0.1-s sub-blocks; the running 0.4 quantile). Cue 3 (absorption) was not kept: the only
    // thresholds that removed a street false pass dropped close-airport clip hits. Cue 4 (reject
    // candidates covered by rising tones, prepare-time) was not kept either: it removed no street
    // false pass and cost 3 AeroSonicDB runs and 7 clip hits (take-offs: engines spool up).
    g.push((
        "cue4 engine".into(),
        DetectParams {
            reject_engine: true,
            ..current.clone()
        },
    ));
    g
}

/// The keep rule: fewer strict false passes than `current`; AeroSonicDB runs within one of
/// `original`; AeroSonicDB strict false passes no more than `original`'s; the held-out phone
/// clip's aircraft run still found (when the split has it).
pub fn keeps(original: &TuneTotals, current: &TuneTotals, row: &TuneTotals) -> bool {
    let phone = match original.street("street-phone") {
        Some(_) => row
            .street("street-phone")
            .is_some_and(|s| s.runs_found >= 1),
        None => true,
    };
    row.strict_fp() < current.strict_fp()
        && row.aero_runs_found + 1 >= original.aero_runs_found
        && row.aero_fp_strict <= original.aero_fp_strict
        && phone
}

/// What `prepare()` depends on: grid points with equal keys share one preparation.
pub fn prepare_key(p: &DetectParams) -> String {
    let level = match p.sub_quantile {
        Some(q) => format!("q{q} block{}", p.sub_block_s),
        None => "mean".into(),
    };
    format!(
        "frame {} hop {} floor {}/{} level {level} engine {}",
        p.frame_s, p.hop_s, p.floor_pct, p.floor_half_s, p.reject_engine
    )
}

/// `hf_db` per pass class (AeroSonicDB vs street) for the absorption threshold: count with and
/// without a value, then min, median, 95th and 99th percentile (nearest rank) and max; and every
/// street pass listed with its time, class and value. Samples: (case id, class, hf_db, start).
pub fn hf_summary(samples: &[(String, crate::detecteval::PassClass, Option<f64>, f64)]) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "\n| group | n | no value | min | p50 | p95 | p99 | max |\n|---|---|---|---|---|---|---|---|"
    );
    let mut groups: Vec<(String, crate::detecteval::PassClass)> = samples
        .iter()
        .map(|x| {
            let side = if x.0.starts_with("street-") {
                "street"
            } else {
                "aero"
            };
            (side.to_string(), x.1)
        })
        .collect();
    groups.sort();
    groups.dedup();
    for (side, class) in &groups {
        let in_group = |x: &&(String, crate::detecteval::PassClass, Option<f64>, f64)| {
            x.1 == *class && x.0.starts_with("street-") == (side == "street")
        };
        let mut v: Vec<f64> = samples
            .iter()
            .filter(in_group)
            .filter_map(|x| x.2)
            .collect();
        let none = samples
            .iter()
            .filter(in_group)
            .filter(|x| x.2.is_none())
            .count();
        v.sort_by(f64::total_cmp);
        let q = |p: f64| -> String {
            if v.is_empty() {
                "—".into()
            } else {
                format!("{:.1}", v[(p * (v.len() - 1) as f64).round() as usize])
            }
        };
        let _ = writeln!(
            s,
            "| {side} {class:?} | {} | {none} | {} | {} | {} | {} | {} |",
            v.len(),
            q(0.0),
            q(0.5),
            q(0.95),
            q(0.99),
            q(1.0)
        );
    }
    let _ = writeln!(
        s,
        "\n| street case | start s | class | hf_db |\n|---|---|---|---|"
    );
    for x in samples.iter().filter(|x| x.0.starts_with("street-")) {
        let v = x.2.map_or("—".to_string(), |v| format!("{v:.1}"));
        let _ = writeln!(s, "| {} | {:.1} | {:?} | {v} |", x.0, x.3, x.1);
    }
    s
}

/// The sweep table: one row per grid point, in grid order.
pub fn table(rows: &[(String, TuneTotals, bool)]) -> String {
    let mut s = String::new();
    let ids: Vec<String> = rows
        .first()
        .map(|r| r.1.street.iter().map(|x| x.id.clone()).collect())
        .unwrap_or_default();
    let _ = write!(
        s,
        "| row | keeps | aero F1 | aero runs | aero FP (PR #24) | per silent h | aero FP strict | clips |"
    );
    for id in &ids {
        let _ = write!(s, " {id}: strict/all (on ignored), runs |");
    }
    let _ = write!(s, " strict total |\n|---|---|---|---|---|---|---|---|");
    for _ in &ids {
        let _ = write!(s, "---|");
    }
    let _ = writeln!(s, "---|");
    for (name, t, keep) in rows {
        let _ = write!(
            s,
            "| {name} | {} | {:.3} | {}/{} | {} | {:.2} | {} | {}/{} |",
            if *keep { "yes" } else { "" },
            t.aero_f1,
            t.aero_runs_found,
            t.aero_runs,
            t.aero_fp,
            t.aero_fp_per_h(),
            t.aero_fp_strict,
            t.clip_hits,
            t.clips
        );
        for id in &ids {
            match t.street(id) {
                Some(x) => {
                    let _ = write!(
                        s,
                        " {}/{} ({}), {}/{} |",
                        x.fp_strict, x.fp, x.on_ignored, x.runs_found, x.runs
                    );
                }
                None => {
                    let _ = write!(s, " — |");
                }
            }
        }
        let _ = writeln!(s, " {} |", t.strict_fp());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn street(id: &str, fp_strict: usize, runs_found: usize, runs: usize) -> StreetTotals {
        StreetTotals {
            id: id.into(),
            fp: fp_strict,
            fp_strict,
            on_ignored: 0,
            runs_found,
            runs,
        }
    }

    /// 26/33 AeroSonicDB runs, 2 strict AeroSonicDB false passes, the phone clip 3 strict false passes
    /// with its aircraft run found, Piccadilly 2.
    fn totals(aero_runs_found: usize, aero_fp_strict: usize, phone: (usize, usize)) -> TuneTotals {
        TuneTotals {
            aero_f1: 0.845,
            aero_runs_found,
            aero_runs: 33,
            aero_fp: aero_fp_strict,
            aero_fp_strict,
            aero_silence_s: 3.0 * 3600.0,
            clip_hits: 100,
            clips: 120,
            street: vec![
                street("street-phone", phone.0, phone.1, 1),
                street("street-piccadilly", 2, 0, 0),
            ],
        }
    }

    #[test]
    fn keeps_against_original_and_current() {
        let original = totals(26, 2, (3, 1));
        let current = original.clone();
        // Equal strict false passes: nothing gained.
        assert!(!keeps(&original, &current, &totals(26, 2, (3, 1))));
        // One fewer, one run lost (26 → 25): kept.
        assert!(keeps(&original, &current, &totals(25, 2, (2, 1))));
        // One fewer, two runs lost (26 → 24): not kept.
        assert!(!keeps(&original, &current, &totals(24, 2, (2, 1))));
        // Three fewer on the street, two more at the airport: total falls, but the AeroSonicDB
        // false passes rose above the original's.
        assert!(!keeps(&original, &current, &totals(26, 4, (0, 1))));
        // The phone clip's aircraft run lost.
        assert!(!keeps(&original, &current, &totals(26, 2, (2, 0))));
        // `current` already lost one run: a row at 25 is still within one of the original.
        let current = totals(25, 2, (2, 1));
        assert!(keeps(&original, &current, &totals(25, 2, (1, 1))));
    }

    #[test]
    fn grid_starts_with_original_then_current() {
        let d = DetectParams {
            threshold_db: 7.5,
            ..DetectParams::default()
        };
        let g = tune_grid(&d);
        assert_eq!(g[0].0, "original");
        assert_eq!(
            serde_json::to_value(&g[0].1).unwrap(),
            serde_json::to_value(DetectParams::legacy()).unwrap()
        );
        assert_eq!(g[1].0, "current");
        assert_eq!(
            serde_json::to_value(&g[1].1).unwrap(),
            serde_json::to_value(&d).unwrap()
        );
    }

    #[test]
    fn grid_adds_the_engine_check_with_its_own_preparation() {
        let current = DetectParams::default();
        let g = tune_grid(&current);
        let cue4: Vec<_> = g.iter().filter(|(n, _)| n.starts_with("cue4")).collect();
        assert_eq!(cue4.len(), 1);
        assert!(cue4[0].1.reject_engine);
        assert_ne!(prepare_key(&cue4[0].1), prepare_key(&current));
        assert!(!g.iter().any(|(n, _)| n.starts_with("cue3")));
    }

    #[test]
    fn hf_summary_gives_quantiles_per_class_and_lists_street_passes() {
        use crate::detecteval::PassClass::*;
        let s = |id: &str, c, v: Option<f64>, t| (id.to_string(), c, v, t);
        let samples = vec![
            s("env-1_AUDIO", Aircraft, Some(-20.0), 10.0),
            s("env-1_AUDIO", Aircraft, Some(-10.0), 300.0),
            s("env-2_AUDIO", Aircraft, Some(-15.0), 50.0),
            s("clip-a", ClipHit, Some(-12.0), 3.0),
            s("clip-b", ClipHit, None, 3.0),
            s("street-phone", StrictFalse, Some(-2.0), 124.0),
        ];
        let out = hf_summary(&samples);
        assert!(
            out.contains("| aero Aircraft | 3 | 0 | -20.0 | -15.0 | -10.0 | -10.0 | -10.0 |"),
            "{out}"
        );
        assert!(out.contains("| aero ClipHit | 1 | 1 |"), "{out}");
        assert!(
            out.contains("| street-phone | 124.0 | StrictFalse | -2.0 |"),
            "{out}"
        );
    }

    #[test]
    fn prepare_key_ignores_judge_only_fields() {
        let d = DetectParams::default();
        let judged = DetectParams {
            threshold_db: d.threshold_db + 1.0,
            min_pass_s: d.min_pass_s + 4.0,
            max_jitter_db: d.max_jitter_db + 1.0,
            ..d.clone()
        };
        assert_eq!(prepare_key(&d), prepare_key(&judged));
        let floor = DetectParams {
            floor_pct: d.floor_pct + 5.0,
            ..d.clone()
        };
        assert_ne!(prepare_key(&d), prepare_key(&floor));
    }
}
