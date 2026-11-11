//! The pass log: every aircraft sound stage 1 found, its curve result and what location used it
//! for, plus every rejected candidate — nothing is dropped without a reason.

use serde::{Deserialize, Serialize};

use super::detect::{Detection, Pass, Rejected};
use super::events::Event;

/// Stage 2's result for one pass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Curve {
    /// Tonal Doppler events (ids from `discover`) on this pass.
    Tonal {
        event_ids: Vec<usize>,
    },
    None {
        reason: String,
    },
}

/// What the location step used the pass for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsedFor {
    Curve,
    TimingLoudness,
    Unused { reason: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PassRecord {
    pub pass: Pass,
    pub curve: Curve,
    pub used_for: UsedFor,
}

/// A tonal curve with no loudness pass under it: an aircraft sound of its own (loudness and
/// tones are complementary; neither gates the other).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToneOnly {
    pub event_id: usize,
    pub t_start: f64,
    pub t_end: f64,
    /// The reason of a rejected candidate under the curve: the conflict, shown, not resolved.
    pub on_rejected: Option<String>,
    pub used: UsedFor,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PassLog {
    pub passes: Vec<PassRecord>,
    pub rejected: Vec<Rejected>,
    /// Tonal events that overlap no pass (kept for location, listed here).
    pub events_outside_passes: Vec<usize>,
    /// The same events as aircraft sounds of their own.
    #[serde(default)]
    pub tone_only: Vec<ToneOnly>,
    pub notes: Vec<String>,
}

/// The reason logged for a pass with no tonal event.
pub const NO_TONE: &str =
    "no tone found on this pass (broadband only); broadband slide not attempted (deferred)";

/// For each pass, the events whose span overlaps it most (an event goes to one pass at most);
/// and the events that overlap no pass.
pub fn assign_tonal(passes: &[Pass], events: &[Event]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut by_pass = vec![Vec::new(); passes.len()];
    let mut outside = Vec::new();
    for e in events {
        let best = passes
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (
                    i,
                    (e.t_end.min(p.t_end) - e.t_start.max(p.t_start)).max(0.0),
                )
            })
            .filter(|&(_, o)| o > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((i, _)) => by_pass[i].push(e.id),
            None => outside.push(e.id),
        }
    }
    (by_pass, outside)
}

/// The pass log for `det` and `events`: each pass gets its tonal events or a "no curve" reason;
/// `used_for` starts as "not yet decided" until the location step fills it.
pub fn build(det: &Detection, events: &[Event]) -> PassLog {
    let (by_pass, outside) = assign_tonal(&det.passes, events);
    PassLog {
        passes: det
            .passes
            .iter()
            .zip(by_pass)
            .map(|(p, ids)| PassRecord {
                pass: p.clone(),
                curve: if ids.is_empty() {
                    Curve::None {
                        reason: NO_TONE.into(),
                    }
                } else {
                    Curve::Tonal { event_ids: ids }
                },
                used_for: UsedFor::Unused {
                    reason: "location not run".into(),
                },
            })
            .collect(),
        rejected: det.rejected.clone(),
        tone_only: events
            .iter()
            .filter(|e| outside.contains(&e.id))
            .map(|e| ToneOnly {
                event_id: e.id,
                t_start: e.t_start,
                t_end: e.t_end,
                // The rejected candidate overlapping the curve most.
                on_rejected: det
                    .rejected
                    .iter()
                    .map(|r| (r, e.t_end.min(r.t_end) - e.t_start.max(r.t_start)))
                    .filter(|&(_, o)| o > 0.0)
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(r, _)| r.reason.clone()),
                used: UsedFor::Unused {
                    reason: "location not run".into(),
                },
            })
            .collect(),
        events_outside_passes: outside,
        notes: det.notes.clone(),
    }
}

/// Only the events that fall on a detected pass (stage-1 gating of tonal events).
pub fn gate_to_passes(events: &[Event], log: &PassLog) -> Vec<Event> {
    events
        .iter()
        .filter(|e| !log.events_outside_passes.contains(&e.id))
        .cloned()
        .collect()
}

/// How a use reads in the log.
fn used_text(u: &UsedFor) -> String {
    match u {
        UsedFor::Curve => "used: its curve".to_string(),
        UsedFor::TimingLoudness => "used: timing and loudness".to_string(),
        UsedFor::Unused { reason } => format!("unused — {reason}"),
    }
}

impl PassLog {
    /// What location does with the tone-only curves: offered to the tonal legs, unless the
    /// opt-in pass gate dropped them.
    pub fn mark_tone_only(&mut self, gated: bool) {
        for t in &mut self.tone_only {
            t.used = if gated {
                UsedFor::Unused {
                    reason: "dropped by --pass-gate".into(),
                }
            } else {
                UsedFor::Curve
            };
        }
    }

    /// One line per pass, per tone-only sound and per rejection, for the terminal.
    pub fn lines(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .passes
            .iter()
            .map(|r| {
                let p = &r.pass;
                let peak = if p.flags.iter().any(|f| f == "cut_at_start") {
                    format!("peak at or before {:.1} s (cut at the start)", p.t_peak)
                } else {
                    format!("peak {:.1} s", p.t_peak)
                };
                let curve = match &r.curve {
                    Curve::Tonal { event_ids } => format!(
                        "curve: tonal (events {})",
                        event_ids
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    Curve::None { reason } => format!("curve: none — {reason}"),
                };
                let used = used_text(&r.used_for);
                format!(
                    "pass {}  {:.1}–{:.1} s  {peak}  +{:.1} dB  {curve}  {used}",
                    p.id, p.t_start, p.t_end, p.excess_db
                )
            })
            .collect();
        out.extend(self.tone_only.iter().map(|t| {
            let what = match &t.on_rejected {
                Some(reason) => format!(
                    "tonal curve (event {}) on a sound rejected as: {reason}",
                    t.event_id
                ),
                None => format!(
                    "tonal curve (event {}), loudness below the pass threshold",
                    t.event_id
                ),
            };
            format!(
                "tone only  {:.1}–{:.1} s  {what};  {}",
                t.t_start,
                t.t_end,
                used_text(&t.used)
            )
        }));
        out.extend(
            self.rejected
                .iter()
                .map(|r| format!("rejected  {:.1}–{:.1} s  {}", r.t_start, r.t_end, r.reason)),
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(id: usize, t0: f64, t1: f64) -> Pass {
        Pass {
            id,
            t_start: t0,
            t_end: t1,
            t_peak: (t0 + t1) / 2.0,
            excess_db: 8.0,
            score: 6.0,
            flags: vec![],
            hf_db: None,
        }
    }

    fn event(id: usize, t0: f64, t1: f64) -> Event {
        serde_json::from_value(serde_json::json!({
            "id": id, "t_start": t0, "t_end": t1,
            "fit": {"v": 60.0, "d": 800.0, "tc": (t0 + t1) / 2.0, "f0": 100.0, "rms_hz": 0.1,
                    "sigma_v": 1.0, "sigma_d": 1.0, "sigma_tc": 0.1, "n": 10},
            "harmonic": 1, "f0_fundamental": 100.0, "ridges": 1, "mean_snr_db": 10.0,
            "points": [[t0, 101.0], [t1, 99.0]],
        }))
        .unwrap()
    }

    #[test]
    fn events_go_to_the_pass_they_overlap_most() {
        let passes = [pass(1, 0.0, 40.0), pass(2, 50.0, 100.0)];
        // Event 7 overlaps pass 1 by 5 s and pass 2 by 30 s.
        let (by_pass, outside) = assign_tonal(&passes, &[event(7, 35.0, 80.0)]);
        assert_eq!(by_pass, vec![vec![], vec![7]]);
        assert!(outside.is_empty());
    }

    #[test]
    fn an_event_outside_every_pass_is_listed() {
        let (by_pass, outside) = assign_tonal(&[pass(1, 0.0, 40.0)], &[event(3, 60.0, 90.0)]);
        assert_eq!(by_pass, vec![Vec::<usize>::new()]);
        assert_eq!(outside, vec![3]);
    }

    #[test]
    fn every_pass_record_has_a_curve_and_a_use() {
        let det = Detection {
            passes: vec![pass(1, 0.0, 40.0), pass(2, 50.0, 100.0)],
            rejected: vec![Rejected {
                t_start: 120.0,
                t_end: 123.0,
                reason: "too short (3.0 s < 20 s)".into(),
            }],
            ..Default::default()
        };
        let log = build(&det, &[event(5, 10.0, 30.0)]);
        assert_eq!(log.passes.len(), 2);
        assert_eq!(log.rejected.len(), 1);
        assert_eq!(log.passes[0].curve, Curve::Tonal { event_ids: vec![5] });
        match &log.passes[1].curve {
            Curve::None { reason } => assert_eq!(reason, NO_TONE),
            c => panic!("{c:?}"),
        }
        for r in &log.passes {
            match &r.used_for {
                UsedFor::Unused { reason } => assert!(!reason.is_empty()),
                u => panic!("before location every pass is Unused with a reason, got {u:?}"),
            }
        }
        // Every line names a pass or a rejection, with its reason.
        let lines = log.lines();
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[2].contains("too short"), "{lines:?}");
    }

    #[test]
    fn tone_only_events_get_their_own_entries() {
        let det = Detection {
            passes: vec![pass(1, 0.0, 40.0)],
            ..Default::default()
        };
        let log = build(&det, &[event(1, 10.0, 30.0), event(2, 60.0, 90.0)]);
        assert_eq!(
            log.tone_only,
            vec![ToneOnly {
                event_id: 2,
                t_start: 60.0,
                t_end: 90.0,
                on_rejected: None,
                used: UsedFor::Unused {
                    reason: "location not run".into()
                },
            }]
        );
        let lines = log.lines();
        assert!(
            lines.iter().any(|l| l.starts_with("tone only  60.0–90.0 s")
                && l.contains("tonal curve (event 2), loudness below the pass threshold")),
            "{lines:?}"
        );
    }

    #[test]
    fn tone_on_a_rejected_sound_shows_the_conflict() {
        let reason = "near sound: keeps high frequencies (1.6–3.2 kHz density -2 dB re 200–800 Hz; a distant aircraft loses them)";
        let det = Detection {
            rejected: vec![Rejected {
                t_start: 55.0,
                t_end: 95.0,
                reason: reason.into(),
            }],
            ..Default::default()
        };
        let ev = [event(4, 60.0, 90.0)];
        let log = build(&det, &ev);
        assert_eq!(log.tone_only[0].on_rejected.as_deref(), Some(reason));
        // Still outside every pass, so it reaches the tonal legs unchanged.
        assert_eq!(log.events_outside_passes, vec![4]);
        assert_eq!(gate_to_passes(&ev, &log).len(), 0);
        let lines = log.lines();
        assert!(
            lines
                .iter()
                .any(|l| l.contains("tonal curve (event 4) on a sound rejected as: near sound")),
            "{lines:?}"
        );
    }

    #[test]
    fn pass_gate_marks_tone_only_entries_dropped() {
        let log = build(&Detection::default(), &[event(2, 60.0, 90.0)]);
        let mut on = log.clone();
        on.mark_tone_only(true);
        assert_eq!(
            on.tone_only[0].used,
            UsedFor::Unused {
                reason: "dropped by --pass-gate".into()
            }
        );
        let mut off = log;
        off.mark_tone_only(false);
        assert_eq!(off.tone_only[0].used, UsedFor::Curve);
        assert!(off.lines().iter().any(|l| l.ends_with("used: its curve")));
    }

    #[test]
    fn gating_keeps_only_events_on_passes() {
        let det = Detection {
            passes: vec![pass(1, 0.0, 40.0)],
            ..Default::default()
        };
        let ev = [event(1, 10.0, 30.0), event(2, 60.0, 90.0)];
        let log = build(&det, &ev);
        let kept: Vec<usize> = gate_to_passes(&ev, &log).iter().map(|e| e.id).collect();
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn pass_log_round_trips_through_json() {
        let det = Detection {
            passes: vec![pass(1, 0.0, 40.0)],
            ..Default::default()
        };
        let log = build(&det, &[]);
        let back: PassLog = serde_json::from_str(&serde_json::to_string(&log).unwrap()).unwrap();
        assert_eq!(back, log);
    }
}
