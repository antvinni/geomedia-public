//! The search's fixed constants, in one place.

/// Still-air sound speed (layered wind is a v2 refinement).
pub const SEARCH_C: f64 = 343.0;
/// Microphone height above ground.
pub const MIC_H: f64 = 1.5;
/// A match must lie within this many σ in slant distance and speed.
pub const GATE_SIGMA: f64 = 4.0;
/// Model-error floors added in quadrature to the fit's σ.
pub const MODEL_TC_S: f64 = 0.5;
pub const MODEL_D_FRAC: f64 = 0.08;
pub const MODEL_V_FRAC: f64 = 0.06;
/// An event no aircraft explains costs this much (so junk events can't veto a cell).
pub const OUTLIER_LL: f64 = -8.0;
/// Each close propeller pass during the recording with no matching event.
pub const SILENCE_LL: f64 = -3.0;
/// Slant within which an unmatched propeller pass counts as "should have been heard".
pub const SILENCE_SLANT_M: f64 = 3000.0;
/// Clock-offset vote bin.
pub const DELTA_BIN_S: f64 = 0.5;
/// Curve-first matching scans δ within this of the CPA-implied δ₀ (the CV fit's tc bias on an
/// accelerating or descending pass), whatever the prior window.
pub const CURVE_TC_BIAS_S: f64 = 15.0;
pub const CURVE_DELTA_STEP_S: f64 = 0.5;
pub const CURVE_FINE_STEP_S: f64 = 0.1;
/// Farthest CPA slant a curve-first match may have.
pub const CURVE_MAX_SLANT_M: f64 = 8000.0;
/// Use every n-th ridge point while scanning (points are ~4× correlated anyway).
pub const CURVE_DECIMATE: usize = 4;
/// A curve-first match's weak tie to the CV fit's CPA time: wide enough for an accelerating or
/// descending pass's tc bias, tight enough to break the along-track tie of a single pass.
pub const CURVE_TC_SIGMA_S: f64 = 5.0;
/// Curve fits use only ridge points within this of the event's CPA time: f₀ is held constant,
/// and propeller RPM changes on approach break that over a long ridge.
pub const CURVE_HALF_WINDOW_S: f64 = 30.0;
/// Most a curve-first match's soft geometry and CPA-time terms may each cost: they break ties
/// near the truth but must never veto an exact curve fit (the CV fit of a long, decelerating
/// pass can be many σ off).
pub const CURVE_SOFT_CAP: f64 = 1.0;
/// A curve-first match's CPA slant and ground speed must agree with the CV fit's d and v within
/// this factor. Wide enough for a decelerating pass's biased fit (Oyster Bay B350: 2.5× in d,
/// 1.4× in v), narrow enough to keep distant aircraft off close road-vehicle ridges (≥ 4×).
pub const CURVE_GEOM_RATIO: f64 = 3.0;
/// Guided curve scoring (`guided.rs`): f₀ range swept, the span around each predicted CPA
/// arrival, the coarse cell grid and its farthest CPA slant, the clock-offset step, the
/// acceptance thresholds (robust z of the curve sum over f₀; fraction of frames ≥ GUIDE_MIN_LEVEL
/// along it), and the measured points a guided event needs. The guide image is standardised
/// per frequency, so GUIDE_MIN_LEVEL is in σ of the noise.
pub const GUIDE_F0: (f64, f64) = (40.0, 400.0);
pub const GUIDE_SPAN_S: f64 = 60.0;
pub const GUIDE_CELL_M: f64 = 2000.0;
pub const GUIDE_MAX_SLANT_M: f64 = 10_000.0;
pub const GUIDE_DELTA_STEP_S: f64 = 0.5;
pub const GUIDE_Z_MIN: f64 = 8.0;
pub const GUIDE_COVERAGE_MIN: f64 = 0.4;
pub const GUIDE_MIN_POINTS: usize = 40;
pub const GUIDE_TOL: f64 = 0.004;
pub const GUIDE_MIN_LEVEL: f32 = 1.0;

/// Each frame's contribution to a curve sum is clipped to ±this (σ): consistency along the
/// curve counts, a few frames on someone else's bright ridge don't.
pub const GUIDE_CLIP: f32 = 3.0;

/// A guided curve must beat the same curve run backwards in time (an impossible rising sweep,
/// same frames) by this much in z (statistics analysis on Oyster Bay's null behaviour: clutter
/// +1.2…+1.7, real passes +3.4…+17).
pub const GUIDE_RISE_MARGIN: f64 = 3.0;

/// The time-shifted null (`guided.rs::null_z`): shifts of the δ sweep, at least this far from
/// the prior, this far apart.
pub const GUIDE_NULL_MIN_SHIFT_S: f64 = 60.0;
pub const GUIDE_NULL_STEP_S: f64 = 20.0;
/// The null needs at least this many shifts that fit the recording (else: no guidance), and
/// uses at most this many.
pub const GUIDE_NULL_MIN_SHIFTS: usize = 8;
pub const GUIDE_NULL_MAX_SHIFTS: usize = 24;
/// Fraction of measured points within half a log bin of the curve (noise: ≈ 1/3 ± 0.03;
/// suite/quiet's faint pass 0.43, a loud pass 0.66). A weak, guide-independent check: the
/// z-over-null gate is the main one.
pub const GUIDE_CENTRED_MIN: f64 = 0.4;
/// A guided curve's z must beat its null by this much (suite/quiet's far pass: +4.3; Oyster
/// Bay's candidates, real street audio: ≤ −0.1 apart from one riding a discovered ridge).
pub const GUIDE_NULL_MARGIN: f64 = 2.0;
/// A guided event with this fraction of its points on an existing event's ridge rides it.
pub const GUIDE_RIDE_FRAC: f64 = 0.2;

/// Decoy aircraft for the guided null (`guided.rs::decoy_shifts`) come from this far either
/// side of the ADS-B window, on the day files already loaded: other traffic of the same day.
pub const GUIDE_DECOY_WINDOW_S: f64 = 10_800.0;

/// Jet tones (guided `Band::Jet`): airframe/engine tones at f₀ 300–3000 Hz, searched only within
/// 3 km slant (fan and high tones are absorbed beyond — B §3: ~29 dB/km at 4 kHz).
pub const JET_F0: (f64, f64) = (300.0, 3000.0);
pub const JET_MAX_SLANT_M: f64 = 3000.0;
pub const JET_CELL_M: f64 = 1000.0;

/// Refine-only guidance (`guided::refine_events`): around a location already found, δ within
/// ±REFINE_DELTA_S of its clock and cells within ±REFINE_RADIUS_M at REFINE_STEP_M; its own z
/// floor (the null, not the floor, is the gate).
pub const REFINE_DELTA_S: f64 = 1.0;
pub const REFINE_RADIUS_M: f64 = 500.0;
pub const REFINE_STEP_M: f64 = 125.0;
pub const REFINE_Z_MIN: f64 = 6.0;

/// Prior σ of the fitted horizontal wind offset with no weather data (winds aloft are rarely
/// beyond 20 m/s at light-aircraft heights).
pub const WIND_SIGMA_FREE: f64 = 8.0;
/// …and as a correction to a supplied weather column (forecast wind error).
pub const WIND_SIGMA_FORECAST: f64 = 3.0;

// Track-first scoring (docs/superpowers/specs/2026-11-06-track-first-design.md).
/// Coarse cell grid for both bands: J needs one δ shared by every aircraft.
pub const TF_CELL_M: f64 = 1000.0;
/// Coarse image: 1 frame/s…
pub const TF_COARSE_HOP_S: f64 = 1.0;
/// …and 48 log bins per octave.
pub const TF_COARSE_BPO: f64 = 48.0;
/// δ step of the per-pair maps.
pub const TF_DELTA_STEP_S: f64 = 1.0;
/// Candidate cells kept from the joint score (best δ each).
pub const TF_TOP_K: usize = 20;
/// Null replicates (each aircraft independently time-shifted).
pub const TF_REPLICATES: usize = 200;
/// Null shifts lie within ± this (keeps the local traffic density).
pub const TF_SHIFT_WINDOW_S: f64 = GUIDE_DECOY_WINDOW_S;
/// The generalized Pareto tail is fitted above this quantile of the replicate maxima.
pub const TF_TAIL_QUANTILE: f64 = 0.8;
/// Accept: global p at most this…
pub const TF_P_ACCEPT: f64 = 0.01;
/// …and still at most this with the strongest aircraft dropped.
pub const TF_P_DROP: f64 = 0.05;
/// Two aircraft have different geometry when their CPA bearings from the cell differ by at
/// least this (mod 180°).
pub const TF_BEARING_MIN_DEG: f64 = 30.0;
/// Explain-away: curves within this many coarse bins…
pub const TF_EXPLAIN_BINS: i32 = 2;
/// …over at least this fraction of the shorter curve's frames are one ridge.
pub const TF_EXPLAIN_FRAC: f64 = 0.5;
/// Fine stage: cell step, radius around the candidate, and δ half-range.
pub const TF_FINE_STEP_M: f64 = 250.0;
pub const TF_FINE_RADIUS_M: f64 = 3000.0;
pub const TF_FINE_DELTA_S: f64 = 2.0;
/// Viterbi f₀ path: at most this many 192-per-octave bins per second…
pub const TF_VITERBI_STEP_BINS_PER_S: f64 = 1.0;
/// …and at most this fractional drift from the starting f₀.
pub const TF_VITERBI_DRIFT: f64 = 0.08;
/// Per-aircraft floor (props and rotors): a pair contributes only its pooled z above this.
/// Measured on silent simulator renders (suite/quiet and suite/windy with 24 decoys, 3×3 cells,
/// 187 200 pooled z): 99th percentile 3.40, 99.9th 4.17 → 3.5. Never set from a pass.
pub const TF_Z_FLOOR: f64 = 3.5;
/// …and at most this much: twice the floor (a noise-percentile cap would be below 1 — the
/// excess of the 99.9th percentile is 0.67 — and could not stop one bright ridge counting as a
/// multiple of a faint one).
pub const TF_Z_CAP: f64 = 2.0 * TF_Z_FLOOR;
/// The jet band's floor: silent faint_jet render on the jet image (19 800 pooled z): 99th
/// percentile 2.86, 99.9th 3.12 → 3.0 (more than 0.5 from the prop floor, so its own).
pub const TF_Z_FLOOR_JET: f64 = 3.0;
pub const TF_Z_CAP_JET: f64 = 2.0 * TF_Z_FLOOR_JET;
/// Two answers closer than this agree.
pub const TF_AGREE_M: f64 = 1000.0;
/// A clock prior at most ± this is narrow enough for the blind leg.
pub const TF_NARROW_S: f64 = 600.0;
/// Padding of the blind window built from metadata creation time.
pub const TF_META_PAD_S: f64 = 60.0;
/// Track-first is skipped on areas larger than this (≈ a 10 km radius).
pub const TF_MAX_AREA_KM2: f64 = 315.0;
