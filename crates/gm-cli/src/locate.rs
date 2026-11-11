//! `geomedia locate`: events from the audio, aircraft from ADS-B (or a simulator truth file),
//! terrain, the search, and a report. Exit code 0 = located, 3 = inconclusive or ambiguous.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use gm_core::Epoch;
use gm_search::{AircraftClass, Candidate, Inputs, Outcome, SearchParams, Verdict, search};
use gm_sources::adsblol;
use gm_sources::area::When;
use gm_sources::terrain::Terrain;

use crate::preflight::parse_plan;

/// Refine-only guidance may move the answer at most the first answer's 90% radius plus this.
pub const REFINE_SLACK_M: f64 = 125.0;

/// Ellipsoidal ground height at (lat, lon).
pub type Ground = Box<dyn Fn(f64, f64) -> f64 + Sync>;

pub struct Args {
    pub file: PathBuf,
    pub area: String,
    pub when: String,
    pub yes: bool,
    pub windows: Vec<String>,
    pub out: Option<PathBuf>,
    pub json: Option<String>,
    pub sim_truth: Option<PathBuf>,
    /// Reuse events from a previous run's events.json instead of decoding the audio.
    pub events: Option<PathBuf>,
    /// Print the per-event evidence at this lat,lon (diagnostics).
    pub debug_cell: Option<String>,
    /// Weather column for the propagation model (ignored for simulator input).
    pub weather: Weather,
    /// Add guided events: real aircraft's predicted curves measured in the spectrogram.
    pub guided: bool,
    /// Also run track-first scoring over `when` and combine it with the blind answer.
    pub track_first: bool,
    /// The blind leg's clock window when `when` is wide (e.g. only the day is known).
    pub blind_when: Option<String>,
    /// Evaluation only: ADS-B from this many days away, re-timed onto the recording's day.
    pub traffic_shift_days: i64,
    /// Evaluation only: a known (lat, lon) for track-first's truth probe.
    pub truth: Option<(f64, f64)>,
    /// Find aircraft sounds (stage 1) and locate passes with no curve by timing and loudness.
    pub envelope: bool,
    /// Development: score every pass by timing and loudness and take that answer.
    pub envelope_only: bool,
    /// Keep only tonal events on a detected aircraft sound.
    pub pass_gate: bool,
}

/// Whether tonal events on no detected aircraft sound are dropped: only with `--pass-gate`, and
/// never when the analyst marked `--event` windows by hand (their mark says where the aircraft is).
pub fn gates_to_passes(a: &Args) -> bool {
    a.pass_gate && a.windows.is_empty()
}

/// The blind leg's clock window: `blind_when` if given; else `when` if narrow (≤ ±TF_NARROW_S);
/// else from the metadata creation time `creation` — start in [ct − duration − TF_META_PAD_S,
/// ct + TF_META_PAD_S], since phones stamp the start or the end (untrusted); else none.
pub fn blind_window(
    when: &When,
    blind_when: Option<When>,
    creation: Option<Epoch>,
    duration_s: f64,
) -> Option<When> {
    use gm_search::params::{TF_META_PAD_S, TF_NARROW_S};
    if let Some(b) = blind_when {
        return Some(b);
    }
    if when.minus_s + when.plus_s <= 2.0 * TF_NARROW_S {
        return Some(*when);
    }
    let ct = creation?;
    let (lo, hi) = (-duration_s - TF_META_PAD_S, TF_META_PAD_S);
    Some(When {
        center: ct.plus_seconds(0.5 * (lo + hi)),
        minus_s: 0.5 * (hi - lo),
        plus_s: 0.5 * (hi - lo),
    })
}

/// Track-first's traffic window: the prior's emissions padded by TF_SHIFT_WINDOW_S either side
/// (its null shifts every aircraft by up to that much).
pub fn traffic_window(when: &When, duration_s: f64) -> (Epoch, Epoch) {
    use gm_search::params::TF_SHIFT_WINDOW_S;
    (
        when.start()
            .plus_seconds(-crate::preflight::EMISSION_PAD_S - TF_SHIFT_WINDOW_S),
        when.end().plus_seconds(duration_s + TF_SHIFT_WINDOW_S),
    )
}

/// UTC dates from `a`'s to `b`'s, inclusive.
pub fn days_between(a: Epoch, b: Epoch) -> Vec<time::Date> {
    let day = |e: Epoch| {
        time::OffsetDateTime::from_unix_timestamp(e.ns.div_euclid(1_000_000_000))
            .map(|t| t.date())
            .ok()
    };
    let (Some(mut d), Some(d1)) = (day(a), day(b)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    while d <= d1 {
        out.push(d);
        match d.next_day() {
            Some(n) => d = n,
            None => break,
        }
    }
    out
}

/// Whether track-first may run on `area` (`Err`: why not).
pub fn track_first_allowed(area: &gm_sources::area::Area) -> std::result::Result<(), String> {
    let km2 = area.area_km2();
    if km2 > gm_search::params::TF_MAX_AREA_KM2 {
        Err(format!(
            "area too large for track-first ({km2:.0} km² > {:.0} km²; use a city-sized area)",
            gm_search::params::TF_MAX_AREA_KM2
        ))
    } else {
        Ok(())
    }
}

/// The metadata creation time (UTC, plausible) of `report`, if any.
fn creation_time(report: Option<&gm_media::MediaReport>) -> Option<Epoch> {
    use gm_media::model::{HintKind, HintValue};
    report?
        .hints
        .iter()
        .find_map(|h| match (&h.kind, &h.value) {
            (
                HintKind::CreationTime,
                HintValue::Time {
                    utc,
                    plausible: true,
                },
            ) => Epoch::from_rfc3339(utc),
            _ => None,
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Weather {
    /// Open-Meteo historical forecast, pressure levels (free, CC BY 4.0, non-commercial terms).
    #[value(name = "open-meteo")]
    OpenMeteo,
    /// Still air plus a freely fitted wind.
    None,
}

/// The atmosphere to search in, the wind-fit prior σ, and a one-line note for the report.
pub fn choose_atmosphere(
    profile: Result<gm_sources::weather::Profile, gm_sources::SourceError>,
    frame: &gm_core::LocalFrame,
    geoid_n_m: f64,
) -> (gm_search::atmos::Atmosphere, f64, String) {
    use gm_search::atmos::Atmosphere;
    use gm_search::params::{WIND_SIGMA_FORECAST, WIND_SIGMA_FREE};
    match profile.map(|p| (gm_sources::weather::column(&p, frame.clone(), geoid_n_m), p)) {
        Ok((Some(col), p)) => {
            let note = format!(
                "weather: {} levels from {} for {}",
                p.levels.len(),
                p.source,
                p.time_utc
            );
            (Atmosphere::Layered(col), WIND_SIGMA_FORECAST, note)
        }
        Ok((None, _)) => (
            Atmosphere::still(),
            WIND_SIGMA_FREE,
            "weather: unavailable (unusable profile); using still air".into(),
        ),
        Err(e) => (
            Atmosphere::still(),
            WIND_SIGMA_FREE,
            format!("weather: unavailable ({e}); using still air"),
        ),
    }
}

/// Everything `locate` computes, for printing or evaluation.
pub struct LocateResult {
    pub outcome: Outcome,
    pub events: Vec<gm_doppler::audio::Event>,
    pub aircraft: Vec<Candidate>,
    pub origin: Epoch,
    pub duration_s: f64,
    pub search_s: f64,
    /// Which weather the atmosphere came from (or why none).
    pub weather: String,
    /// Ellipsoidal ground height used (for evaluation at a truth position).
    pub ground: Ground,
    /// The blind leg's own answer (`outcome` is the combined one).
    pub blind: Outcome,
    /// Track-first's result, and why it did not run when `None`.
    pub track_first: Option<gm_search::trackfirst::TfOutcome>,
    pub track_first_note: String,
    /// Which method the combined answer comes from, and why.
    pub source: gm_search::trackfirst::Source,
    pub combine_note: String,
    /// Track-first's clock origin (its δ is relative to it; `origin` is the blind leg's).
    pub tf_origin: Epoch,
    /// Every aircraft sound found, its curve result and what location used it for.
    pub pass_log: Option<gm_doppler::audio::PassLog>,
    /// The timing-and-loudness leg (δ on the blind leg's clock), and why it did not run.
    pub envelope: Option<gm_search::envelope::EnvOutcome>,
    pub envelope_note: String,
}

pub fn locate(a: &Args) -> Result<LocateResult> {
    gm_sources::paths::touch_media(&a.file);
    let (plan, report) = parse_plan(Some(&a.file), None, &a.area, &a.when)?;
    let given_blind = a.blind_when.as_deref().map(When::parse).transpose()?;
    let creation = creation_time(report.as_ref());
    let bw = blind_window(&plan.when, given_blind, creation, plan.duration_s);
    match (&bw, given_blind.is_some(), creation) {
        (Some(w), false, Some(ct)) if w.center != plan.when.center => eprintln!(
            "blind leg: clock window from the metadata creation time {} (untrusted)",
            ct.to_rfc3339()
        ),
        (None, _, _) => eprintln!(
            "blind leg: no narrow time (give --blind-when, or a narrow --when): it finds passes but does not locate"
        ),
        _ => {}
    }
    // The blind leg's clock: its own window (and origin); track-first's: the whole prior.
    let bwhen = bw.unwrap_or(plan.when);
    let tf_allowed = if a.track_first {
        track_first_allowed(&plan.area)
    } else {
        Err("track-first off (--no-track-first)".to_string())
    };
    if let Err(why) = &tf_allowed {
        eprintln!("track-first: not run: {why}");
    }
    let want_imgs = a.guided || tf_allowed.is_ok();
    let shift_s = a.traffic_shift_days as f64 * 86_400.0;
    let seed = report
        .as_ref()
        .and_then(|r| u64::from_str_radix(r.file.sha256.get(..16)?, 16).ok())
        .unwrap_or(0x5eed);
    let wins_of = |duration: f64| {
        a.windows
            .iter()
            .map(|w| crate::parse_window(w, duration))
            .collect::<Result<Vec<_>>>()
    };
    let decode = || {
        gm_doppler::audio::decode(&a.file)
            .with_context(|| format!("cannot read audio from {}", a.file.display()))
    };
    // The decoded audio is kept for guidance; with --events it's decoded only when guided.
    let (mut events, duration, audio, det): (Vec<gm_doppler::audio::Event>, f64, _, _) =
        match &a.events {
            Some(p) => {
                let v: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(p).with_context(|| format!("cannot read {}", p.display()))?,
                )?;
                let ev: Vec<gm_doppler::audio::Event> = serde_json::from_value(v["events"].clone())
                    .context("events.json: bad events")?;
                // A cache from an earlier guided run: guidance is redone from the audio (or not).
                let ev = ev.into_iter().filter(|e| !e.guided).collect();
                let audio = if want_imgs || a.envelope {
                    Some(decode()?)
                } else {
                    None
                };
                let det = audio.as_ref().filter(|_| a.envelope).map(|au| {
                    gm_doppler::audio::detect(au, &gm_doppler::audio::DetectParams::default())
                });
                (
                    ev,
                    v["duration_s"].as_f64().unwrap_or(plan.duration_s),
                    audio.filter(|_| want_imgs),
                    det,
                )
            }
            None => {
                let audio = decode()?;
                let duration = audio.samples.len() as f64 / f64::from(audio.sample_rate.max(1));
                let wins = wins_of(duration)?;
                let ev = gm_doppler::audio::discover(
                    &audio,
                    &gm_doppler::audio::EventParams::default(),
                    &wins,
                );
                let det = a.envelope.then(|| {
                    gm_doppler::audio::detect(&audio, &gm_doppler::audio::DetectParams::default())
                });
                (ev, duration, want_imgs.then_some(audio), det)
            }
        };
    let mut pass_log = det
        .as_ref()
        .map(|d| gm_doppler::audio::passlog::build(d, &events));
    if let Some(log) = pass_log.as_mut() {
        log.mark_tone_only(gates_to_passes(a));
    }
    if a.pass_gate && !a.windows.is_empty() && pass_log.is_some() {
        eprintln!("pass gate: off (--event windows were given by hand)");
    }
    if gates_to_passes(a)
        && let Some(log) = &pass_log
    {
        let before = events.len();
        events = gm_doppler::audio::passlog::gate_to_passes(&events, log);
        eprintln!(
            "pass gate: {} of {before} tonal events lie on an aircraft sound",
            events.len()
        );
    }
    let origin = bwhen.center;
    let tf_origin = plan.when.center;
    // Track-first's traffic span on its own clock (null shifts stay inside it).
    let mut traffic_span = {
        let (w0, w1) = traffic_window(&plan.when, duration);
        (w0.seconds_since(tf_origin), w1.seconds_since(tf_origin))
    };
    #[allow(clippy::type_complexity)]
    let (aircraft, decoys, traffic, ground, geoid_n): (
        Vec<Candidate>,
        Vec<Candidate>,
        Vec<Candidate>,
        Ground,
        Option<f64>,
    ) = match &a.sim_truth {
        Some(p) => {
            let truth: gm_sim::Truth = serde_json::from_slice(
                &std::fs::read(p).with_context(|| format!("cannot read {}", p.display()))?,
            )
            .with_context(|| format!("{} is not a geomedia sim truth file", p.display()))?;
            let start = Epoch::from_rfc3339(&truth.start_utc).context("truth start_utc")?;
            let cands_at = |o: Epoch| -> Vec<Candidate> {
                let shift = start.seconds_since(o);
                truth
                    .tracks
                    .iter()
                    .map(|t| {
                        let mut tr = t.track.clone();
                        for k in &mut tr.knots {
                            k.t += shift;
                        }
                        Candidate {
                            icao: t.icao24.clone(),
                            type_code: Some(t.type_code.clone()),
                            class: AircraftClass::Prop,
                            tracks: vec![tr],
                        }
                    })
                    .collect()
            };
            let h = truth.observer.h_m;
            // Simulator traffic is all there is: its span is the prior's ± the shift window.
            (
                cands_at(origin),
                Vec::new(),
                cands_at(tf_origin),
                Box::new(move |_, _| h),
                None,
            )
        }
        None => {
            // Every query reads ADS-B `shift_s` away (evaluation's wrong-day controls) and
            // re-times it onto the recording's day.
            let (t0, t1) = {
                let p = crate::preflight::Plan {
                    area: plan.area.clone(),
                    when: bwhen,
                    duration_s: plan.duration_s,
                    days: Vec::new(),
                    duration_given: false,
                };
                p.window()
            };
            let (w0, w1) = traffic_window(&plan.when, duration);
            let sh = |e: Epoch| e.plus_seconds(shift_s);
            let mut need: Vec<time::Date> = days_between(sh(t0), sh(t1));
            if bw.is_none() {
                need.clear();
            }
            let tf_days = if load_traffic(&tf_allowed, a.envelope) {
                days_between(sh(w0), sh(w1))
            } else {
                Vec::new()
            };
            let mut all: Vec<time::Date> = need.iter().chain(&tf_days).copied().collect();
            all.sort();
            all.dedup();
            let mut tracks = Vec::new();
            let mut decoy_tracks = Vec::new();
            let mut traffic_tracks = Vec::new();
            let mut loaded: Vec<time::Date> = Vec::new();
            for d in all {
                let progress = &mut |name: &str, done: u64, total: Option<u64>| {
                    eprint!(
                        "\r  {name}: {} / {}",
                        gm_sources::human_size(done),
                        total.map_or("?".into(), gm_sources::human_size)
                    );
                };
                // The blind leg's days are required; track-first's extra days are best
                // effort (its span is clipped to what loaded).
                let day = if need.contains(&d) {
                    adsblol::ensure_day(d, a.yes, progress)?
                } else {
                    match adsblol::ensure_day(d, a.yes, progress) {
                        Ok(x) => x,
                        Err(e) => {
                            eprintln!(
                                "track-first: ADS-B for {d} not available ({e}); its traffic span is clipped"
                            );
                            continue;
                        }
                    }
                };
                loaded.push(d);
                let idx = adsblol::day_index(&day)?;
                if need.contains(&d) {
                    tracks.extend(gm_sources::query::query(
                        &idx,
                        &day,
                        &plan.area,
                        sh(t0),
                        sh(t1),
                        sh(origin),
                    )?);
                    if a.guided {
                        // Decoys for the guided null: other traffic of the same day file.
                        let (v0, v1) = decoy_window(t0, t1);
                        decoy_tracks.extend(gm_sources::query::query(
                            &idx,
                            &day,
                            &plan.area,
                            sh(v0),
                            sh(v1),
                            sh(origin),
                        )?);
                    }
                }
                if tf_days.contains(&d) {
                    traffic_tracks.extend(gm_sources::query::query(
                        &idx,
                        &day,
                        &plan.area,
                        sh(w0),
                        sh(w1),
                        sh(tf_origin),
                    )?);
                }
            }
            if load_traffic(&tf_allowed, a.envelope) {
                // The span the traffic actually covers: the loaded days within the window.
                let day_start = |d: time::Date| {
                    Epoch::from_rfc3339(&format!("{d}T00:00:00Z")).map(|e| e.plus_seconds(-shift_s))
                };
                let lo = loaded
                    .iter()
                    .filter(|d| tf_days.contains(d))
                    .min()
                    .and_then(|&d| day_start(d));
                let hi = loaded
                    .iter()
                    .filter(|d| tf_days.contains(d))
                    .max()
                    .and_then(|&d| day_start(d))
                    .map(|e| e.plus_seconds(86_400.0));
                if let (Some(lo), Some(hi)) = (lo, hi) {
                    traffic_span = (
                        traffic_span.0.max(lo.seconds_since(tf_origin)),
                        traffic_span.1.min(hi.seconds_since(tf_origin)),
                    );
                }
            }
            let types = gm_sources::types::type_table().unwrap_or_default();
            let terrain = Terrain::ensure(&plan.area, a.yes)?;
            let (clat, clon) = plan.area.centre();
            let n = terrain.geoid_n(clat, clon);
            let cands = gm_search::candidates_from(tracks, &types);
            let decoys = without(gm_search::candidates_from(decoy_tracks, &types), &cands);
            let traffic = gm_search::candidates_from(traffic_tracks, &types);
            if a.guided && bw.is_some() {
                eprintln!("guided null: {} decoy aircraft (±3 h)", decoys.len());
            }
            if load_traffic(&tf_allowed, a.envelope) {
                eprintln!(
                    "traffic: {} aircraft in the area over the prior ± 3 h",
                    traffic.len()
                );
            }
            (
                cands,
                decoys,
                traffic,
                Box::new(move |la, lo| terrain.ground_h_ell(la, lo)),
                Some(n),
            )
        }
    };
    let (final_atmos, sigma, weather) = match (geoid_n, a.weather) {
        (Some(n), Weather::OpenMeteo) => {
            let (clat, clon) = plan.area.centre();
            let frame = gm_core::LocalFrame::new(gm_core::Geodetic {
                lat_deg: clat,
                lon_deg: clon,
                h_m: 0.0,
            });
            choose_atmosphere(gm_sources::weather::profile(clat, clon, origin), &frame, n)
        }
        (None, _) => (
            gm_search::atmos::Atmosphere::still(),
            gm_search::params::WIND_SIGMA_FREE,
            "weather: not used for simulator input".to_string(),
        ),
        (Some(_), Weather::None) => (
            gm_search::atmos::Atmosphere::still(),
            gm_search::params::WIND_SIGMA_FREE,
            "weather: off (--weather none); still air".to_string(),
        ),
    };
    eprintln!("{}", gm_report::sanitize_terminal(&weather));
    // Coarse and fine search in still air; the weather column only in the final refinement.
    let atmos = gm_search::atmos::Atmosphere::still();
    let final_atmos = match final_atmos {
        gm_search::atmos::Atmosphere::Layered(_) => Some(final_atmos),
        gm_search::atmos::Atmosphere::Uniform(_) => None,
    };
    // The guide image, built once: the free guided sweep now, refine-only guidance after the
    // search; the audio itself is no longer needed.
    // The guide images, built once: propeller/rotor combs and jets' airframe/engine tones. The
    // free guided sweep runs now, refine-only guidance after the search; the audio itself is no
    // longer needed.
    use gm_doppler::audio::guide::{GuideParams, guide_image_with};
    use gm_search::Band;
    let imgs: Option<Vec<(Band, gm_doppler::audio::guide::GuideImage)>> =
        audio.as_ref().map(|au| {
            vec![
                (Band::Prop, guide_image_with(au, &GuideParams::prop())),
                (Band::Jet, guide_image_with(au, &GuideParams::jet())),
            ]
        });
    // The noise level over time, for the silence check (measured, not assumed).
    let noise = audio
        .as_ref()
        .map(gm_search::noise::NoiseProfile::from_audio);
    drop(audio);
    if let Some(imgs) = imgs.as_ref().filter(|_| a.guided && bw.is_some()) {
        let wins = wins_of(duration)?;
        // Each band against the discovered events; then claimants are compared across bands.
        let mut all = Vec::new();
        for (band, img) in imgs {
            let t = std::time::Instant::now();
            let found = gm_search::guided_events(
                img,
                &Inputs {
                    events: &events,
                    aircraft: &aircraft,
                    area: &plan.area,
                    when: bwhen,
                    duration_s: duration,
                    ground: &*ground,
                    atmos: &atmos,
                    noise: noise.as_ref(),
                    decoys: &decoys,
                },
                *band,
            );
            let guided: Vec<_> = found
                .into_iter()
                .filter(|e| {
                    wins.is_empty()
                        || wins
                            .iter()
                            .any(|&(lo, hi)| e.fit.tc >= lo && e.fit.tc <= hi)
                })
                .collect();
            eprintln!(
                "guided ({band:?}): {} event(s) from {} candidate aircraft in {:.1} s",
                guided.len(),
                aircraft.iter().filter(|c| band.wants(c.class)).count(),
                t.elapsed().as_secs_f64()
            );
            for e in &guided {
                eprintln!(
                    "  guided event #{} ← {} (z {:.1}, null {:.1}{}): tc {:.1} s, {:.0} m/s, {:.0} m",
                    e.id,
                    gm_report::sanitize_terminal(e.guide_icao.as_deref().unwrap_or("?")),
                    e.guide_z.unwrap_or(f64::NAN),
                    e.guide_null_z.unwrap_or(f64::NAN),
                    e.guide_null_n
                        .map(|(s, d)| format!(" (shifts {s}, decoys {d})"))
                        .unwrap_or_default(),
                    e.fit.tc,
                    e.fit.v,
                    e.fit.d
                );
            }
            all.extend(guided);
        }
        let n0 = events.len();
        for (k, mut e) in gm_search::drop_contested(all).into_iter().enumerate() {
            e.id = n0 + k + 1;
            events.push(e);
        }
    }
    let inp = Inputs {
        events: &events,
        aircraft: &aircraft,
        area: &plan.area,
        when: bwhen,
        duration_s: duration,
        ground: &*ground,
        atmos: &atmos,
        noise: noise.as_ref(),
        decoys: &decoys,
    };
    if let Some(dc) = &a.debug_cell {
        debug_cell(&inp, dc)?;
    }
    let t = std::time::Instant::now();
    let params = SearchParams {
        wind_sigma: Some(sigma),
        final_atmos,
        ..SearchParams::default()
    };
    let outcome = if bw.is_some() {
        search(&inp, &params)
    } else {
        Outcome {
            verdict: Verdict::Inconclusive("no narrow time: blind leg not located".into()),
            best: None,
            top: vec![],
            mirror: None,
            delta_s: 0.0,
            matched: vec![],
            silent: vec![],
            region90_km2: 0.0,
            on_boundary: false,
            heat: vec![],
            frame_origin: gm_core::Geodetic {
                lat_deg: plan.area.centre().0,
                lon_deg: plan.area.centre().1,
                h_m: 0.0,
            },
            wind_enu: None,
            atmosphere: atmos.describe(),
        }
    };
    // Refine-only guidance: with a location found, test the other aircraft's curves at that
    // place and clock only. Their events may sharpen the answer — it must stay located and
    // inside the first answer's 90% region — never create one.
    let refine_imgs = imgs.as_ref().filter(|_| a.guided);
    let (outcome, events) = match (refine_imgs, &outcome.verdict, &outcome.best) {
        (Some(imgs), Verdict::Located, Some(b)) => {
            let t = std::time::Instant::now();
            let matched: Vec<String> = outcome.matched.iter().map(|(_, i)| i.clone()).collect();
            let mut extra = Vec::new();
            for (band, img) in imgs {
                let mut more = gm_search::refine_events(
                    img,
                    &Inputs {
                        events: &events,
                        ..inp
                    },
                    *band,
                    b.east,
                    b.north,
                    outcome.delta_s,
                    &matched,
                );
                extra.append(&mut more);
            }
            let mut extra = gm_search::drop_contested(extra);
            for (k, e) in extra.iter_mut().enumerate() {
                e.id = events.len() + k + 1;
            }
            eprintln!(
                "refine: {} event(s) at the located cell in {:.1} s",
                extra.len(),
                t.elapsed().as_secs_f64()
            );
            for e in &extra {
                eprintln!(
                    "  refine event #{} ← {} (z {:.1}, null {:.1}{}): tc {:.1} s, {:.0} m/s, {:.0} m",
                    e.id,
                    gm_report::sanitize_terminal(e.guide_icao.as_deref().unwrap_or("?")),
                    e.guide_z.unwrap_or(f64::NAN),
                    e.guide_null_z.unwrap_or(f64::NAN),
                    e.guide_null_n
                        .map(|(s, d)| format!(" (shifts {s}, decoys {d})"))
                        .unwrap_or_default(),
                    e.fit.tc,
                    e.fit.v,
                    e.fit.d
                );
            }
            if extra.is_empty() {
                (outcome, events)
            } else {
                let mut more = events.clone();
                more.extend(extra);
                let o2 = search(
                    &Inputs {
                        events: &more,
                        ..inp
                    },
                    &params,
                );
                let r90 = (outcome.region90_km2 / std::f64::consts::PI).sqrt() * 1000.0;
                let kept = match (&o2.verdict, &o2.best) {
                    (Verdict::Located, Some(b2)) => {
                        let moved = (b2.east - b.east).hypot(b2.north - b.north);
                        eprintln!(
                            "refine: re-search moved {moved:.0} m (first answer's 90% radius {r90:.0} m); 90% region {:.2} → {:.2} km²",
                            outcome.region90_km2, o2.region90_km2
                        );
                        moved <= r90 + crate::locate::REFINE_SLACK_M
                    }
                    _ => {
                        eprintln!("refine: re-search not located; keeping the first answer");
                        false
                    }
                };
                if kept { (o2, more) } else { (outcome, events) }
            }
        }
        _ => (outcome, events),
    };
    // Track-first over the whole prior, back to back with the blind leg.
    let (tf, tf_note) = match (&imgs, tf_allowed) {
        (Some(imgs), Ok(())) => {
            let seed_tc: Vec<f64> = events
                .iter()
                .filter(|e| !e.guided)
                .map(|e| e.fit.tc)
                .collect();
            let r = gm_search::trackfirst::track_first(
                imgs,
                &Inputs {
                    events: &events,
                    aircraft: &traffic,
                    area: &plan.area,
                    when: plan.when,
                    duration_s: duration,
                    ground: &*ground,
                    atmos: &atmos,
                    noise: noise.as_ref(),
                    decoys: &[],
                },
                seed,
                &seed_tc,
                traffic_span,
                a.truth,
            );
            let t = |k: &str| r.timings.iter().find(|x| x.0 == k).map_or(0.0, |x| x.1);
            eprintln!(
                "track-first: {} pairs, maps {:.1} s, joint {:.1} s, null {:.1} s ({} replicates), fine {:.1} s → {}",
                r.pairs,
                t("maps"),
                t("joint"),
                t("null"),
                r.replicates,
                t("fine"),
                gm_report::sanitize_terminal(&verdict_text(&r.outcome.verdict))
            );
            (Some(r), String::new())
        }
        (None, Ok(())) => (None, "no audio for track-first".to_string()),
        (_, Err(why)) => (None, why),
    };
    // One answer. Track-first's δ is on its own clock: put it on the blind leg's.
    let tf_on_blind = tf.as_ref().map(|r| Outcome {
        delta_s: r.outcome.delta_s + tf_origin.seconds_since(origin),
        ..r.outcome.clone()
    });
    let (combined, source, mut combine_note) =
        gm_search::trackfirst::combine(&outcome, tf_on_blind.as_ref());
    // Timing and loudness for aircraft sounds with no tonal curve.
    let still = gm_search::atmos::Atmosphere::still();
    let (envelope, envelope_note) = match (&det, pass_log.as_mut()) {
        (_, _) if !a.envelope => (None, "timing and loudness off (--no-envelope)".to_string()),
        (Some(d), Some(log)) => {
            let ids: Vec<usize> = log
                .passes
                .iter()
                .filter(|r| {
                    a.envelope_only || matches!(r.curve, gm_doppler::audio::Curve::None { .. })
                })
                .map(|r| r.pass.id)
                .collect();
            if ids.is_empty() {
                (
                    None,
                    if log.passes.is_empty() {
                        "no aircraft sound found".to_string()
                    } else {
                        "every aircraft sound has a tonal curve".to_string()
                    },
                )
            } else {
                let envs = gm_search::envelope::pass_envelopes(d, &ids);
                let inp = Inputs {
                    events: &[],
                    aircraft: &[],
                    area: &plan.area,
                    when: plan.when,
                    duration_s: duration,
                    ground: &*ground,
                    atmos: &still,
                    noise: None,
                    decoys: &[],
                };
                let te = std::time::Instant::now();
                let mut e = gm_search::envelope::envelope_search(
                    &inp,
                    &traffic,
                    &envs,
                    &gm_search::envelope::EnvParams::default(),
                );
                // Onto the blind leg's clock.
                e.delta_s += tf_origin.seconds_since(origin);
                eprintln!(
                    "timing and loudness: {} pass(es), {} aircraft, {:.1} s → {:?} (p {:.3})",
                    envs.len(),
                    traffic.len(),
                    te.elapsed().as_secs_f64(),
                    e.verdict,
                    e.p
                );
                (Some(e), String::new())
            }
        }
        _ => (
            None,
            "no audio for stage 1 (events from a file)".to_string(),
        ),
    };
    let combined = match &envelope {
        Some(e) => {
            let (o, note) =
                gm_search::envelope::combine_with_envelope(combined, e, a.envelope_only);
            combine_note = if combine_note.is_empty() {
                note
            } else {
                format!("{combine_note}; {note}")
            };
            o
        }
        None => combined,
    };
    if let Some(log) = pass_log.as_mut() {
        let ids: Vec<usize> = log
            .passes
            .iter()
            .filter(|r| a.envelope_only || matches!(r.curve, gm_doppler::audio::Curve::None { .. }))
            .map(|r| r.pass.id)
            .collect();
        mark_uses(log, &ids, envelope.as_ref(), &envelope_note);
    }
    let search_s = t.elapsed().as_secs_f64();
    Ok(LocateResult {
        outcome: combined,
        events,
        aircraft,
        origin,
        duration_s: duration,
        search_s,
        weather,
        ground,
        blind: outcome,
        track_first: tf,
        track_first_note: tf_note,
        source,
        combine_note,
        tf_origin,
        pass_log,
        envelope,
        envelope_note,
    })
}

/// What location used each pass for: its tonal curve; timing and loudness when that leg gave
/// an answer (a band counts: it is reported); otherwise unused, with the leg's reason.
pub fn mark_uses(
    log: &mut gm_doppler::audio::PassLog,
    ids: &[usize],
    env: Option<&gm_search::envelope::EnvOutcome>,
    not_run: &str,
) {
    use gm_doppler::audio::{Curve, UsedFor};
    for r in &mut log.passes {
        let scored = ids.contains(&r.pass.id);
        r.used_for = match (scored, env, &r.curve) {
            (false, _, Curve::Tonal { .. }) => UsedFor::Curve,
            (false, _, Curve::None { .. }) => UsedFor::Unused {
                reason: if not_run.is_empty() {
                    "not scored".into()
                } else {
                    not_run.into()
                },
            },
            (true, None, _) => UsedFor::Unused {
                reason: not_run.into(),
            },
            (true, Some(e), _) => match &e.verdict {
                gm_search::envelope::EnvVerdict::Inconclusive { reason } => UsedFor::Unused {
                    reason: format!("timing and loudness inconclusive: {reason}"),
                },
                _ if !e.per_pass.iter().any(|x| x.0 == r.pass.id) => UsedFor::Unused {
                    reason: "too short for timing and loudness (fewer than 3 blocks)".into(),
                },
                _ => UsedFor::TimingLoudness,
            },
        };
    }
}

/// Whether to load the ±3 h traffic: for track-first, or for the timing-and-loudness leg even
/// when track-first is refused for the area (its own area cap applies).
pub fn load_traffic(tf_allowed: &std::result::Result<(), String>, envelope: bool) -> bool {
    tf_allowed.is_ok() || envelope
}

/// The run's JSON document (with the pass log and the envelope) and its HTML report.
pub fn documents(res: &LocateResult, a: &Args) -> Result<(String, String)> {
    let LocateResult {
        outcome: o,
        events,
        aircraft,
        origin,
        search_s: secs,
        weather,
        blind,
        track_first: tf,
        track_first_note,
        source,
        combine_note,
        tf_origin,
        pass_log,
        envelope,
        envelope_note,
        ..
    } = res;
    let (origin, secs, source, tf_origin) = (*origin, *secs, *source, *tf_origin);
    let mut doc = report_json(o, events, aircraft, origin, secs);
    doc["model"]["weather"] = serde_json::Value::String(weather.clone());
    doc["source"] = serde_json::to_value(source)?;
    doc["combine"] = serde_json::Value::String(combine_note.clone());
    doc["blind_verdict"] = serde_json::Value::String(verdict_text(&blind.verdict));
    add_pass_log(
        &mut doc,
        pass_log.as_ref(),
        envelope.as_ref(),
        envelope_note,
    )?;
    doc["track_first"] = match tf {
        Some(r) => {
            let mut v = serde_json::to_value(r)?;
            v["recording_start_utc"] =
                serde_json::Value::String(tf_origin.plus_seconds(r.outcome.delta_s).to_rfc3339());
            v
        }
        None => serde_json::json!({ "not_run": track_first_note }),
    };
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    let html = report_html(
        o,
        events,
        aircraft,
        origin,
        &a.file,
        if a.sim_truth.is_some() {
            "a simulator truth file"
        } else {
            "ADS-B (adsb.lol)"
        },
        weather,
    );
    let html = html.replacen(
        "</main></body></html>",
        &format!(
            "{}</main></body></html>",
            tf_html(
                blind,
                tf.as_ref(),
                track_first_note,
                combine_note,
                tf_origin
            ) + &pass_html(pass_log.as_ref(), envelope.as_ref(), envelope_note)
        ),
        1,
    );
    Ok((text, html))
}

pub fn run(a: &Args) -> Result<u8> {
    let res = locate(a)?;
    let (text, html) = documents(&res, a)?;
    let LocateResult {
        outcome: o,
        events,
        aircraft,
        origin,
        duration_s: duration,
        blind,
        track_first: tf,
        track_first_note,
        source,
        combine_note,
        tf_origin,
        pass_log,
        envelope,
        envelope_note,
        ..
    } = res;
    if a.json.as_deref() == Some("-") {
        print!("{text}");
    } else {
        print!(
            "{}",
            summary(&o, &events, &aircraft, origin, aircraft.len())
        );
        print!(
            "{}",
            tf_text(
                &blind,
                tf.as_ref(),
                &track_first_note,
                source,
                &combine_note,
                tf_origin
            )
        );
        print!(
            "{}",
            pass_text(pass_log.as_ref(), envelope.as_ref(), &envelope_note)
        );
    }
    if let Some(p) = a.json.as_deref().filter(|p| *p != "-") {
        std::fs::write(p, &text).with_context(|| format!("cannot write {p}"))?;
    }
    // Every run gets its own managed folder (listed and cleaned by `geomedia data`); `--out`
    // additionally receives a copy that is the user's own.
    let run = crate::runs::new_run("locate", Some(&a.file))?;
    let out = a.out.as_deref();
    crate::runs::write_both(&run, out, "locate.json", text.as_bytes())?;
    crate::runs::write_both(
        &run,
        out,
        "events.json",
        &serde_json::to_vec(&serde_json::json!({"duration_s": duration, "events": events}))?,
    )?;
    crate::runs::write_both(&run, out, "locate.html", html.as_bytes())?;
    crate::runs::finish_run(
        &run,
        &crate::runs::RunInfo {
            kind: "locate".into(),
            file: a.file.display().to_string(),
            area: a.area.clone(),
            when: a.when.clone(),
            verdict: verdict_kind(&o.verdict).into(),
        },
    )?;
    let shown = gm_report::sanitize_terminal(&run.join("locate.html").display().to_string());
    let copy =
        out.map(|o| gm_report::sanitize_terminal(&o.join("locate.html").display().to_string()));
    let lines =
        std::iter::once(format!("report: {shown}")).chain(copy.map(|c| format!("copy:   {c}")));
    for l in lines {
        if a.json.as_deref() == Some("-") {
            eprintln!("{l}")
        } else {
            println!("{l}")
        }
    }
    Ok(if o.verdict == Verdict::Located { 0 } else { 3 })
}

/// The two legs and how they were combined, for the terminal.
/// Stage 1–3 records into the locate JSON: every pass, every rejected candidate, the envelope.
fn add_pass_log(
    doc: &mut serde_json::Value,
    log: Option<&gm_doppler::audio::PassLog>,
    env: Option<&gm_search::envelope::EnvOutcome>,
    env_note: &str,
) -> Result<()> {
    match log {
        Some(l) => {
            doc["passes"] = serde_json::to_value(&l.passes)?;
            doc["rejected"] = serde_json::to_value(&l.rejected)?;
            doc["events_outside_passes"] = serde_json::to_value(&l.events_outside_passes)?;
            doc["tone_only"] = serde_json::to_value(&l.tone_only)?;
            doc["pass_notes"] = serde_json::to_value(&l.notes)?;
        }
        None => doc["passes"] = serde_json::json!({ "not_run": env_note }),
    }
    doc["envelope"] = match env {
        Some(e) => serde_json::to_value(e)?,
        None => serde_json::json!({ "not_run": env_note }),
    };
    Ok(())
}

fn pass_text(
    log: Option<&gm_doppler::audio::PassLog>,
    env: Option<&gm_search::envelope::EnvOutcome>,
    env_note: &str,
) -> String {
    let mut s = String::from("\naircraft sounds (stage 1):\n");
    match log {
        Some(l) if l.passes.is_empty() && l.rejected.is_empty() => s.push_str("  none found\n"),
        Some(l) => {
            for line in l.lines() {
                s.push_str(&format!("  {}\n", gm_report::sanitize_terminal(&line)));
            }
        }
        None => s.push_str(&format!("  not run: {env_note}\n")),
    }
    match env {
        Some(e) => s.push_str(&format!(
            "timing and loudness: {:?} (p {:.3}{}); per pass: {}\n",
            e.verdict,
            e.p,
            e.p_drop.map_or(String::new(), |x| format!(
                ", without the strongest pass {x:.3}"
            )),
            e.per_pass
                .iter()
                .map(|(id, icao, sc)| format!(
                    "pass {id} → {} ({sc:.1})",
                    icao.as_deref().unwrap_or("none")
                ))
                .collect::<Vec<_>>()
                .join(", ")
        )),
        None => s.push_str(&format!("timing and loudness: not run: {env_note}\n")),
    }
    s
}

fn pass_html(
    log: Option<&gm_doppler::audio::PassLog>,
    env: Option<&gm_search::envelope::EnvOutcome>,
    env_note: &str,
) -> String {
    let esc = gm_report::html::escape_html;
    let mut s = String::from("<section><h2>Aircraft sounds</h2>");
    match log {
        Some(l) => {
            s.push_str("<table><tr><th>pass</th><th>from–to (s)</th><th>peak</th><th>excess</th><th>curve</th><th>used for</th></tr>");
            for r in &l.passes {
                let p = &r.pass;
                let curve = match &r.curve {
                    gm_doppler::audio::Curve::Tonal { event_ids } => format!(
                        "tonal (events {})",
                        event_ids
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    gm_doppler::audio::Curve::None { reason } => format!("none — {reason}"),
                };
                let used = match &r.used_for {
                    gm_doppler::audio::UsedFor::Curve => "its curve".to_string(),
                    gm_doppler::audio::UsedFor::TimingLoudness => "timing and loudness".to_string(),
                    gm_doppler::audio::UsedFor::Unused { reason } => format!("unused — {reason}"),
                };
                s.push_str(&format!(
                    "<tr><td>{}</td><td>{:.1}–{:.1}</td><td>{:.1} s</td><td>+{:.1} dB</td><td>{}</td><td>{}</td></tr>",
                    p.id, p.t_start, p.t_end, p.t_peak, p.excess_db, esc(&curve), esc(&used)
                ));
            }
            for t in &l.tone_only {
                let curve = match &t.on_rejected {
                    Some(reason) => format!("tonal; on a sound rejected as: {reason}"),
                    None => "tonal; loudness below the pass threshold".to_string(),
                };
                let used = match &t.used {
                    gm_doppler::audio::UsedFor::Curve => "its curve".to_string(),
                    gm_doppler::audio::UsedFor::TimingLoudness => "timing and loudness".to_string(),
                    gm_doppler::audio::UsedFor::Unused { reason } => format!("unused — {reason}"),
                };
                s.push_str(&format!(
                    "<tr><td>tone only (event {})</td><td>{:.1}–{:.1}</td><td>—</td><td>—</td><td>{}</td><td>{}</td></tr>",
                    t.event_id, t.t_start, t.t_end, esc(&curve), esc(&used)
                ));
            }
            s.push_str("</table>");
            if !l.rejected.is_empty() {
                s.push_str("<p>Rejected candidates:</p><ul>");
                for r in &l.rejected {
                    s.push_str(&format!(
                        "<li>{:.1}–{:.1} s: {}</li>",
                        r.t_start,
                        r.t_end,
                        esc(&r.reason)
                    ));
                }
                s.push_str("</ul>");
            }
        }
        None => s.push_str(&format!("<p>Not run: {}</p>", esc(env_note))),
    }
    s.push_str("<h2>Timing and loudness</h2>");
    match env {
        Some(e) => s.push_str(&format!(
            "<p>{} — p {:.3}{}. Region: {} cells.</p>",
            esc(&format!("{:?}", e.verdict)),
            e.p,
            e.p_drop.map_or(String::new(), |x| format!(
                ", without the strongest pass {x:.3}"
            )),
            e.region.len()
        )),
        None => s.push_str(&format!("<p>Not run: {}</p>", esc(env_note))),
    }
    s.push_str("</section>");
    s
}

fn tf_text(
    blind: &Outcome,
    tf: Option<&gm_search::trackfirst::TfOutcome>,
    note: &str,
    source: gm_search::trackfirst::Source,
    combine_note: &str,
    tf_origin: Epoch,
) -> String {
    let clean = gm_report::sanitize_terminal;
    let mut s = String::new();
    let _ = writeln!(s, "blind leg: {}", clean(&verdict_text(&blind.verdict)));
    match tf {
        None => {
            let _ = writeln!(s, "track-first: not run ({})", clean(note));
        }
        Some(r) => {
            let _ = writeln!(
                s,
                "track-first: {} ({} aircraft, {} pairs, p {}, without the strongest {}, {} null replicates)",
                clean(&verdict_text(&r.outcome.verdict)),
                r.aircraft,
                r.pairs,
                r.p.map_or("–".into(), |p| format!("{p:.4}")),
                r.p_drop.map_or("–".into(), |p| format!("{p:.4}")),
                r.replicates
            );
            if r.outcome.best.is_some() && !matches!(r.outcome.verdict, Verdict::Inconclusive(_)) {
                let _ = writeln!(
                    s,
                    "  start {} (δ {:+.2} s)",
                    tf_origin.plus_seconds(r.outcome.delta_s).to_rfc3339(),
                    r.outcome.delta_s
                );
            }
            for c in &r.contributors {
                let _ = writeln!(
                    s,
                    "  {} {} {}: slant {:.0} m, bearing {:.0}°, coarse z {:.1} (+{:.2}), fine z {} vs null {}, centred {}, drift {}",
                    if c.counts { "✓" } else { "✗" },
                    clean(&c.icao),
                    clean(c.type_code.as_deref().unwrap_or("?")),
                    c.slant_m,
                    c.bearing_deg,
                    c.z_coarse,
                    c.contribution,
                    c.z_fine.map_or("–".into(), |z| format!("{z:.1}")),
                    c.null_fine.map_or("–".into(), |z| format!("{z:.1}")),
                    c.centred.map_or("–".into(), |x| format!("{x:.2}")),
                    c.drift_pct.map_or("–".into(), |x| format!("{x:+.1}%")),
                );
            }
        }
    }
    let _ = writeln!(s, "combined: {source:?} — {}", clean(combine_note));
    s
}

/// The same for the HTML report.
fn tf_html(
    blind: &Outcome,
    tf: Option<&gm_search::trackfirst::TfOutcome>,
    note: &str,
    combine_note: &str,
    tf_origin: Epoch,
) -> String {
    let e = gm_report::escape_html;
    let mut s = String::from("<h2>Track-first</h2><section>");
    let _ = write!(s, "<p>Blind leg: {}.</p>", e(&verdict_text(&blind.verdict)));
    match tf {
        None => {
            let _ = write!(s, "<p>Track-first did not run: {}.</p>", e(note));
        }
        Some(r) => {
            let _ = write!(
                s,
                "<p>Track-first: {} — {} aircraft, {} pairs; p {} (without the strongest {}) from {} null replicates.</p>",
                e(&verdict_text(&r.outcome.verdict)),
                r.aircraft,
                r.pairs,
                r.p.map_or("–".into(), |p| format!("{p:.4}")),
                r.p_drop.map_or("–".into(), |p| format!("{p:.4}")),
                r.replicates
            );
            if !r.contributors.is_empty() {
                s.push_str("<table><tr><th>counts</th><th>aircraft</th><th>type</th><th>slant m</th><th>bearing</th><th>coarse z</th><th>fine z</th><th>null</th><th>centred</th><th>drift</th><th>f₀ Hz</th></tr>");
                for c in &r.contributors {
                    let _ = write!(
                        s,
                        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.0}</td><td>{:.0}°</td><td>{:.1}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                        if c.counts { "yes" } else { "no" },
                        e(&c.icao),
                        e(c.type_code.as_deref().unwrap_or("?")),
                        c.slant_m,
                        c.bearing_deg,
                        c.z_coarse,
                        c.z_fine.map_or("–".into(), |z| format!("{z:.1}")),
                        c.null_fine.map_or("–".into(), |z| format!("{z:.1}")),
                        c.centred.map_or("–".into(), |x| format!("{x:.2}")),
                        c.drift_pct.map_or("–".into(), |x| format!("{x:+.1}%")),
                        c.f0_hz
                            .map_or("–".into(), |(a, b)| format!("{a:.1} → {b:.1}")),
                    );
                }
                s.push_str("</table>");
            }
            if r.outcome.best.is_some() && !matches!(r.outcome.verdict, Verdict::Inconclusive(_)) {
                let _ = write!(
                    s,
                    "<p>Recording start {}.</p>",
                    e(&tf_origin.plus_seconds(r.outcome.delta_s).to_rfc3339())
                );
            }
        }
    }
    let _ = write!(s, "<p>Combined: {}.</p></section>", e(combine_note));
    s
}

fn verdict_text(v: &Verdict) -> String {
    match v {
        Verdict::Located => "located".into(),
        Verdict::AmbiguousMirror => {
            "ambiguous: mirror solution (one pass can't tell the two sides of its track apart)"
                .into()
        }
        Verdict::AmbiguousStream => {
            "ambiguous: one approach stream (every aircraft that fits lies on one line)".into()
        }
        Verdict::Inconclusive(why) => format!("inconclusive: {why}"),
    }
}

fn type_of(aircraft: &[Candidate], icao: &str) -> String {
    aircraft
        .iter()
        .find(|a| a.icao == icao)
        .and_then(|a| a.type_code.clone())
        .unwrap_or_else(|| "?".into())
}

fn summary(
    o: &Outcome,
    events: &[gm_doppler::audio::Event],
    aircraft: &[Candidate],
    origin: Epoch,
    n_aircraft: usize,
) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{} Doppler events, {n_aircraft} candidate aircraft",
        events.len()
    );
    let _ = writeln!(s, "{}", verdict_text(&o.verdict));
    let osm = |g: &gm_core::Geodetic| {
        format!(
            "{:.6}, {:.6}  https://www.openstreetmap.org/?mlat={}&mlon={}#map=16/{}/{}",
            g.lat_deg, g.lon_deg, g.lat_deg, g.lon_deg, g.lat_deg, g.lon_deg
        )
    };
    let clean = gm_report::sanitize_terminal;
    // A pin is shown only for a located verdict; an ambiguous one shows both candidates; an
    // inconclusive one shows no coordinates at all (anti-goal: no confident wrong pin).
    match (&o.verdict, &o.best) {
        (Verdict::Located, Some(b)) => {
            let _ = writeln!(s, "location: {}", osm(&b.geo));
        }
        (Verdict::AmbiguousMirror | Verdict::AmbiguousStream, Some(b)) => {
            let _ = writeln!(s, "candidate A: {}", osm(&b.geo));
            if let Some(m) = &o.mirror {
                let _ = writeln!(s, "candidate B: {}", osm(&m.geo));
            }
        }
        _ => {}
    }
    if o.verdict != Verdict::Inconclusive(String::new())
        && !matches!(o.verdict, Verdict::Inconclusive(_))
    {
        let _ = writeln!(
            s,
            "recording start: {} (clock offset {:+.2} s)",
            origin.plus_seconds(o.delta_s).to_rfc3339(),
            o.delta_s
        );
        let _ = writeln!(s, "{}", wind_line(o));
        let _ = writeln!(
            s,
            "90% region: {:.2} km²{}",
            o.region90_km2,
            if o.on_boundary {
                "  (at the edge of --area: the answer may lie outside it)"
            } else {
                ""
            }
        );
    }
    for (ei, icao) in &o.matched {
        if let Some(e) = events.get(*ei) {
            let _ = writeln!(
                s,
                "  event #{}{} (tc {:.1} s, {:.0} m/s, {:.0} m) ← {} {}",
                e.id,
                if e.guided { " [guided]" } else { "" },
                e.fit.tc,
                e.fit.v,
                e.fit.d,
                clean(icao),
                clean(&type_of(aircraft, icao))
            );
        }
    }
    if !o.silent.is_empty() {
        let _ = writeln!(
            s,
            "  should have been heard but weren't: {}",
            clean(&o.silent.join(", "))
        );
    }
    if o.verdict == Verdict::Located {
        for (i, r) in o.top.iter().take(5).enumerate() {
            let _ = writeln!(
                s,
                "  #{} {:.6}, {:.6}  p={:.3}",
                i + 1,
                r.geo.lat_deg,
                r.geo.lon_deg,
                r.prob
            );
        }
    }
    s
}

fn verdict_kind(v: &Verdict) -> &'static str {
    match v {
        Verdict::Located => "located",
        Verdict::AmbiguousMirror => "ambiguous_mirror",
        Verdict::AmbiguousStream => "ambiguous_stream",
        Verdict::Inconclusive(_) => "inconclusive",
    }
}

fn report_json(
    o: &Outcome,
    events: &[gm_doppler::audio::Event],
    aircraft: &[Candidate],
    origin: Epoch,
    secs: f64,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "verdict": verdict_text(&o.verdict),
        "verdict_kind": verdict_kind(&o.verdict),
        "located": o.verdict == Verdict::Located,
        "best": if matches!(o.verdict, Verdict::Inconclusive(_)) { None } else { o.best.clone() },
        "mirror": o.mirror,
        "top": o.top,
        "clock_offset_s": o.delta_s,
        "recording_start_utc": origin.plus_seconds(o.delta_s).to_rfc3339(),
        "region90_km2": o.region90_km2,
        "on_boundary": o.on_boundary,
        "matched": o.matched.iter().map(|(e, i)| serde_json::json!({"event": events.get(*e).map(|x| x.id), "icao": i, "type": type_of(aircraft, i)})).collect::<Vec<_>>(),
        "silent": o.silent,
        "events": events.iter().map(|e| serde_json::json!({"id": e.id, "tc": e.fit.tc, "v": e.fit.v, "d": e.fit.d, "f0": e.fit.f0, "rms_hz": e.fit.rms_hz, "guided": e.guided, "guide_icao": e.guide_icao, "guide_z": e.guide_z})).collect::<Vec<_>>(),
        "candidate_aircraft": aircraft.len(),
        "search_seconds": secs,
        "model": {"atmosphere": o.atmosphere, "wind_fit_enu": o.wind_enu.map(|(e, n)| [e, n])},
    })
}

/// "wind: nuisance fit 3.5 m/s from 270° (absorbs model error; not a wind measurement)".
fn wind_line(o: &Outcome) -> String {
    match o.wind_enu {
        Some((e, n)) => {
            let from = (-e).atan2(-n).to_degrees().rem_euclid(360.0);
            format!(
                "wind: nuisance fit {:.1} m/s from {from:.0}° (absorbs model error; not a wind measurement); atmosphere: {}",
                e.hypot(n),
                o.atmosphere
            )
        }
        None => format!("wind: not fitted; atmosphere: {}", o.atmosphere),
    }
}

fn report_html(
    o: &Outcome,
    events: &[gm_doppler::audio::Event],
    aircraft: &[Candidate],
    origin: Epoch,
    file: &Path,
    source: &str,
    weather: &str,
) -> String {
    let e = gm_report::escape_html;
    let name = file.file_name().map_or_else(
        || file.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let (mut x0, mut x1, mut y0, mut y1) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for &(x, y, _) in &o.heat {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
    }
    let pmax = o
        .heat
        .iter()
        .map(|h| h.2)
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let step = 200.0;
    let (w, h) = ((x1 - x0 + step).max(step), (y1 - y0 + step).max(step));
    let mut svg = format!(
        r#"<svg viewBox="0 0 {w:.0} {h:.0}" role="img" aria-label="probability over the area" style="width:100%;max-height:70vh;background:var(--card)">"#
    );
    let to = |x: f64, y: f64| (x - x0 + step / 2.0, y1 - y + step / 2.0);
    for &(x, y, p) in &o.heat {
        let (sx, sy) = to(x, y);
        let a = (p / pmax).powf(0.35);
        let _ = write!(
            svg,
            r#"<rect x="{:.0}" y="{:.0}" width="{step:.0}" height="{step:.0}" fill="var(--accent)" fill-opacity="{a:.3}"/>"#,
            sx - step / 2.0,
            sy - step / 2.0
        );
    }
    let marker = |r: &gm_search::Ranked, cls: &str, rad: f64| {
        let (sx, sy) = to(r.east, r.north);
        format!(r#"<circle cx="{sx:.0}" cy="{sy:.0}" r="{rad:.0}" class="{cls}"/>"#)
    };
    let pin = !matches!(o.verdict, Verdict::Inconclusive(_));
    for r in o.top.iter().take(5).filter(|_| pin) {
        svg.push_str(&marker(r, "top", 40.0));
    }
    if let Some(m) = &o.mirror {
        svg.push_str(&marker(m, "mirror", 90.0));
    }
    if let Some(b) = o.best.as_ref().filter(|_| pin) {
        svg.push_str(&marker(b, "best", 90.0));
    }
    svg.push_str("</svg>");
    let mut rows = String::new();
    for ev in events {
        let m = o
            .matched
            .iter()
            .find(|(i, _)| events.get(*i).is_some_and(|x| x.id == ev.id))
            .map(|(_, icao)| format!("{} {}", icao, type_of(aircraft, icao)));
        let _ = write!(
            rows,
            "<tr><td>#{}{}</td><td>{:.1} s</td><td>{:.0} m/s</td><td>{:.0} m</td><td>{:.1} Hz</td><td>{}</td></tr>",
            ev.id,
            match (&ev.guide_icao, ev.guide_z) {
                (Some(i), Some(z)) if ev.guided => {
                    format!(r#" <span class="sub">guided by {} (z {z:.1})</span>"#, e(i))
                }
                _ => String::new(),
            },
            ev.fit.tc,
            ev.fit.v,
            ev.fit.d,
            ev.fit.f0,
            e(&m.unwrap_or_else(|| "—".into()))
        );
    }
    let pin = !matches!(o.verdict, Verdict::Inconclusive(_));
    let best = o.best.as_ref().filter(|_| pin).map_or("— (no location: see verdict)".into(), |b| {
        format!(r#"{:.6}, {:.6} · <a href="https://www.openstreetmap.org/?mlat={}&amp;mlon={}#map=16/{}/{}">map</a>"#, b.geo.lat_deg, b.geo.lon_deg, b.geo.lat_deg, b.geo.lon_deg, b.geo.lat_deg, b.geo.lon_deg)
    });
    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark">
<title>geomedia locate — {name}</title><style>
:root {{ --bg:#fbfaf8; --fg:#1d1d1b; --muted:#6b6a66; --line:#e3e1dc; --card:#fff; --accent:#2f5d8a; --best:#c0392b; --mirror:#b9770e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecebe8; --muted:#a09e99; --line:#2e2d2b; --card:#1e1e1c; --accent:#8ab4dc; --best:#ff7b6b; --mirror:#f5c26b; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif }} main {{ max-width:960px; margin:0 auto; padding:24px 16px 48px }}
h1 {{ font-size:22px; margin:0 0 4px; word-break:break-all }} h2 {{ font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:28px 0 8px }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; overflow-x:auto }}
table {{ border-collapse:collapse; width:100% }} th,td {{ text-align:left; padding:6px 10px 6px 0; border-bottom:1px solid var(--line) }} a {{ color:var(--accent) }}
circle.best {{ fill:none; stroke:var(--best); stroke-width:30 }} circle.mirror {{ fill:none; stroke:var(--mirror); stroke-width:24; stroke-dasharray:40 30 }} circle.top {{ fill:var(--fg); opacity:.6 }}
.sub {{ color:var(--muted) }}</style></head><body><main>
<h1>{name_e}</h1><p class="sub">geomedia locate · {verdict}</p>
<h2>Result</h2><section><table>
<tr><th>Best</th><td>{best}</td></tr>
<tr><th>Recording start</th><td>{start} (clock offset {delta:+.2} s)</td></tr>
<tr><th>90% region</th><td>{region:.2} km²{edge}</td></tr>
<tr><th>Mirror</th><td>{mirror}</td></tr></table></section>
<h2>Probability over the area</h2><section>{svg}<p class="sub">Coarse 200 m cells; solid ring = best, dashed = mirror, dots = top 5. North is up.</p></section>
<h2>Doppler events</h2><section><table><tr><th>Event</th><th>CPA time</th><th>Speed</th><th>CPA distance</th><th>f0</th><th>Matched aircraft</th></tr>{rows}</table></section>
<footer class="sub" style="margin-top:32px">Model: {model}. {weather}. Aircraft from {source} ({n} candidates). A located result is evidence, not proof.</footer>
</main></body></html>
"#,
        name = e(&name),
        name_e = e(&name),
        verdict = e(&verdict_text(&o.verdict)),
        start = origin.plus_seconds(o.delta_s).to_rfc3339(),
        delta = o.delta_s,
        region = o.region90_km2,
        edge = if o.on_boundary {
            " — best is at the edge of --area"
        } else {
            ""
        },
        mirror = o.mirror.as_ref().map_or("—".into(), |m| format!(
            "{:.6}, {:.6}",
            m.geo.lat_deg, m.geo.lon_deg
        )),
        model = e(&wind_line(o)),
        weather = e(weather),
        n = aircraft.len(),
        source = e(source),
    )
}

/// Per-event evidence at one point: gated matches, curve scores, and the nearest aircraft passes.
fn debug_cell(inp: &Inputs, spec: &str) -> Result<()> {
    let (la, lo) = spec.split_once(',').context("--debug-cell lat,lon")?;
    let (la, lo): (f64, f64) = (la.trim().parse()?, lo.trim().parse()?);
    let (clat, clon) = inp.area.centre();
    let frame = gm_core::LocalFrame::new(gm_core::Geodetic {
        lat_deg: clat,
        lon_deg: clon,
        h_m: 0.0,
    });
    let p = frame.to_enu(gm_core::geodetic_to_ecef(gm_core::Geodetic {
        lat_deg: la,
        lon_deg: lo,
        h_m: 0.0,
    }));
    let cell = gm_search::grid::cell_at(&frame, p.x, p.y, &|a, b| (inp.ground)(a, b));
    eprintln!("debug cell {la},{lo}: mic h_ell {:.1} m", cell.geo.h_m);
    let range = (-inp.when.minus_s, inp.when.plus_s);
    for e in inp.events {
        let f = &e.fit;
        eprintln!(
            "event #{} tc {:.2} v {:.1}±{:.1} d {:.0}±{:.0} f0 {:.1} rms {:.2} pts {}",
            e.id,
            f.tc,
            f.v,
            f.sigma_v,
            f.d,
            f.sigma_d,
            f.f0,
            f.rms_hz,
            e.points.len()
        );
        for m in gm_search::matches(e, &cell, inp.aircraft, range, inp.atmos) {
            let c = gm_search::curve_ll(
                e,
                &m,
                &cell,
                &inp.aircraft[m.aircraft],
                m.delta_s,
                inp.atmos,
            );
            eprintln!(
                "   MATCH {} δ {:+.2} slant {:.0} gs {:.1} ll {:.2} curve {:?}",
                m.icao, m.delta_s, m.slant_m, m.speed_mps, m.ll, c
            );
        }
        for m in gm_search::curve_matches(e, &cell, inp.aircraft, range, inp.atmos) {
            eprintln!(
                "   CURVE {} δ {:+.2} slant {:.0} gs {:.1} ll {:.2}",
                m.icao, m.delta_s, m.slant_m, m.speed_mps, m.ll
            );
        }
        // Nearest passes regardless of gates.
        let mut near: Vec<(f64, String, f64, f64)> = Vec::new();
        for a in inp.aircraft {
            for tr in &a.tracks {
                if let Some(w) = tr.window(f.tc - 150.0, f.tc + 90.0)
                    && let Some(c) = inp.atmos.cpa(&w, cell.ecef)
                    && (c.t_arrive - f.tc).abs() < 30.0
                {
                    near.push((
                        c.t_arrive - f.tc,
                        format!("{} {:?}", a.icao, a.type_code),
                        c.slant_m,
                        c.ground_speed_mps,
                    ));
                }
            }
        }
        near.sort_by(|x, y| x.0.abs().total_cmp(&y.0.abs()));
        for (d, who, s, g) in near.iter().take(4) {
            eprintln!("   near {who} δ {d:+.2} slant {s:.0} gs {g:.1}");
        }
    }
    Ok(())
}

/// The ADS-B window for decoy aircraft: GUIDE_DECOY_WINDOW_S either side of `t0`..`t1`.
fn decoy_window(t0: Epoch, t1: Epoch) -> (Epoch, Epoch) {
    let w = gm_search::params::GUIDE_DECOY_WINDOW_S;
    (t0.plus_seconds(-w), t1.plus_seconds(w))
}

/// `cands` without any aircraft that `keep_out` has (by ICAO), order kept.
fn without(cands: Vec<Candidate>, keep_out: &[Candidate]) -> Vec<Candidate> {
    cands
        .into_iter()
        .filter(|c| keep_out.iter().all(|k| k.icao != c.icao))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_marked_event_windows_are_never_gated() {
        let mut a = Args {
            file: "a.wav".into(),
            area: "40.8840,-73.5110~5km".into(),
            when: "2026-09-08T20:40:00Z±5s".into(),
            yes: false,
            windows: vec![],
            out: None,
            json: None,
            sim_truth: None,
            events: None,
            debug_cell: None,
            weather: Weather::None,
            guided: true,
            track_first: true,
            blind_when: None,
            traffic_shift_days: 0,
            truth: None,
            envelope: true,
            envelope_only: false,
            pass_gate: true,
        };
        assert!(gates_to_passes(&a));
        a.windows = vec!["0:12-0:41".into()];
        assert!(!gates_to_passes(&a));
        a.windows.clear();
        a.pass_gate = false;
        assert!(!gates_to_passes(&a));
    }

    #[test]
    fn decoy_window_pads_three_hours() {
        let t0 = Epoch::from_rfc3339("2026-09-08T20:17:29Z").unwrap();
        let t1 = t0.plus_seconds(3600.0);
        let (w0, w1) = decoy_window(t0, t1);
        assert_eq!(t0.seconds_since(w0), 10_800.0);
        assert_eq!(w1.seconds_since(t1), 10_800.0);
    }

    #[test]
    fn decoys_exclude_prior_candidates() {
        let c = |icao: &str| Candidate {
            icao: icao.into(),
            type_code: None,
            class: gm_search::AircraftClass::Prop,
            tracks: vec![],
        };
        let out = without(vec![c("a"), c("b"), c("c"), c("d")], &[c("b"), c("d")]);
        let ids: Vec<&str> = out.iter().map(|x| x.icao.as_str()).collect();
        assert_eq!(ids, ["a", "c"]);
    }

    #[test]
    fn weather_failure_falls_back() {
        let frame = gm_core::LocalFrame::new(gm_core::Geodetic {
            lat_deg: 40.88,
            lon_deg: -73.52,
            h_m: 0.0,
        });
        let err = Err(gm_sources::SourceError::Http {
            url: "u".into(),
            reason: "offline".into(),
        });
        let (atm, sigma, note) = choose_atmosphere(err, &frame, -32.0);
        assert!(matches!(atm, gm_search::atmos::Atmosphere::Uniform(_)));
        assert_eq!(sigma, gm_search::params::WIND_SIGMA_FREE);
        assert!(
            note.contains("unavailable") && note.contains("still air"),
            "{note}"
        );
        let json = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/weather/openmeteo-pl.json"
        ))
        .unwrap();
        let hour = Epoch::from_rfc3339("2026-09-08T20:17:29Z").unwrap();
        let p = gm_sources::weather::parse_openmeteo(&json, hour);
        let (atm, sigma, note) = choose_atmosphere(p, &frame, -32.0);
        assert!(matches!(atm, gm_search::atmos::Atmosphere::Layered(_)));
        assert_eq!(sigma, gm_search::params::WIND_SIGMA_FORECAST);
        assert!(
            note.contains("Open-Meteo") && note.contains("2026-09-08T20:00:00Z"),
            "{note}"
        );
    }

    #[test]
    fn locate_returns_outcome_for_sim() {
        let dir = tempfile::tempdir().unwrap();
        let scen = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sim/two_props.toml");
        let s = gm_sim::load(&scen).unwrap();
        let r = gm_sim::render(&s).unwrap();
        let wav = dir.path().join("a.wav");
        gm_sim::write_wav(&wav, s.sample_rate, &r.samples).unwrap();
        let truth = dir.path().join("truth.json");
        std::fs::write(&truth, serde_json::to_vec(&r.truth).unwrap()).unwrap();
        let res = locate(&Args {
            file: wav,
            area: "40.8840,-73.5110~5km".into(),
            when: "2026-09-08T20:40:00Z±5s".into(),
            yes: false,
            windows: vec![],
            out: None,
            json: None,
            sim_truth: Some(truth),
            events: None,
            debug_cell: None,
            weather: Weather::OpenMeteo,
            // Both passes are discovered: guidance must add no duplicate.
            guided: true,
            track_first: false,
            blind_when: None,
            traffic_shift_days: 0,
            truth: None,
            envelope: true,
            envelope_only: false,
            pass_gate: false,
        })
        .unwrap();
        assert_eq!(res.outcome.verdict, Verdict::Located);
        assert_eq!(res.events.len(), 2);
        assert_eq!(res.aircraft.len(), 2);
    }

    #[test]
    fn locate_runs_both_legs_for_sim() {
        let dir = tempfile::tempdir().unwrap();
        let scen = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sim/two_props.toml");
        let s = gm_sim::load(&scen).unwrap();
        let r = gm_sim::render(&s).unwrap();
        let wav = dir.path().join("a.wav");
        gm_sim::write_wav(&wav, s.sample_rate, &r.samples).unwrap();
        let truth = dir.path().join("truth.json");
        std::fs::write(&truth, serde_json::to_vec(&r.truth).unwrap()).unwrap();
        let res = locate(&Args {
            file: wav,
            area: "40.8840,-73.5110~5km".into(),
            when: "2026-09-08T20:40:00Z±5s".into(),
            yes: false,
            windows: vec![],
            out: None,
            json: None,
            sim_truth: Some(truth),
            events: None,
            debug_cell: None,
            weather: Weather::OpenMeteo,
            guided: true,
            track_first: true,
            blind_when: None,
            traffic_shift_days: 0,
            truth: None,
            envelope: true,
            envelope_only: false,
            pass_gate: false,
        })
        .unwrap();
        assert_eq!(res.blind.verdict, Verdict::Located);
        let tf = res.track_first.as_ref().expect("track-first ran");
        // The simulator's two aircraft are all the traffic there is: no null can be built, so
        // track-first abstains and the combined answer is the blind one.
        assert_ne!(
            tf.outcome.verdict,
            Verdict::Located,
            "{:?}",
            tf.outcome.verdict
        );
        assert_eq!(res.outcome.verdict, Verdict::Located);
        assert_eq!(res.source, gm_search::trackfirst::Source::Blind);
    }

    fn when(s: &str) -> When {
        When::parse(s).unwrap()
    }

    #[test]
    fn blind_window_prefers_given_then_narrow_then_metadata() {
        let day = when("2026-09-08T00:00:00Z/2026-09-09T00:00:00Z");
        let narrow = when("2026-09-08T20:10:00Z±60s");
        let ct = Epoch::from_rfc3339("2026-09-08T20:20:10Z").unwrap();
        // Given wins.
        assert_eq!(
            blind_window(&day, Some(narrow), Some(ct), 603.0),
            Some(narrow)
        );
        // A narrow --when is itself the window.
        assert_eq!(blind_window(&narrow, None, Some(ct), 603.0), Some(narrow));
        // A day: from the creation time, start ∈ [ct − dur − 60, ct + 60].
        let w = blind_window(&day, None, Some(ct), 603.0).unwrap();
        assert!((w.start().seconds_since(ct) + 663.0).abs() < 1e-6);
        assert!((w.end().seconds_since(ct) - 60.0).abs() < 1e-6);
        // A day and no metadata: none.
        assert_eq!(blind_window(&day, None, None, 603.0), None);
    }

    #[test]
    fn traffic_window_pads_three_hours_and_spans_days() {
        let w = when("2026-09-08T23:00:00Z/2026-09-09T00:30:00Z");
        let (a, b) = traffic_window(&w, 600.0);
        let t0 = Epoch::from_rfc3339("2026-09-08T23:00:00Z").unwrap();
        let t1 = Epoch::from_rfc3339("2026-09-09T00:30:00Z").unwrap();
        assert!((t0.seconds_since(a) - 10_800.0 - 120.0).abs() < 1e-6);
        assert!((b.seconds_since(t1) - 10_800.0 - 600.0).abs() < 1e-6);
        let days = days_between(a, b);
        assert_eq!(
            days.iter().map(|d| d.to_string()).collect::<Vec<_>>(),
            ["2026-09-08", "2026-09-09"]
        );
    }

    #[test]
    fn oversized_area_skips_track_first() {
        let big = gm_sources::area::Area::Circle {
            lat: 40.88,
            lon: -73.55,
            radius_m: 40_000.0,
        };
        assert!(
            track_first_allowed(&big)
                .unwrap_err()
                .contains("area too large")
        );
        let city = gm_sources::area::Area::Circle {
            lat: 40.88,
            lon: -73.55,
            radius_m: 8_000.0,
        };
        assert!(track_first_allowed(&city).is_ok());
    }

    #[test]
    fn envelope_runs_when_track_first_is_refused_for_area() {
        let refused: std::result::Result<(), String> = Err("area too large".into());
        assert!(load_traffic(&refused, true));
        assert!(!load_traffic(&refused, false));
        assert!(load_traffic(&Ok(()), false));
    }

    /// The broadband fixture through `locate` (simulator truth as traffic).
    fn broadband_locate() -> LocateResult {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sim/broadband/single.toml");
        let s = gm_sim::load(&p).unwrap();
        let r = gm_sim::render(&s).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("audio.wav");
        gm_sim::write_wav(&wav, s.sample_rate, &r.samples).unwrap();
        let truth = dir.path().join("truth.json");
        std::fs::write(&truth, serde_json::to_vec(&r.truth).unwrap()).unwrap();
        locate(&Args {
            file: wav,
            // ~450 km²: track-first is refused for the area, the envelope leg still runs.
            area: "40.8765,-73.5250~12km".into(),
            when: "2026-09-08T20:10:00Z±60s".into(),
            yes: false,
            windows: vec![],
            out: None,
            json: None,
            sim_truth: Some(truth),
            events: None,
            debug_cell: None,
            weather: Weather::None,
            guided: false,
            track_first: true,
            blind_when: None,
            traffic_shift_days: 0,
            truth: None,
            envelope: true,
            envelope_only: false,
            pass_gate: false,
        })
        .unwrap()
    }

    #[test]
    fn a_pass_without_curve_feeds_timing_and_loudness() {
        let res = broadband_locate();
        assert!(
            res.track_first.is_none(),
            "track-first should be refused for the area"
        );
        let log = res.pass_log.as_ref().expect("stage 1 ran");
        assert!(
            !log.passes.is_empty(),
            "the jet was not found: {:?}",
            log.rejected
        );
        assert!(
            log.passes
                .iter()
                .any(|r| r.used_for == gm_doppler::audio::UsedFor::TimingLoudness)
        );
        let env = res.envelope.as_ref().expect("the envelope leg ran");
        assert!(
            !matches!(env.verdict, gm_search::envelope::EnvVerdict::Located { .. }),
            "one pass must not be located: {:?}",
            env.verdict
        );
        assert!(
            env.per_pass
                .iter()
                .any(|p| p.1.as_deref() == Some("a1b2c3")),
            "{:?}",
            env.per_pass
        );
    }

    #[test]
    fn every_pass_has_a_use_after_locate() {
        let res = broadband_locate();
        for r in &res.pass_log.as_ref().unwrap().passes {
            if let gm_doppler::audio::UsedFor::Unused { reason } = &r.used_for {
                assert!(
                    !reason.is_empty() && reason != "location not run",
                    "{reason}"
                );
            }
        }
    }

    #[test]
    fn json_lists_rejected_candidates_with_reasons() {
        let log = gm_doppler::audio::PassLog {
            rejected: vec![gm_doppler::audio::Rejected {
                t_start: 12.0,
                t_end: 15.0,
                reason: "too short (3.0 s < 20 s)".into(),
            }],
            ..Default::default()
        };
        let mut doc = serde_json::json!({});
        add_pass_log(&mut doc, Some(&log), None, "not run").unwrap();
        assert_eq!(doc["rejected"][0]["reason"], "too short (3.0 s < 20 s)");
        assert_eq!(doc["envelope"]["not_run"], "not run");
    }

    #[test]
    fn tone_only_sounds_reach_the_json_and_the_report() {
        let log = gm_doppler::audio::PassLog {
            tone_only: vec![gm_doppler::audio::passlog::ToneOnly {
                event_id: 2,
                t_start: 60.0,
                t_end: 90.0,
                on_rejected: Some("not sustained (loud in bursts over 25.0 s)".into()),
                used: gm_doppler::audio::UsedFor::Curve,
            }],
            ..Default::default()
        };
        let mut doc = serde_json::json!({});
        add_pass_log(&mut doc, Some(&log), None, "not run").unwrap();
        assert_eq!(doc["tone_only"][0]["event_id"], 2);
        let html = pass_html(Some(&log), None, "not run");
        assert!(html.contains("tone only (event 2)"), "{html}");
        assert!(
            html.contains("on a sound rejected as: not sustained"),
            "{html}"
        );
    }

    #[test]
    fn an_inconclusive_envelope_leaves_its_passes_unused_with_the_reason() {
        let pass = |id: usize| gm_doppler::audio::PassRecord {
            pass: gm_doppler::audio::Pass {
                id,
                t_start: 0.0,
                t_end: 30.0,
                t_peak: 15.0,
                excess_db: 6.0,
                score: 5.0,
                flags: vec![],
                hf_db: None,
            },
            curve: gm_doppler::audio::Curve::None {
                reason: "no tone".into(),
            },
            used_for: gm_doppler::audio::UsedFor::Unused {
                reason: "location not run".into(),
            },
        };
        let mut log = gm_doppler::audio::PassLog {
            passes: vec![pass(1), pass(2)],
            ..Default::default()
        };
        let env = gm_search::envelope::EnvOutcome {
            verdict: gm_search::envelope::EnvVerdict::Inconclusive {
                reason: "traffic too sparse for a null".into(),
            },
            p: 1.0,
            p_drop: None,
            delta_s: 0.0,
            per_pass: vec![],
            region: vec![],
            best: None,
            notes: vec![],
        };
        mark_uses(&mut log, &[1, 2], Some(&env), "");
        for r in &log.passes {
            match &r.used_for {
                gm_doppler::audio::UsedFor::Unused { reason } => {
                    assert!(reason.contains("traffic too sparse"), "{reason}")
                }
                u => panic!("{u:?}"),
            }
        }
        let band = gm_search::envelope::EnvOutcome {
            verdict: gm_search::envelope::EnvVerdict::Band {
                lat: 0.0,
                lon: 0.0,
                length_m: 1.0,
                width_m: 1.0,
                axis_deg: 0.0,
                mirror: false,
            },
            per_pass: vec![(1, Some("a1b2c3".into()), 3.0)],
            ..env
        };
        mark_uses(&mut log, &[1], Some(&band), "");
        assert_eq!(
            log.passes[0].used_for,
            gm_doppler::audio::UsedFor::TimingLoudness
        );
    }
}
