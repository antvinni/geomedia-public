//! `geomedia preflight`: what a locate run over this area and time would need, before any heavy
//! work. Metadata is offered as a suggestion, never applied.

use std::path::Path;

use anyhow::{Context, Result};
use gm_core::Epoch;
use gm_media::{HintKind, HintValue};
use gm_sources::adsblol::{self, DayIndex};
use gm_sources::area::{Area, When};
use gm_sources::terrain::Terrain;

/// Sound reaching the microphone during the recording left the aircraft up to this much earlier.
pub const EMISSION_PAD_S: f64 = 120.0;
/// Search grid spacing for the cell count.
pub const CELL_M: f64 = 200.0;

pub struct Plan {
    pub area: Area,
    pub when: When,
    pub duration_s: f64,
    pub days: Vec<time::Date>,
    /// The length came from --duration because the file's couldn't be read.
    pub duration_given: bool,
}

impl Plan {
    /// UNIX window of emissions that can be heard during the recording.
    pub fn window(&self) -> (Epoch, Epoch) {
        (
            self.when.start().plus_seconds(-EMISSION_PAD_S),
            self.when.end().plus_seconds(self.duration_s),
        )
    }
}

pub fn parse_plan(
    file: Option<&Path>,
    given_duration: Option<f64>,
    area: &str,
    when: &str,
) -> Result<(Plan, Option<gm_media::MediaReport>)> {
    let Some(file) = file else {
        let duration_s = given_duration.context(
            "give the media file, or --duration (e.g. 1h, 45m, 1:30:00) to check coverage before you have it",
        )?;
        let mut plan = plan_for(area, when, duration_s)?;
        plan.duration_given = true;
        return Ok((plan, None));
    };
    let report =
        gm_media::inspect(file).with_context(|| format!("cannot read {}", file.display()))?;
    let duration_s = report
        .container
        .as_ref()
        .and_then(|c| c.duration_s)
        .or_else(|| {
            report
                .streams
                .iter()
                .filter_map(|s| s.duration_s)
                .reduce(f64::max)
        })
        .filter(|d| *d > 0.0)
        .or_else(|| {
            // Some containers (e.g. WebM/Opus) don't state a duration we can read: decode to measure.
            let a = gm_doppler::audio::decode(file).ok()?;
            Some(a.samples.len() as f64 / f64::from(a.sample_rate.max(1)))
        })
        .unwrap_or(0.0);
    let (duration_s, given) = match given_duration {
        Some(d) if duration_s <= 0.0 => (d, true),
        _ => (duration_s, false),
    };
    let mut plan = plan_for(area, when, duration_s)?;
    plan.duration_given = given;
    Ok((plan, Some(report)))
}

/// The plan for an area, a time and a recording length.
fn plan_for(area: &str, when: &str, duration_s: f64) -> Result<Plan> {
    let when = When::parse(when)?;
    let area = Area::parse(area, &mut |q| {
        let g = gm_sources::geocode::nominatim(q)?;
        eprintln!(
            "geocoded {q:?} → {} ({:.5}, {:.5})",
            gm_report::sanitize_terminal(&g.display_name),
            g.lat,
            g.lon
        );
        Ok(g)
    })?;
    let mut plan = Plan {
        area,
        when,
        duration_s,
        days: Vec::new(),
        duration_given: false,
    };
    let (a, b) = plan.window();
    let day_of = |e: Epoch| {
        time::OffsetDateTime::from_unix_timestamp(e.ns.div_euclid(1_000_000_000)).map(|t| t.date())
    };
    let (d0, d1) = (day_of(a)?, day_of(b)?);
    let mut d = d0;
    while d <= d1 {
        plan.days.push(d);
        d = d.next_day().context("date overflow")?;
    }
    Ok(plan)
}

/// Today's UTC date.
fn today_utc() -> time::Date {
    time::OffsetDateTime::now_utc().date()
}

pub fn run(
    file: Option<&Path>,
    duration: Option<f64>,
    area: &str,
    when: &str,
    yes: bool,
) -> Result<()> {
    let (plan, report) = parse_plan(file, duration, area, when)?;
    let s = gm_report::sanitize_terminal;
    match file {
        Some(f) => {
            println!(
                "File: {} ({:.1} s)",
                s(&f.display().to_string()),
                plan.duration_s
            );
            if plan.duration_given {
                println!("  note: the file's length unknown; using --duration");
            } else if duration.is_some() {
                println!("  note: --duration ignored; the file's own length is used");
            }
        }
        None => println!(
            "No media file: metadata not checked; duration {:.0} s (given)",
            plan.duration_s
        ),
    }
    for w in report.iter().flat_map(|r| &r.warnings) {
        println!("  warning: {}", s(w));
    }
    for h in report.iter().flat_map(|r| &r.hints) {
        match (&h.kind, &h.value) {
            (
                HintKind::Location,
                HintValue::Location {
                    lat_deg, lon_deg, ..
                },
            ) => {
                println!(
                    "metadata suggests --area {lat_deg:.4},{lon_deg:.4}~2km (untrusted: {})",
                    s(&h.source)
                );
            }
            (
                HintKind::CreationTime,
                HintValue::Time {
                    utc,
                    plausible: true,
                },
            ) => {
                println!(
                    "metadata suggests --when {}±10m (untrusted: {})",
                    s(utc),
                    s(&h.source)
                );
            }
            _ => {}
        }
    }
    let (lat, lon) = plan.area.centre();
    let kind = match plan.area {
        Area::Circle { .. } => "circle",
        Area::BBox { .. } => "box",
        Area::Polygon(_) => "polygon",
    };
    let cells = (plan.area.area_km2() * 1e6 / (CELL_M * CELL_M)).ceil();
    println!(
        "Area: {kind} centred {lat:.6}, {lon:.6}; {:.1} km²; {cells:.0} cells at {CELL_M:.0} m",
        plan.area.area_km2()
    );
    let (a, b) = plan.window();
    println!(
        "Time: {} → {} (emission window incl. {EMISSION_PAD_S:.0} s of propagation)",
        a.to_rfc3339(),
        b.to_rfc3339()
    );
    let mut need: Vec<String> = Vec::new();
    // Aircraft near the area across the cached days (None while any day is still missing).
    let mut aircraft: Option<usize> = Some(0);
    let mut unpublished: Vec<String> = Vec::new();
    for &d in &plan.days {
        let dir = adsblol::day_dir(d);
        match adsblol::local_parts(&dir) {
            Some(parts) => {
                let idx: DayIndex = adsblol::index_parts(d, &parts, &dir)?;
                let day = adsblol::Day { date: d, parts };
                let n = gm_sources::query::query(&idx, &day, &plan.area, a, b, a)?.len();
                aircraft = aircraft.map(|k| k + n);
                println!(
                    "ADS-B {}: cached, {n} aircraft within {:.0} km of the area during the window",
                    adsblol::date_str(d),
                    gm_sources::query::PAD_M / 1000.0
                );
            }
            None if d >= today_utc() => {
                println!(
                    "ADS-B {}: not published yet (adsb.lol publishes each UTC day after it ends)",
                    adsblol::date_str(d)
                );
                unpublished.push(adsblol::date_str(d));
                aircraft = None;
            }
            None if yes => {
                let day = adsblol::ensure_day(d, true, &mut |name, done, total| {
                    eprint!(
                        "\r  {name}: {} / {}",
                        gm_sources::human_size(done),
                        total.map_or("?".into(), gm_sources::human_size)
                    );
                })?;
                eprintln!();
                let idx = adsblol::day_index(&day)?;
                let n = gm_sources::query::query(&idx, &day, &plan.area, a, b, a)?.len();
                aircraft = aircraft.map(|k| k + n);
                println!(
                    "ADS-B {}: downloaded, {n} aircraft within {:.0} km of the area during the window",
                    adsblol::date_str(d),
                    gm_sources::query::PAD_M / 1000.0
                );
            }
            None => {
                let size = adsblol::day_assets(d)
                    .map(|v| gm_sources::human_size(v.iter().map(|x| x.2).sum()))
                    .unwrap_or_else(|e| format!("size unknown: {e}"));
                println!("ADS-B {}: not cached ({size})", adsblol::date_str(d));
                aircraft = None;
                need.push(format!("ADS-B {} ({size})", adsblol::date_str(d)));
            }
        }
    }
    let tp = Terrain::plan(&plan.area);
    if yes && (!tp.missing_tiles.is_empty() || !tp.geoid_cached) {
        Terrain::download(&plan.area, true)?;
    }
    let tp = Terrain::plan(&plan.area);
    println!(
        "terrain: {} tile{} needed, {} cached ({} assumed sea level: no GLO-30 data); geoid {}",
        tp.tiles.len(),
        if tp.tiles.len() == 1 { "" } else { "s" },
        tp.tiles.len() - tp.missing_tiles.len(),
        tp.ocean_tiles.len(),
        if tp.geoid_cached { "cached" } else { "missing" }
    );
    if !tp.missing_tiles.is_empty() {
        need.push(format!(
            "{} terrain tile(s) (~{})",
            tp.missing_tiles.len(),
            gm_sources::human_size(tp.missing_tiles.len() as u64 * (30 << 20))
        ));
    }
    if !tp.geoid_cached {
        need.push("EGM2008 geoid (81 MB)".into());
    }
    let no_coverage = aircraft == Some(0);
    if no_coverage {
        println!(
            "warning: no aircraft came within {:.0} km of the area in the window; the Doppler method needs them (see `geomedia sources --cached --area …` for other days)",
            gm_sources::query::PAD_M / 1000.0
        );
    }
    if !unpublished.is_empty() && need.is_empty() {
        println!(
            "waiting for ADS-B: {} not published yet (see `geomedia sources --cached --area …` for past days)",
            unpublished.join(", ")
        );
    } else if need.is_empty() {
        println!(
            "{}",
            if no_coverage {
                "no ADS-B coverage"
            } else {
                "ready"
            }
        );
    } else {
        println!(
            "needs downloads: {} (run with --yes to fetch)",
            need.join(", ")
        );
    }
    Ok(())
}
