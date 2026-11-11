mod curvetest;
mod detecteval;
mod detecttune;
mod discbench;
mod eval;
mod locate;
mod preflight;
mod prompt;
mod runs;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

/// Estimate where a media recording was made.
#[derive(Parser)]
#[command(name = "geomedia", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum DataAction {
    /// Delete kept items on demand: lists them, then asks (or needs --yes). Only touches the
    /// cache and kept-data folders `geomedia data` shows.
    Clean {
        /// all, adsb, terrain, geoid, weather, types, geocode, runs, media or partial.
        what: String,
        /// One item by name, as `geomedia data` lists it: a date, a tile (N40W074), a run id.
        item: Option<String>,
        /// Only items not used for this long (90d, 12h, 4w).
        #[arg(long)]
        unused_for: Option<String>,
        /// Delete without asking.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Report a file's container, streams and embedded metadata (all untrusted).
    Meta {
        /// Media file to inspect.
        file: PathBuf,
        /// Also copy meta.html and meta.json here [default: managed run folder only; see `geomedia data`].
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Also write the JSON report to PATH, or print it to stdout instead of the summary with "-".
        #[arg(long, value_name = "PATH|-")]
        json: Option<String>,
        /// Do not write meta.html / meta.json.
        #[arg(long)]
        no_report: bool,
    },
    /// Find aircraft Doppler events (tonal passes) in a recording's audio.
    Events {
        /// Media file.
        file: PathBuf,
        /// Only search this window (seconds or mm:ss, e.g. 12:30-13:10); repeatable.
        #[arg(long = "event", value_name = "A-B")]
        windows: Vec<String>,
        /// Sound speed for the fit, m/s.
        #[arg(long, default_value_t = 343.0)]
        c: f64,
        /// Write JSON to PATH, or print it instead of the summary with "-".
        #[arg(long, value_name = "PATH|-")]
        json: Option<String>,
        /// Print the discovery log instead, for the --event windows (or the whole file): every
        /// fragment the search started from, the pass it grew into with its tracks, and why that
        /// became an event or not.
        #[arg(long)]
        explain: bool,
        /// Development: how ridges become events (pursuit, or the earlier whole-ridge stage).
        #[arg(long, hide = true, value_parser = ["pursuit", "ridges"], default_value = "pursuit")]
        stage: String,
    },
    /// Measure free discovery on simulated passes (class × slant × SNR): what it finds, how
    /// precisely, and how many false events — no ADS-B, no location search.
    DiscoverBench {
        /// Only this class (piston, turboprop, helicopter, jet-low-tone, jet-high-tone).
        #[arg(long)]
        class: Option<String>,
        /// Only the geometry nearest this slant (km).
        #[arg(long)]
        slant_km: Option<f64>,
        /// Only this SNR (dB).
        #[arg(long)]
        snr: Option<f64>,
        /// Real background audio mixed under each pass at the cell's SNR (instead of white noise).
        #[arg(long, value_name = "FILE")]
        background: Option<PathBuf>,
        /// Round-trip each cell through a lossy codec with ffmpeg: opus or aac.
        #[arg(long)]
        codec: Option<String>,
        /// Phone-style high-pass cut-off (Hz).
        #[arg(long, value_name = "HZ")]
        highpass: Option<f64>,
        /// Inject no pass: measure the false-event rate under these conditions.
        #[arg(long)]
        no_pass: bool,
    },
    /// List every data source geomedia can use, with licence, cost and cache status.
    Sources,
    /// What geomedia keeps — source data and kept runs/media: where, how big, when last used.
    Data {
        #[command(subcommand)]
        action: Option<DataAction>,
        /// Aircraft counts per cached day and terrain coverage for this area.
        #[arg(long)]
        area: Option<String>,
        /// Print JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run locate over a manifest of ground-truth cases and report coverage and accuracy.
    Eval {
        /// Manifest (TOML, [[case]] entries).
        manifest: PathBuf,
        /// Also copy results.json and summary.md here [default: managed run folder only].
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Only the cases of this split (`split = "…"` in the manifest).
        #[arg(long)]
        split: Option<String>,
        /// Stage-1 tuning sweep over the split's detect cases (prints a table; no run folder).
        #[arg(long, hide = true)]
        detect_tune: bool,
        /// Curves cases: keep only tonal events on a detected pass (measurement of gating).
        #[arg(long, hide = true)]
        pass_gate: bool,
    },
    /// Estimate where a recording was made from aircraft Doppler passes (exit 0 located, 3 not).
    Locate {
        /// Media file.
        file: PathBuf,
        /// Where it may have been recorded (see `preflight --help`).
        #[arg(long)]
        area: Option<String>,
        /// When the recording started, with a zone, ± uncertainty (see `preflight --help`).
        #[arg(long)]
        when: Option<String>,
        /// Download whatever data is missing.
        #[arg(long)]
        yes: bool,
        /// Only use events in this window (seconds or mm:ss); repeatable.
        #[arg(long = "event", value_name = "A-B")]
        windows: Vec<String>,
        /// Also copy the report here [default: managed run folder only; see `geomedia data`].
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Also write the JSON report to PATH, or print it with "-".
        #[arg(long, value_name = "PATH|-")]
        json: Option<String>,
        /// Use a `geomedia sim` truth.json as the aircraft source (testing/evaluation only).
        #[arg(long, value_name = "TRUTH.json")]
        sim_truth: Option<PathBuf>,
        /// Reuse a previous run's events.json instead of decoding the audio.
        #[arg(long, value_name = "events.json")]
        events: Option<PathBuf>,
        /// Print per-event evidence at lat,lon (diagnostics).
        #[arg(long, value_name = "LAT,LON", hide = true)]
        debug_cell: Option<String>,
        /// Weather for the propagation model.
        #[arg(long, value_enum, default_value = "open-meteo")]
        weather: locate::Weather,
        /// Skip guided scoring (each nearby aircraft's predicted curve looked for in the
        /// spectrogram; finds passes too faint for free discovery).
        #[arg(long)]
        no_guided: bool,
        /// Skip track-first scoring (every aircraft of the prior scored jointly at each place and
        /// time; with only the day known, it is the leg that searches the whole day).
        #[arg(long)]
        no_track_first: bool,
        /// Skip stage 1 (finding aircraft sounds) and the timing-and-loudness location of
        /// sounds with no Doppler curve.
        #[arg(long)]
        no_envelope: bool,
        /// Development: locate every aircraft sound by timing and loudness only.
        #[arg(long, hide = true)]
        envelope_only: bool,
        /// Keep only tonal events that lie on a detected aircraft sound. Cuts false curves on
        /// silence, but also drops faint passes whose tone shows and whose loudness doesn't rise
        /// (Oyster Bay is then not located), so it is off by default. Ignored with --event
        /// windows or --no-envelope.
        #[arg(long)]
        pass_gate: bool,
        /// The blind leg's clock window when --when is wide, e.g. 2026-09-08T20:17:29Z±60s
        /// [default: --when if within ±10 min, else the metadata creation time (untrusted)].
        #[arg(long, value_name = "WHEN")]
        blind_when: Option<String>,
        /// Evaluation only: read ADS-B this many days away, re-timed onto the recording's day
        /// (wrong-day controls).
        #[arg(
            long,
            value_name = "DAYS",
            default_value_t = 0,
            hide = true,
            allow_hyphen_values = true
        )]
        traffic_shift_days: i64,
    },
    /// Check what a locate run needs (data, downloads, coverage) for an area and time.
    Preflight {
        /// Media file (optional: without one, give --duration to check coverage in advance).
        file: Option<PathBuf>,
        /// Where it may have been recorded: lat,lon~5km | S,W,N,E | file.geojson | country:GB | place name.
        #[arg(long)]
        area: Option<String>,
        /// When the recording started: 2026-09-08T20:17:29Z[±5s] | A/B (a zone is required).
        #[arg(long)]
        when: Option<String>,
        /// Download and index whatever is missing.
        #[arg(long)]
        yes: bool,
        /// Recording length when there's no file yet: 90 | 1:30 | 1:00:00 | 45m | 1h.
        #[arg(long, value_name = "LENGTH")]
        duration: Option<String>,
    },
    /// Render a synthetic flyover recording (audio.wav + truth.json) from a scenario TOML.
    Sim {
        /// Scenario file.
        scenario: PathBuf,
        /// Output directory.
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    // Bring the data folders to the current layout (one-off moves are reported once).
    for s in gm_sources::paths::ensure_layout() {
        eprintln!("data layout: {}", gm_report::sanitize_terminal(&s));
    }
    let result = match cli.command {
        Command::Meta {
            file,
            out,
            json,
            no_report,
        } => meta(&file, out, json.as_deref(), no_report),
        Command::Sim { scenario, out } => sim(&scenario, &out),
        Command::Sources => sources(),
        Command::DiscoverBench {
            class,
            slant_km,
            snr,
            background,
            codec,
            highpass,
            no_pass,
        } => (|| -> Result<()> {
            let background = background
                .map(|p| -> Result<std::sync::Arc<Vec<f32>>> {
                    let a = gm_doppler::audio::decode(&p)
                        .with_context(|| format!("cannot read background {}", p.display()))?;
                    Ok(std::sync::Arc::new(
                        gm_doppler::audio::decimate(&a, 8000).samples,
                    ))
                })
                .transpose()?;
            discbench::run(
                &discbench::Filter {
                    class,
                    slant_km,
                    snr,
                },
                &discbench::Conditions {
                    background,
                    codec,
                    highpass_hz: highpass,
                    no_pass,
                },
            )
        })(),
        Command::Data {
            action:
                Some(DataAction::Clean {
                    what,
                    item,
                    unused_for,
                    yes,
                }),
            ..
        } => data_clean(&what, item.as_deref(), unused_for.as_deref(), yes),
        Command::Data {
            action: None,
            area,
            json,
        } => data_list(area.as_deref(), json),
        Command::Eval {
            manifest,
            out,
            split,
            detect_tune,
            pass_gate,
        } => {
            if detect_tune {
                eval::detect_tune(&manifest, split.as_deref().unwrap_or("tuning"))
            } else {
                eval::run(&manifest, out, split.as_deref(), pass_gate)
            }
        }
        Command::Preflight {
            file,
            area,
            when,
            yes,
            duration,
        } => area_when(area, when, file.as_deref()).and_then(|(area, when)| {
            duration
                .as_deref()
                .map(parse_duration)
                .transpose()
                .and_then(|d| preflight::run(file.as_deref(), d, &area, &when, yes))
        }),
        Command::Locate {
            file,
            area,
            when,
            yes,
            windows,
            out,
            json,
            sim_truth,
            events,
            debug_cell,
            weather,
            no_guided,
            no_track_first,
            no_envelope,
            envelope_only,
            pass_gate,
            blind_when,
            traffic_shift_days,
        } => match area_when(area, when, Some(&file)).and_then(|(area, when)| {
            locate::run(&locate::Args {
                file,
                area,
                when,
                yes,
                windows,
                out,
                json,
                sim_truth,
                events,
                debug_cell,
                weather,
                guided: !no_guided,
                track_first: !no_track_first,
                blind_when,
                traffic_shift_days,
                truth: None,
                envelope: !no_envelope,
                envelope_only,
                pass_gate,
            })
        }) {
            Ok(code) => return ExitCode::from(code),
            Err(e) => Err(e),
        },
        Command::Events {
            file,
            windows,
            c,
            json,
            explain,
            stage,
        } => {
            let stage = match stage.as_str() {
                "ridges" => gm_doppler::audio::Stage::Ridges,
                _ => gm_doppler::audio::Stage::Pursuit,
            };
            events(&file, &windows, c, json.as_deref(), explain, stage)
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", gm_report::sanitize_terminal(&format!("{e:#}")));
            ExitCode::from(1)
        }
    }
}

fn meta(file: &Path, out: Option<PathBuf>, json: Option<&str>, no_report: bool) -> Result<()> {
    let report =
        gm_media::inspect(file).with_context(|| format!("cannot read {}", file.display()))?;
    let json_text = gm_report::render_json(&report);
    let json_to_stdout = json == Some("-");
    if json_to_stdout {
        print!("{json_text}");
    } else {
        print!("{}", gm_report::render_text(&report));
    }
    if let Some(path) = json.filter(|p| *p != "-") {
        std::fs::write(path, &json_text).with_context(|| format!("cannot write {path}"))?;
    }
    if !no_report {
        // A managed run folder (see `geomedia data`); `--out` also gets a copy.
        let run = runs::new_run("meta", Some(file))?;
        let out = out.as_deref();
        runs::write_both(
            &run,
            out,
            "meta.html",
            gm_report::render_html(&report).as_bytes(),
        )?;
        runs::write_both(&run, out, "meta.json", json_text.as_bytes())?;
        runs::finish_run(
            &run,
            &runs::RunInfo {
                kind: "meta".into(),
                file: file.display().to_string(),
                ..Default::default()
            },
        )?;
        let shown = gm_report::sanitize_terminal(&run.join("meta.html").display().to_string());
        let mut lines = vec![format!("report: {shown}")];
        if let Some(o) = out {
            lines.push(format!(
                "copy:   {}",
                gm_report::sanitize_terminal(&o.join("meta.html").display().to_string())
            ));
        }
        for l in lines {
            if json_to_stdout {
                eprintln!("{l}");
            } else {
                println!("{l}");
            }
        }
    }
    Ok(())
}

fn sources() -> Result<()> {
    let cache = gm_sources::cache_dir();
    println!(
        "cache: {}",
        gm_report::sanitize_terminal(&cache.display().to_string())
    );
    println!(
        "{:<14} {:<15} {:<44} {:<30} {:<18} {:<28} {:<7}",
        "id", "kind", "licence", "coverage", "cost", "status", "cached"
    );
    for s in gm_sources::registry() {
        let dir = cache.join(source_dir(s.id));
        let cached = dir_size(&dir).map_or_else(|| "no".to_string(), human_size);
        println!(
            "{:<14} {:<15} {:<44} {:<30} {:<18} {:<28} {:<7}",
            s.id, s.kind, s.licence, s.coverage, s.cost, s.status, cached
        );
    }
    Ok(())
}

/// "3 d ago", "5 h ago", "just now", or "—".
fn ago(t: Option<i64>) -> String {
    let Some(t) = t else { return "—".into() };
    let s = time::OffsetDateTime::now_utc().unix_timestamp() - t;
    match s {
        s if s < 120 => "just now".into(),
        s if s < 7200 => format!("{} min ago", s / 60),
        s if s < 172_800 => format!("{} h ago", s / 3600),
        s => format!("{} d ago", s / 86_400),
    }
}

/// `--unused-for`: `90d`, `12h`, `4w`, or a bare number of days. Returns seconds.
fn parse_age(s: &str) -> Result<i64> {
    let t = s.trim();
    let bad = || anyhow::anyhow!("--unused-for {s:?}: use e.g. 90d, 12h or 4w");
    let (n, unit) = match t.char_indices().last() {
        Some((i, c)) if c.is_ascii_alphabetic() => (&t[..i], c),
        _ => (t, 'd'),
    };
    let unit = match unit {
        'h' => 3600,
        'd' => 86_400,
        'w' => 7 * 86_400,
        _ => return Err(bad()),
    };
    let n: i64 = n.parse().map_err(|_| bad())?;
    if !(0..=100_000).contains(&n) {
        return Err(bad());
    }
    Ok(n * unit)
}

/// `geomedia data clean`: list what would go, ask (or require --yes), delete, say what was
/// freed. Only items inside the cache and kept-data folders are ever touched.
fn data_clean(what: &str, item: Option<&str>, unused_for: Option<&str>, yes: bool) -> Result<()> {
    use gm_sources::clean::{Selector, execute_clean, plan_clean};
    use std::path::PathBuf;
    let cache = gm_sources::cache_dir();
    let kept = gm_sources::paths::data_dir();
    gm_sources::clean::check_bases(
        &cache,
        &kept,
        std::env::var_os("HOME").map(PathBuf::from).as_deref(),
    )?;
    let sel = Selector::parse(what, item)?;
    let unused = unused_for.map(parse_age).transpose()?;
    let inv = gm_sources::inventory::inventory(&cache, &kept);
    let plan = plan_clean(&inv, &sel, unused)?;
    let clean = |s: &str| gm_report::sanitize_terminal(s);
    for s in &plan.skipped {
        println!("  skipped {}", clean(s));
    }
    let items = plan.items;
    if items.is_empty() {
        println!("nothing to clean");
        return Ok(());
    }
    for it in &items {
        println!(
            "  {:<44} {:>9}  {}",
            clean(&it.label),
            human_size(it.bytes),
            clean(&it.path.display().to_string())
        );
    }
    let total: u64 = items.iter().map(|i| i.bytes).sum();
    let count = |k: &str| items.iter().filter(|i| i.kind == k).count();
    let n = items.len();
    println!("{n} item(s), {}", human_size(total));
    let recordings = count("media");
    if recordings > 0 {
        println!("includes {recordings} recordings (media): kept data, not re-downloadable");
    }
    let runs = count("runs");
    if runs > 0 {
        println!("includes {runs} run result folder(s)");
    }
    if !yes {
        if !prompt::interactive() {
            println!("nothing deleted (add --yes)");
            return Ok(());
        }
        let what = if recordings > 0 {
            format!(
                "Delete {n} items ({}), including {recordings} recordings? [y/N] ",
                human_size(total)
            )
        } else {
            format!("Delete {n} items ({})? [y/N] ", human_size(total))
        };
        eprint!("{what}");
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        if !matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            println!("nothing deleted");
            return Ok(());
        }
    }
    let freed = execute_clean(&items, &[cache, kept])?;
    println!("freed {}", human_size(freed));
    Ok(())
}

/// `geomedia data`: what geomedia keeps, where, how big, when last used (and, for an area,
/// how the cached ADS-B days and terrain cover it).
fn data_list(area: Option<&str>, json: bool) -> Result<()> {
    use gm_sources::inventory::{aircraft_near, inventory};
    let cache = gm_sources::cache_dir();
    let kept = gm_sources::paths::data_dir();
    let inv = inventory(&cache, &kept);
    let area = area
        .map(|a| {
            gm_sources::area::Area::parse(a, &mut |q| gm_sources::geocode::nominatim(q))
                .with_context(|| format!("bad --area {a:?}"))
        })
        .transpose()?;
    // Each day is its own archive: count them in parallel, saying what's happening meanwhile.
    let near: Vec<Option<usize>> = match &area {
        None => vec![None; inv.adsb_days.len()],
        Some(a) => std::thread::scope(|scope| {
            let handles: Vec<_> = inv
                .adsb_days
                .iter()
                .map(|d| {
                    eprintln!("checking {} …", d.date);
                    let cache = &cache;
                    scope.spawn(move || aircraft_near(cache, &d.date, a))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        }),
    };
    let terrain = area.as_ref().map(gm_sources::terrain::Terrain::plan);
    if json {
        let mut v = serde_json::to_value(&inv)?;
        v["cache"] = serde_json::json!(cache.display().to_string());
        v["data"] = serde_json::json!(kept.display().to_string());
        if let Some(t) = &terrain {
            v["area"] = serde_json::json!({
                "aircraft_within_30km": near,
                "terrain_tiles_needed": t.tiles.len(),
                "terrain_tiles_missing": t.missing_tiles.len(),
                "terrain_tiles_ocean": t.ocean_tiles.len(),
                "geoid_cached": t.geoid_cached,
            });
        }
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    let clean = |s: &str| gm_report::sanitize_terminal(s);
    let total = |kinds: &[&str]| -> u64 {
        inv.items
            .iter()
            .filter(|i| kinds.contains(&i.kind.as_str()))
            .map(|i| i.bytes)
            .sum()
    };
    const SOURCE: [&str; 7] = [
        "adsb", "terrain", "geoid", "weather", "types", "geocode", "partial",
    ];
    const KEPT: [&str; 2] = ["media", "runs"];
    println!(
        "source data (re-downloadable): {}  {}",
        clean(&cache.display().to_string()),
        human_size(total(&SOURCE))
    );
    println!(
        "kept data:                     {}  {}",
        clean(&kept.display().to_string()),
        human_size(total(&KEPT))
    );
    if inv.items.is_empty() {
        println!("nothing kept");
        return Ok(());
    }
    let sections = [
        ("adsb", "ADS-B days (adsb.lol)"),
        ("terrain", "terrain tiles (Copernicus GLO-30)"),
        ("geoid", "geoid (EGM2008)"),
        ("weather", "weather (Open-Meteo)"),
        ("types", "aircraft types (tar1090-db)"),
        ("geocode", "geocoding (Nominatim)"),
        ("partial", "interrupted downloads"),
        ("media", "media"),
        ("runs", "runs"),
    ];
    for (kind, title) in sections {
        let items: Vec<_> = inv.items.iter().filter(|i| i.kind == kind).collect();
        if items.is_empty() {
            continue;
        }
        let sum: u64 = items.iter().map(|i| i.bytes).sum();
        println!("\n{title} — {} item(s), {}", items.len(), human_size(sum));
        for it in items {
            let cover = if kind == "adsb" {
                let n = inv
                    .adsb_days
                    .iter()
                    .position(|d| d.date == it.name)
                    .and_then(|i| near.get(i).copied().flatten());
                match (&area, n) {
                    (None, _) => String::new(),
                    (Some(_), Some(k)) => {
                        format!("  {k} aircraft within 30 km of the area (UTC day)")
                    }
                    (Some(_), None) => "  (index on first use)".into(),
                }
            } else {
                String::new()
            };
            println!(
                "  {:<34} {:>9}  {:>12}  {}{}  {}",
                clean(&it.name),
                human_size(it.bytes),
                ago(it.last_used),
                clean(&it.detail),
                cover,
                clean(&it.path.display().to_string())
            );
        }
        if kind == "terrain"
            && let Some(t) = &terrain
        {
            println!(
                "  terrain for area: {} tiles needed, {} cached, {} ocean{}",
                t.tiles.len(),
                t.tiles.len() - t.missing_tiles.len(),
                t.ocean_tiles.len(),
                if t.missing_tiles.is_empty() {
                    ""
                } else {
                    " (run preflight --yes to fetch the rest)"
                }
            );
        }
    }
    println!("\nclean up on demand with `geomedia data clean …`");
    Ok(())
}

/// Cache sub-directory used by each source.
fn source_dir(id: &str) -> &'static str {
    match id {
        "adsblol" => "adsb",
        "tar1090-types" => "types",
        "glo30" => "dem/glo30",
        "egm2008" => "geoid",
        "nominatim" => "geocode",
        "openmeteo" => "weather",
        _ => "none",
    }
}

fn dir_size(p: &Path) -> Option<u64> {
    let mut total = 0;
    let mut stack = vec![p.to_path_buf()];
    let mut any = false;
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let m = e.metadata().ok()?;
            if m.is_dir() {
                stack.push(e.path());
            } else {
                total += m.len();
                any = true;
            }
        }
    }
    any.then_some(total)
}

fn human_size(n: u64) -> String {
    match n {
        n if n < 1024 => format!("{n} B"),
        n if n >= 1 << 30 => format!("{:.1} GB", n as f64 / f64::from(1u32 << 30)),
        n if n >= 1 << 20 => format!("{:.0} MB", n as f64 / f64::from(1u32 << 20)),
        n => format!("{:.0} KB", n as f64 / 1024.0),
    }
}

/// Seconds from "93.5", "1:33.5" or "1:01:33".
fn parse_time(s: &str) -> Option<f64> {
    let mut total = 0.0;
    for part in s.split(':') {
        let v: f64 = part.trim().parse().ok()?;
        if !v.is_finite() || v < 0.0 {
            return None;
        }
        total = total * 60.0 + v;
    }
    Some(total)
}

fn parse_window(w: &str, duration: f64) -> Result<(f64, f64)> {
    let shown = gm_report::sanitize_terminal(w);
    let (a, b) = w
        .split_once('-')
        .ok_or_else(|| anyhow::anyhow!("window {shown}: expected START-END"))?;
    let (lo, hi) = match (parse_time(a), parse_time(b)) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => anyhow::bail!("window {shown}: times must be seconds or mm:ss"),
    };
    if hi <= lo {
        anyhow::bail!("window {shown}: end must be after start");
    }
    if lo >= duration {
        anyhow::bail!("window {shown}: starts after the audio ends ({duration:.1} s)");
    }
    Ok((lo, hi.min(duration)))
}

fn events(
    file: &Path,
    windows: &[String],
    c: f64,
    json: Option<&str>,
    explain: bool,
    stage: gm_doppler::audio::Stage,
) -> Result<()> {
    if !(c.is_finite() && c > 0.0) {
        anyhow::bail!("--c must be a positive sound speed");
    }
    let audio = gm_doppler::audio::decode(file)
        .with_context(|| format!("cannot read audio from {}", file.display()))?;
    let duration = audio.samples.len() as f64 / f64::from(audio.sample_rate.max(1));
    let wins = windows
        .iter()
        .map(|w| parse_window(w, duration))
        .collect::<Result<Vec<_>>>()?;
    let params = gm_doppler::audio::EventParams {
        c,
        stage,
        ..Default::default()
    };
    if explain {
        let spans = if wins.is_empty() {
            vec![(0.0, duration)]
        } else {
            wins.clone()
        };
        // The template stage dedups against the ridge stage's events (they carry no z).
        let ridge: Vec<gm_doppler::audio::Event> =
            gm_doppler::audio::discover(&audio, &params, &[])
                .into_iter()
                .filter(|e| e.guide_z.is_none())
                .collect();
        for w in spans {
            for line in gm_doppler::audio::events::explain(&audio, &params, &wins, w) {
                println!("{}", gm_report::sanitize_terminal(&line));
            }
            for line in gm_doppler::audio::templates::explain_templates(&audio, &ridge, w) {
                println!("{}", gm_report::sanitize_terminal(&line));
            }
        }
        return Ok(());
    }
    let found = gm_doppler::audio::discover(&audio, &params, &wins);
    // Stage 1 over the whole file (it needs the background around each pass).
    let det = gm_doppler::audio::detect(&audio, &gm_doppler::audio::DetectParams::default());
    let log = gm_doppler::audio::passlog::build(&det, &found);
    let doc = serde_json::json!({
        "schema_version": 2,
        "file": file.display().to_string(),
        "c": c,
        "duration_s": duration,
        "windows": wins,
        "events": found,
        "passes": log.passes,
        "rejected": log.rejected,
        "events_outside_passes": log.events_outside_passes,
        "pass_notes": log.notes,
    });
    let mut text = serde_json::to_string_pretty(&doc)?;
    text.push('\n');
    if let Some(path) = json.filter(|p| *p != "-") {
        std::fs::write(path, &text).with_context(|| format!("cannot write {path}"))?;
    }
    if json == Some("-") {
        print!("{text}");
        return Ok(());
    }
    if log.passes.is_empty() {
        println!("no aircraft sounds found");
    }
    for line in log.lines() {
        println!("{}", gm_report::sanitize_terminal(&line));
    }
    if found.is_empty() {
        println!("no Doppler events found");
    }
    for e in &found {
        let f = &e.fit;
        println!(
            "#{}  tc {:.2} s  v {:.1}±{:.1} m/s  d {:.0}±{:.0} m  f0 {:.1} Hz (k={})  rms {:.2} Hz  snr {:.0} dB  {:?}",
            e.id,
            f.tc,
            f.v,
            f.sigma_v,
            f.d,
            f.sigma_d,
            e.f0_fundamental,
            e.harmonic,
            f.rms_hz,
            e.mean_snr_db,
            gm_doppler::audio::source_class(e)
        );
    }
    Ok(())
}

fn sim(scenario: &Path, out: &Path) -> Result<()> {
    let s = gm_sim::load(scenario)?;
    let r = gm_sim::render(&s)?;
    std::fs::create_dir_all(out).with_context(|| format!("cannot create {}", out.display()))?;
    let wav = out.join("audio.wav");
    gm_sim::write_wav(&wav, s.sample_rate, &r.samples)
        .with_context(|| format!("cannot write {}", wav.display()))?;
    let truth = out.join("truth.json");
    let mut json = serde_json::to_string_pretty(&r.truth)?;
    json.push('\n');
    std::fs::write(&truth, json).with_context(|| format!("cannot write {}", truth.display()))?;
    let shown = |p: &Path| gm_report::sanitize_terminal(&p.display().to_string());
    println!(
        "wrote {} ({} s, {} Hz), truth: {}",
        shown(&wav),
        s.duration_s,
        s.sample_rate,
        shown(&truth)
    );
    Ok(())
}

/// `--area` and `--when` as given, or asked for in a terminal (metadata offered as the
/// default, never applied unasked); outside a terminal a missing one is an error.
fn area_when(
    area: Option<String>,
    when: Option<String>,
    file: Option<&Path>,
) -> Result<(String, String)> {
    if let (Some(a), Some(w)) = (&area, &when) {
        return Ok((a.clone(), w.clone()));
    }
    if !prompt::interactive() {
        let missing: Vec<&str> = [("--area", area.is_none()), ("--when", when.is_none())]
            .iter()
            .filter(|(_, m)| *m)
            .map(|(f, _)| *f)
            .collect();
        anyhow::bail!(
            "{} {} required (or run in a terminal to be asked)",
            missing.join(" and "),
            if missing.len() == 1 { "is" } else { "are" }
        );
    }
    let hint = prompt::suggestion(file);
    prompt::fill_missing(
        area,
        when,
        &hint,
        &mut prompt::Terminal,
        &mut |a| {
            gm_sources::area::Area::parse(a, &mut |q| gm_sources::geocode::nominatim(q))?;
            Ok(())
        },
        &mut |w| {
            gm_sources::area::When::parse(w)?;
            Ok(())
        },
    )
}

/// Longest recording `--duration` accepts (a preflight spanning more days is a mistake).
const MAX_DURATION_S: f64 = 24.0 * 3600.0;

/// A recording length: seconds ("90"), "m:ss", "h:mm:ss", or "45m" / "1.5h" / "30s".
fn parse_duration(s: &str) -> Result<f64> {
    let t = s.trim();
    let bad = || {
        anyhow::anyhow!("--duration {s:?}: use seconds, m:ss, h:mm:ss, 45m or 1.5h (at most 24 h)")
    };
    let num = |x: &str| -> Result<f64> {
        let v: f64 = x.parse().map_err(|_| bad())?;
        if v.is_finite() && v >= 0.0 {
            Ok(v)
        } else {
            Err(bad())
        }
    };
    let v = if let Some(n) = t.strip_suffix('h') {
        num(n)? * 3600.0
    } else if let Some(n) = t.strip_suffix('m') {
        num(n)? * 60.0
    } else if let Some(n) = t.strip_suffix('s') {
        num(n)?
    } else if t.contains(':') {
        let parts: Vec<f64> = t.split(':').map(num).collect::<Result<_>>()?;
        // Fields after the first are minutes or seconds of the clock: below 60.
        if parts.len() > 3 || parts.len() < 2 || parts[1..].iter().any(|x| *x >= 60.0) {
            return Err(bad());
        }
        parts.iter().fold(0.0, |acc, x| acc * 60.0 + x)
    } else {
        num(t)?
    };
    if v > 0.0 && v <= MAX_DURATION_S {
        Ok(v)
    } else {
        Err(bad())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn locate_pass_gate(argv: &[&str]) -> bool {
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Locate { pass_gate, .. } => pass_gate,
            _ => panic!("not a locate command"),
        }
    }

    // Off by default: with it on, Oyster Bay goes from located 276 m to inconclusive
    // (docs/evaluation.md, "Gating tonal events to passes").
    #[test]
    fn locate_does_not_gate_tonal_events_unless_asked() {
        assert!(!locate_pass_gate(&["geomedia", "locate", "a.wav"]));
        assert!(locate_pass_gate(&[
            "geomedia",
            "locate",
            "a.wav",
            "--pass-gate"
        ]));
    }

    #[test]
    fn parse_age_forms() {
        for (s, v) in [
            ("90d", 90 * 86_400),
            ("12h", 43_200),
            ("4w", 28 * 86_400),
            ("7", 7 * 86_400),
        ] {
            assert_eq!(parse_age(s).unwrap(), v, "{s}");
        }
        for s in ["", "d", "-1d", "3y", "1.5d", "999999d"] {
            assert!(parse_age(s).is_err(), "{s}");
        }
    }

    #[test]
    fn parse_duration_forms() {
        for (s, v) in [
            ("90", 90.0),
            ("1:30", 90.0),
            ("1:00:00", 3600.0),
            ("30m", 1800.0),
            ("1h", 3600.0),
            ("1.5h", 5400.0),
            (" 90 ", 90.0),
            ("45s", 45.0),
            ("90:00", 5400.0),
            ("24h", 86_400.0),
        ] {
            assert_eq!(parse_duration(s).unwrap(), v, "{s}");
        }
        for s in [
            "", "-5", "abc", "1:2:3:4", "0", "NaN", "inf", "1e9", "25h", "1:-30", "0:90", "1:75:00",
        ] {
            assert!(parse_duration(s).is_err(), "{s}");
        }
    }
}
