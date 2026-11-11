//! Asking for --area and --when in a terminal, with the file's metadata offered as an editable
//! default (never applied without the user pressing Enter on it).

use anyhow::Result;

/// Line-based terminal I/O, so the prompts can be scripted in tests.
pub trait PromptIo {
    /// Show `prompt` and read one line; `None` at end of input.
    fn ask(&mut self, prompt: &str) -> Option<String>;
    fn say(&mut self, line: &str);
}

/// Metadata-derived defaults for the prompts.
#[derive(Debug, Clone, Default)]
pub struct Suggestion {
    pub area: Option<String>,
    pub when: Option<String>,
    /// Where they came from, e.g. "moov/udta/©xyz".
    pub source: String,
}

/// How many invalid answers before giving up.
pub const MAX_TRIES: usize = 3;

/// Ask for one value until it validates (Enter takes `default`), at most MAX_TRIES times.
fn ask_one(
    io: &mut dyn PromptIo,
    flag: &str,
    forms: &str,
    default: Option<&str>,
    source: &str,
    check: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<String> {
    let prompt = match default {
        Some(d) => {
            format!("{flag} ({forms})\n  [Enter = {d}, from metadata {source}; untrusted] > ")
        }
        None => format!("{flag} ({forms}) > "),
    };
    for _ in 0..MAX_TRIES {
        let Some(line) = io.ask(&prompt) else {
            anyhow::bail!("{flag} is required (input ended)");
        };
        let v = match (line.trim(), default) {
            ("", Some(d)) => d.to_string(),
            ("", None) => {
                io.say(&format!("  {flag} is required; try again"));
                continue;
            }
            (t, _) => t.to_string(),
        };
        match check(&v) {
            Ok(()) => return Ok(v),
            Err(e) => io.say(&format!("  {e:#}; try again")),
        }
    }
    anyhow::bail!("{flag}: no valid answer after {MAX_TRIES} tries")
}

/// `area` and `when` as given, asking for whichever is missing.
pub fn fill_missing(
    area: Option<String>,
    when: Option<String>,
    hint: &Suggestion,
    io: &mut dyn PromptIo,
    check_area: &mut dyn FnMut(&str) -> Result<()>,
    check_when: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<(String, String)> {
    let asked = area.is_none() || when.is_none();
    let area = match area {
        Some(a) => a,
        None => ask_one(
            io,
            "--area",
            "lat,lon~5km | S,W,N,E | file.geojson | country:GB | place name",
            hint.area.as_deref(),
            &hint.source,
            check_area,
        )?,
    };
    let when = match when {
        Some(w) => w,
        None => ask_one(
            io,
            "--when",
            "2026-09-08T20:17:29Z±5s | A/B, with a zone",
            hint.when.as_deref(),
            &hint.source,
            check_when,
        )?,
    };
    if asked {
        io.say(&format!("using: --area {area} --when {when}"));
    }
    Ok((area, when))
}

/// Stdin/stderr prompts.
pub struct Terminal;

impl PromptIo for Terminal {
    fn ask(&mut self, prompt: &str) -> Option<String> {
        use std::io::Write;
        eprint!("{prompt}");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line),
        }
    }
    fn say(&mut self, line: &str) {
        eprintln!("{}", gm_report::sanitize_terminal(line));
    }
}

/// Whether to prompt at all: only with a person at a terminal.
pub fn interactive() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Defaults from the file's metadata hints (as preflight prints them).
pub fn suggestion(file: Option<&std::path::Path>) -> Suggestion {
    use gm_media::{HintKind, HintValue};
    let mut s = Suggestion::default();
    let Some(report) = file.and_then(|f| gm_media::inspect(f).ok()) else {
        return s;
    };
    for h in &report.hints {
        match (&h.kind, &h.value, s.area.is_none(), s.when.is_none()) {
            (
                HintKind::Location,
                HintValue::Location {
                    lat_deg, lon_deg, ..
                },
                true,
                _,
            ) => {
                s.area = Some(format!("{lat_deg:.4},{lon_deg:.4}~2km"));
                s.source = h.source.clone();
            }
            (
                HintKind::CreationTime,
                HintValue::Time {
                    utc,
                    plausible: true,
                },
                _,
                true,
            ) => {
                s.when = Some(format!("{utc}±10m"));
                if s.source.is_empty() {
                    s.source = h.source.clone();
                }
            }
            _ => {}
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Script {
        answers: Vec<&'static str>,
        asked: Vec<String>,
        said: Vec<String>,
    }

    impl PromptIo for Script {
        fn ask(&mut self, prompt: &str) -> Option<String> {
            self.asked.push(prompt.to_string());
            if self.answers.is_empty() {
                None
            } else {
                Some(self.answers.remove(0).to_string())
            }
        }
        fn say(&mut self, line: &str) {
            self.said.push(line.to_string());
        }
    }

    fn script(answers: &[&'static str]) -> Script {
        Script {
            answers: answers.to_vec(),
            asked: vec![],
            said: vec![],
        }
    }

    fn check_area(s: &str) -> Result<()> {
        // Real parser; place names fail (no network in tests).
        gm_sources::area::Area::parse(s, &mut |q| {
            Err(gm_sources::SourceError::Invalid(format!(
                "no geocoder for {q:?}"
            )))
        })?;
        Ok(())
    }

    fn check_when(s: &str) -> Result<()> {
        gm_sources::area::When::parse(s)?;
        Ok(())
    }

    fn hint() -> Suggestion {
        Suggestion {
            area: Some("51.5072,-0.1276~2km".into()),
            when: Some("2026-09-08T20:17:29Z±10m".into()),
            source: "moov/udta/©xyz".into(),
        }
    }

    fn run(
        area: Option<&str>,
        when: Option<&str>,
        h: &Suggestion,
        io: &mut Script,
    ) -> Result<(String, String)> {
        fill_missing(
            area.map(String::from),
            when.map(String::from),
            h,
            io,
            &mut check_area,
            &mut check_when,
        )
    }

    #[test]
    fn accepts_metadata_default() {
        let mut io = script(&["", ""]);
        let (a, w) = run(None, None, &hint(), &mut io).unwrap();
        assert_eq!(a, "51.5072,-0.1276~2km");
        assert_eq!(w, "2026-09-08T20:17:29Z±10m");
        assert!(
            io.asked.iter().all(|p| p.contains("untrusted")),
            "{:?}",
            io.asked
        );
        assert!(
            io.said
                .iter()
                .any(|l| l.contains("using: --area 51.5072,-0.1276~2km --when")),
            "{:?}",
            io.said
        );
    }

    #[test]
    fn typed_values_win() {
        let mut io = script(&["40.8765,-73.525~5km", "2026-09-08T20:17:29Z±5s"]);
        let (a, w) = run(None, None, &hint(), &mut io).unwrap();
        assert_eq!(a, "40.8765,-73.525~5km");
        assert_eq!(w, "2026-09-08T20:17:29Z±5s");
    }

    #[test]
    fn invalid_then_valid() {
        let mut io = script(&[
            "nowhere",
            "40.8765,-73.525~5km",
            "yesterday",
            "2026-09-08T20:17:29Z±5s",
        ]);
        let (a, w) = run(None, None, &Suggestion::default(), &mut io).unwrap();
        assert_eq!(a, "40.8765,-73.525~5km");
        assert_eq!(w, "2026-09-08T20:17:29Z±5s");
        assert_eq!(io.asked.len(), 4);
        assert!(
            io.said.iter().filter(|l| l.contains("try again")).count() == 2,
            "{:?}",
            io.said
        );
    }

    #[test]
    fn empty_without_default_reasks_then_gives_up() {
        let mut io = script(&["", "", ""]);
        let e = run(
            None,
            Some("2026-09-08T20:17:29Z±5s"),
            &Suggestion::default(),
            &mut io,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("--area"), "{e}");
        assert_eq!(io.asked.len(), MAX_TRIES);
    }

    #[test]
    fn eof_is_error() {
        let mut io = script(&[]);
        assert!(run(None, None, &hint(), &mut io).is_err());
    }

    #[test]
    fn given_flags_not_asked() {
        let mut io = script(&[]);
        let (a, w) = run(
            Some("40.8765,-73.525~5km"),
            Some("2026-09-08T20:17:29Z±5s"),
            &hint(),
            &mut io,
        )
        .unwrap();
        assert_eq!(
            (a.as_str(), w.as_str()),
            ("40.8765,-73.525~5km", "2026-09-08T20:17:29Z±5s")
        );
        assert!(io.asked.is_empty() && io.said.is_empty());
    }
}
