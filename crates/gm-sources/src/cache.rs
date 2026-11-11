//! The local cache and resumable, atomic downloads.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::SourceError;

/// `$GEOMEDIA_CACHE`, else the platform cache directory + `geomedia`.
pub fn cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("GEOMEDIA_CACHE").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        return home.join("Library/Caches/geomedia");
    }
    std::env::var_os("XDG_CACHE_HOME")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".cache"))
        .join("geomedia")
}

/// The User-Agent sent on every request (no personal data).
pub fn user_agent() -> String {
    format!(
        "geomedia/{} (+https://github.com/antvinni/geomedia-public)",
        env!("CARGO_PKG_VERSION")
    )
}

/// Connecting, and waiting for response headers, never take longer than this (the body of a
/// multi-GB download may take as long as it needs).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

/// An HTTP agent that reports status codes instead of failing on them.
pub(crate) fn agent() -> ureq::Agent {
    agent_with(None)
}

/// `agent()` with an optional bound on the whole request (small API calls).
pub(crate) fn agent_with(global: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .user_agent(user_agent())
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .timeout_global(global)
        .build()
        .into()
}

fn io_err(path: &Path) -> impl Fn(std::io::Error) -> SourceError + '_ {
    move |source| SourceError::Io {
        path: path.display().to_string(),
        source,
    }
}

pub struct Fetch {
    pub url: String,
    pub dest: PathBuf,
    pub expected_len: Option<u64>,
    /// Bound on the whole request (small API calls); `None` for large downloads.
    pub timeout: Option<Duration>,
}

/// Download `f.url` to `f.dest`, resuming a previous partial download (`dest.part`).
pub fn fetch(
    f: &Fetch,
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<PathBuf, SourceError> {
    let http = |reason: String| SourceError::Http {
        url: f.url.clone(),
        reason,
    };
    if let Ok(m) = std::fs::metadata(&f.dest)
        && f.expected_len.is_none_or(|n| n == m.len())
    {
        return Ok(f.dest.clone());
    }
    if let Some(dir) = f.dest.parent() {
        std::fs::create_dir_all(dir).map_err(io_err(dir))?;
    }
    let mut part = f.dest.clone().into_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    let have = std::fs::metadata(&part).map_or(0, |m| m.len());
    let mut req = agent_with(f.timeout).get(&f.url);
    if have > 0 {
        req = req.header("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().map_err(|e| http(e.to_string()))?;
    let status = resp.status().as_u16();
    let (mut out, mut done) = match status {
        206 => (
            OpenOptions::new()
                .append(true)
                .open(&part)
                .map_err(io_err(&part))?,
            have,
        ),
        200 => (File::create(&part).map_err(io_err(&part))?, 0),
        416 if f.expected_len == Some(have) => {
            std::fs::rename(&part, &f.dest).map_err(io_err(&f.dest))?;
            return Ok(f.dest.clone());
        }
        s => {
            return Err(SourceError::HttpStatus {
                url: f.url.clone(),
                status: s,
            });
        }
    };
    let total = f.expected_len.or_else(|| {
        resp.headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|n| n + done)
    });
    let mut body = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = match body.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                out.flush().map_err(io_err(&part))?;
                return Err(http(format!(
                    "interrupted after {done} bytes ({e}); run again to resume"
                )));
            }
        };
        out.write_all(&buf[..n]).map_err(io_err(&part))?;
        done += n as u64;
        progress(done, total);
    }
    out.flush().map_err(io_err(&part))?;
    drop(out);
    if let Some(want) = total
        && done != want
    {
        if done > want {
            let _ = std::fs::remove_file(&part);
        }
        return Err(http(format!(
            "got {done} bytes, expected {want}; run again to resume"
        )));
    }
    std::fs::rename(&part, &f.dest).map_err(io_err(&f.dest))?;
    Ok(f.dest.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    #[test]
    fn cache_dir_env_override() {
        // SAFETY: every test that sets GEOMEDIA_* holds ENV_LOCK.
        let _g = crate::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("GEOMEDIA_CACHE");
        unsafe { std::env::set_var("GEOMEDIA_CACHE", "/tmp/gm-test-cache") };
        assert_eq!(cache_dir(), PathBuf::from("/tmp/gm-test-cache"));
        match prev {
            Some(v) => unsafe { std::env::set_var("GEOMEDIA_CACHE", v) },
            None => unsafe { std::env::remove_var("GEOMEDIA_CACHE") },
        }
        assert!(user_agent().starts_with("geomedia/0.1"));
        assert!(!user_agent().contains('@'));
    }

    #[test]
    fn stalled_server_times_out() {
        // A server that accepts and never answers must not hang a fetch with a timeout.
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/w.json", l.local_addr().unwrap());
        std::thread::spawn(move || {
            let held: Vec<_> = l.incoming().take(1).collect();
            std::thread::sleep(std::time::Duration::from_secs(30));
            drop(held);
        });
        let dir = tempfile::tempdir().unwrap();
        let t = std::time::Instant::now();
        let r = fetch(
            &Fetch {
                url,
                dest: dir.path().join("w.json"),
                expected_len: None,
                timeout: Some(std::time::Duration::from_secs(1)),
            },
            &mut |_, _| {},
        );
        assert!(r.is_err());
        assert!(t.elapsed().as_secs_f64() < 10.0, "{:?}", t.elapsed());
    }

    /// Serve `body` with Range support;    /// Serve `body` with Range support; the first `cut_first` connections stop after `cut_at` bytes.
    fn serve(
        body: Vec<u8>,
        cut_first: usize,
        cut_at: usize,
        wrong_len: bool,
    ) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file.bin", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let mut ranges = Vec::new();
            for (i, conn) in l.incoming().take(2).enumerate() {
                let mut s = conn.unwrap();
                let mut start = 0usize;
                let mut r = BufReader::new(s.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    if r.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        ranges.push(v.trim().to_string());
                        start = v.trim().trim_end_matches('-').parse().unwrap();
                    }
                }
                let rest = &body[start..];
                let len = if wrong_len {
                    rest.len() + 10
                } else {
                    rest.len()
                };
                let status = if start > 0 {
                    "206 Partial Content"
                } else {
                    "200 OK"
                };
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
                );
                let n = if i < cut_first {
                    cut_at.min(rest.len())
                } else {
                    rest.len()
                };
                let _ = s.write_all(&rest[..n]);
            }
            ranges
        });
        (url, h)
    }

    #[test]
    fn fetch_resumes() {
        let body: Vec<u8> = (0..1_048_576u32).map(|i| (i % 251) as u8).collect();
        let (url, h) = serve(body.clone(), 1, 307_200, false);
        let dir = tempfile::tempdir().unwrap();
        let f = Fetch {
            url,
            dest: dir.path().join("file.bin"),
            expected_len: Some(body.len() as u64),
            timeout: None,
        };
        assert!(
            fetch(&f, &mut |_, _| {}).is_err(),
            "first attempt is cut short"
        );
        let p = fetch(&f, &mut |_, _| {}).unwrap();
        assert_eq!(std::fs::read(p).unwrap(), body);
        assert_eq!(h.join().unwrap(), vec!["307200-".to_string()]);
    }

    #[test]
    fn fetch_length_mismatch_errors() {
        let (url, _h) = serve(vec![1, 2, 3], 0, 0, false);
        let dir = tempfile::tempdir().unwrap();
        let f = Fetch {
            url,
            dest: dir.path().join("x"),
            expected_len: Some(10),
            timeout: None,
        };
        let e = fetch(&f, &mut |_, _| {}).unwrap_err().to_string();
        assert!(e.contains("expected 10"), "{e}");
        assert!(!dir.path().join("x").exists());
    }
}
