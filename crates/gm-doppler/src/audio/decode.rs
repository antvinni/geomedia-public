//! Decode a media file's first audio track to mono f32. Symphonia first; for codecs it lacks
//! (e.g. Opus), fall back to an `ffmpeg` binary on PATH if one is installed.

use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};

use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

/// Rate requested from ffmpeg when it is used as the fallback decoder.
const FFMPEG_RATE: u32 = 16_000;

/// Mono audio.
#[derive(Clone, Debug)]
pub struct Audio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    /// Decoder problems that may affect timing (e.g. frames lost to corruption).
    pub warnings: Vec<String>,
}

impl Audio {
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Self {
        Self {
            sample_rate,
            samples,
            warnings: Vec::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("no audio track in this file")]
    NoAudioTrack,
    #[error("cannot decode audio: {0}")]
    Decode(String),
    #[error("cannot decode {codec}: install ffmpeg or convert to WAV")]
    NeedsFfmpeg { codec: String },
}

pub fn decode(path: &Path) -> Result<Audio, AudioError> {
    let io = |source| AudioError::Io {
        path: path.display().to_string(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    // Still images never carry audio; don't send them to the ffmpeg fallback.
    let mut head = [0u8; 64];
    let n = std::io::Read::read(&mut file, &mut head).map_err(io)?;
    if matches!(
        gm_media::sniff(&head[..n]),
        gm_media::Format::Jpeg | gm_media::Format::Png | gm_media::Format::Heif
    ) {
        return Err(AudioError::NoAudioTrack);
    }
    let file = File::open(path).map_err(io)?;
    let mut a = match decode_symphonia(file) {
        Err(Unsupported(codec)) => return decode_ffmpeg(path, &codec),
        Err(Failed(e)) => return Err(e),
        Ok(a) => a,
    };
    // MP4/MOV signal AAC priming only in the edit list, which symphonia ignores.
    if let Ok(f) = File::open(path)
        && let Ok(len) = f.metadata().map(|m| m.len())
        && let Some(priming) = gm_media::bmff::audio_priming_s(std::io::BufReader::new(f), len)
        && a.warnings.iter().all(|w| !w.starts_with("trimmed"))
    {
        let n = ((priming * f64::from(a.sample_rate)).round() as usize).min(a.samples.len());
        a.samples.drain(..n);
    }
    Ok(a)
}

enum SymFailure {
    /// Container recognised but no decoder (or no container reader) for it.
    Unsupported(String),
    Failed(AudioError),
}
use SymFailure::{Failed, Unsupported};

fn decode_symphonia(file: File) -> Result<Audio, SymFailure> {
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut reader = match symphonia::default::get_probe().probe(
        &Hint::new(),
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    ) {
        Ok(r) => r,
        Err(SymError::Unsupported(what)) => return Err(Unsupported(what.to_string())),
        Err(e) => return Err(Failed(AudioError::Decode(e.to_string()))),
    };
    let track = reader
        .default_track(TrackType::Audio)
        .or_else(|| reader.first_track(TrackType::Audio))
        .ok_or(Failed(AudioError::NoAudioTrack))?;
    let Some(CodecParameters::Audio(params)) = track.codec_params.clone() else {
        return Err(Failed(AudioError::NoAudioTrack));
    };
    let track_id = track.id;
    let time_base = track.time_base;
    let track_delay = track.delay.unwrap_or(0) as usize;
    let declared_frames = track.num_frames;
    let mut decoder = match symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
    {
        Ok(d) => d,
        Err(_) => return Err(Unsupported(params.codec.to_string())),
    };
    let mut rate = params.sample_rate.unwrap_or(0);
    let mut out: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut trimmed_any = false;
    // Timebase units → frames at the decoded rate.
    let frames_of = |ticks: u64, rate: u32| -> usize {
        match time_base {
            Some(tb) => tb
                .calc_time(symphonia::core::units::Timestamp::new(
                    ticks.min(i64::MAX as u64) as i64,
                ))
                .map_or(0, |t| {
                    (t.as_secs_f64() * f64::from(rate)).round().max(0.0) as usize
                }),
            None => ticks as usize,
        }
    };
    loop {
        let packet = match reader.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Failed(AudioError::Decode(e.to_string()))),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            // A corrupt packet is skipped; the timestamp check below fills its span with silence.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(Failed(AudioError::Decode(e.to_string()))),
        };
        rate = buf.spec().rate();
        let ch = buf.spec().channels().count().max(1);
        buf.copy_to_vec_interleaved::<f32>(&mut interleaved);
        let inv = 1.0 / ch as f32;
        let mut mono: Vec<f32> = interleaved
            .chunks_exact(ch)
            .map(|f| f.iter().sum::<f32>() * inv)
            .collect();
        let (ts, te) = (
            frames_of(packet.trim_start.get(), rate),
            frames_of(packet.trim_end.get(), rate),
        );
        if ts + te > 0 {
            trimmed_any = true;
            let end = mono.len().saturating_sub(te);
            mono = mono
                .get(ts.min(end)..end)
                .map(<[f32]>::to_vec)
                .unwrap_or_default();
        }
        // Keep the time base: pad any gap (lost or undecodable packets) up to this packet's PTS.
        let pts = packet.pts.get();
        if pts > 0 {
            let expected = frames_of(pts as u64, rate);
            if expected > out.len() && expected - out.len() < 60 * rate as usize {
                out.resize(expected, 0.0);
            }
        }
        out.extend_from_slice(&mono);
    }
    let mut warnings = Vec::new();
    // Containers that signal encoder delay only on the track rather than per packet.
    if !trimmed_any && track_delay > 0 {
        out.drain(..track_delay.min(out.len()));
        trimmed_any = true;
    }
    if trimmed_any {
        warnings.push("trimmed encoder delay".to_string());
    }
    if let Some(declared) = declared_frames.filter(|&d| d > 0) {
        let lost = declared as i64 - out.len() as i64;
        if lost > i64::from(rate) / 10 {
            warnings.push(format!(
                "missing {:.2} s of audio (decoded {} of {} declared frames, likely corruption); event times after the damage may be early",
                lost as f64 / f64::from(rate.max(1)),
                out.len(),
                declared
            ));
        }
    }
    if rate == 0 {
        return Err(Failed(AudioError::Decode("unknown sample rate".into())));
    }
    Ok(Audio {
        sample_rate: rate,
        samples: out,
        warnings,
    })
}

fn decode_ffmpeg(path: &Path, codec: &str) -> Result<Audio, AudioError> {
    decode_ffmpeg_with(Path::new("ffmpeg"), path, codec)
}

/// ffmpeg reads `proto:` prefixes from input names; force the file protocol.
pub(crate) fn ffmpeg_input(path: &Path) -> String {
    format!("file:{}", path.display())
}

pub(crate) fn decode_ffmpeg_with(
    program: &Path,
    path: &Path,
    codec: &str,
) -> Result<Audio, AudioError> {
    // `output()` drains stdout and stderr concurrently, so a chatty stderr can't deadlock us.
    let out = Command::new(program)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(ffmpeg_input(path))
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            &FFMPEG_RATE.to_string(),
            "-f",
            "f32le",
            "-",
        ])
        .stdin(Stdio::null())
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(AudioError::NeedsFfmpeg {
                codec: codec.to_string(),
            });
        }
        Err(e) => return Err(AudioError::Decode(format!("could not run ffmpeg: {e}"))),
    };
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr);
        let tail: String = msg
            .chars()
            .rev()
            .take(400)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return Err(AudioError::Decode(format!(
            "ffmpeg failed: {}",
            tail.trim()
        )));
    }
    let (frames, _) = out.stdout.as_chunks::<4>();
    let samples: Vec<f32> = frames.iter().map(|b| f32::from_le_bytes(*b)).collect();
    if samples.is_empty() {
        return Err(AudioError::NoAudioTrack);
    }
    Ok(Audio::new(FFMPEG_RATE, samples))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
    }

    /// A 16-bit PCM WAV with the given interleaved channels.
    pub(crate) fn write_pcm(path: &Path, rate: u32, channels: &[Vec<f32>]) {
        let n = channels[0].len();
        let ch = channels.len() as u16;
        let mut data = Vec::new();
        for i in 0..n {
            for c in channels {
                data.extend_from_slice(&((c[i].clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
            }
        }
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&ch.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2 * u32::from(ch)).to_le_bytes());
        out.extend_from_slice(&(2 * ch).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn decodes_wav() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.wav");
        let tone: Vec<f32> = (0..8000).map(|i| 0.5 * (i as f32 * 0.1).sin()).collect();
        write_pcm(&p, 8000, std::slice::from_ref(&tone));
        let a = decode(&p).unwrap();
        assert_eq!(a.sample_rate, 8000);
        assert_eq!(a.samples.len(), 8000);
        assert!((a.samples[100] - tone[100]).abs() < 1e-3);
    }

    #[test]
    fn stereo_downmix() {
        let dir = tempfile::tempdir().unwrap();
        let tone: Vec<f32> = (0..4000).map(|i| 0.5 * (i as f32 * 0.2).sin()).collect();
        let neg: Vec<f32> = tone.iter().map(|x| -x).collect();
        let p = dir.path().join("cancel.wav");
        write_pcm(&p, 4000, &[tone.clone(), neg]);
        let a = decode(&p).unwrap();
        assert!(a.samples.iter().all(|x| x.abs() < 1e-3));
        let q = dir.path().join("same.wav");
        write_pcm(&q, 4000, &[tone.clone(), tone.clone()]);
        let b = decode(&q).unwrap();
        assert!((b.samples[50] - tone[50]).abs() < 1e-3);
    }

    #[test]
    fn mp4_aac_decodes() {
        let a = decode(&fixture("reencoded.mp4")).unwrap();
        assert_eq!(a.sample_rate, 48000);
        assert!(
            (a.samples.len() as f64 / 48000.0 - 1.0).abs() < 0.1,
            "{}",
            a.samples.len()
        );
    }

    #[test]
    fn missing_audio_track_error() {
        assert!(matches!(
            decode(&fixture("gps.jpg")),
            Err(AudioError::NoAudioTrack | AudioError::Decode(_))
        ));
        assert!(matches!(
            decode(Path::new("/nope/x.wav")),
            Err(AudioError::Io { .. })
        ));
    }

    fn onset(a: &Audio, thresh: f32) -> f64 {
        a.samples.iter().position(|v| v.abs() > thresh).unwrap() as f64 / f64::from(a.sample_rate)
    }

    #[test]
    fn aac_priming_trimmed() {
        let a = decode(&fixture("onset.m4a")).unwrap();
        let t = onset(&a, 0.25);
        assert!((t - 1.0).abs() < 0.01, "onset {t}");
    }

    #[test]
    fn corrupt_packets_keep_time_base() {
        let a = decode(&fixture("onset_corrupt.mp3")).unwrap();
        let dur = a.samples.len() as f64 / f64::from(a.sample_rate);
        // Frames lost to resync can't be located in MP3; the loss must at least be reported.
        if (dur - 30.0).abs() >= 0.08 {
            assert!(
                a.warnings.iter().any(|w| w.starts_with("missing")),
                "duration {dur}, warnings {:?}",
                a.warnings
            );
            return;
        }
        // Skip the corrupted region (3–18 s); the clean tone starts at 20 s.
        let tail = Audio {
            sample_rate: a.sample_rate,
            samples: a.samples[(19.0 * f64::from(a.sample_rate)) as usize..].to_vec(),
            warnings: Vec::new(),
        };
        let t = 19.0 + onset(&tail, 0.25);
        assert!((t - 20.0).abs() < 0.03, "onset {t}");
    }

    #[test]
    fn ffmpeg_stderr_flood_does_not_hang() {
        let dir = tempfile::tempdir().unwrap();
        let fake = dir.path().join("fake-ffmpeg");
        // 2 MB on stderr, then 8000 float samples on stdout.
        std::fs::write(
            &fake,
            "#!/bin/sh\nhead -c 2000000 /dev/zero | tr '\\0' 'x' >&2\nhead -c 32000 /dev/zero\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let f = fake.clone();
        std::thread::spawn(move || {
            let _ = tx
                .send(decode_ffmpeg_with(&f, Path::new("x.webm"), "opus").map(|a| a.samples.len()));
        });
        let got = rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .expect("ffmpeg fallback hung");
        assert_eq!(got.unwrap(), 8000);
    }

    #[test]
    fn ffmpeg_input_is_a_file_url() {
        assert_eq!(ffmpeg_input(Path::new("cache:x.ogg")), "file:cache:x.ogg");
        assert_eq!(ffmpeg_input(Path::new("/a/b.webm")), "file:/a/b.webm");
    }
}
