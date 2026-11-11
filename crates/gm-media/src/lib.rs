//! Media intake: format sniffing and embedded metadata extraction.

pub mod audio;
pub mod bmff;
pub mod exif;
pub mod file;
pub mod iso6709;
pub mod model;
pub mod sniff;
pub mod traces;
pub mod when;

pub use file::file_info;
pub use iso6709::parse_iso6709;
pub use model::*;
pub use sniff::sniff;

use std::path::Path;

/// Inspect `path`: sniff its format from content, extract everything it says about itself.
/// Only failing to open or read the file is an error; unknown or malformed content is a report.
pub fn inspect(path: &Path) -> std::io::Result<MediaReport> {
    use std::io::{BufReader, Read};

    let info = file_info(path)?;
    let len = info.size_bytes;
    let mut report = MediaReport::new(info);
    let mut head = Vec::with_capacity(64);
    std::fs::File::open(path)?.take(64).read_to_end(&mut head)?;
    report.format = sniff(&head);
    let (container, streams, hints, warnings) = match report.format {
        Format::Bmff => bmff::extract_bmff(BufReader::new(std::fs::File::open(path)?), len),
        Format::Jpeg | Format::Png | Format::Heif => {
            let (h, w) = exif::extract_exif(BufReader::new(std::fs::File::open(path)?));
            (None, Vec::new(), h, w)
        }
        Format::Wav
        | Format::Mp3
        | Format::Flac
        | Format::Ogg
        | Format::Matroska
        | Format::Aiff => audio::extract_audio(path, report.format),
        Format::Unrecognised => (None, Vec::new(), Vec::new(), Vec::new()),
    };
    report.container = container;
    report.streams = streams;
    report.hints = hints;
    report.warnings = warnings;
    Ok(report)
}

#[cfg(test)]
mod inspect_tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
    }

    #[test]
    fn inspect_wrong_extension() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.wav");
        std::fs::copy(fixture("phone_gps.mov"), &p).unwrap();
        let r = inspect(&p).unwrap();
        assert_eq!(r.format, Format::Bmff);
        assert!(r.hints.iter().any(|h| h.kind == HintKind::Location));
        assert_eq!(r.streams.len(), 2);
    }

    #[test]
    fn inspect_unrecognised() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("hello.bin");
        std::fs::write(&p, b"hello world").unwrap();
        let r = inspect(&p).unwrap();
        assert_eq!(r.format, Format::Unrecognised);
        assert!(r.is_metadata_empty());
        assert_eq!(r.file.size_bytes, 11);
    }

    #[test]
    fn inspect_dispatches_each_family() {
        for (name, want) in [
            ("tagged.flac", Format::Flac),
            ("gps.jpg", Format::Jpeg),
            ("plain.wav", Format::Wav),
        ] {
            let r = inspect(&fixture(name)).unwrap();
            assert_eq!(r.format, want, "{name}");
        }
        assert!(
            inspect(&fixture("gps.jpg"))
                .unwrap()
                .hints
                .iter()
                .any(|h| h.kind == HintKind::Location)
        );
        assert!(!inspect(&fixture("tagged.flac")).unwrap().streams.is_empty());
        assert!(inspect(&fixture("plain.wav")).unwrap().is_metadata_empty());
    }

    #[test]
    fn inspect_missing_file_is_err() {
        assert!(inspect(Path::new("/definitely/not/here.mov")).is_err());
    }
}
