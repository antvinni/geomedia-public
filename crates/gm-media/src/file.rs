//! Filesystem facts about the input: size, content hash, mtime.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::model::FileInfo;

const CHUNK: usize = 1 << 20;

/// Size, streamed SHA-256 and filesystem mtime of `path`.
pub fn file_info(path: &Path) -> std::io::Result<FileInfo> {
    let mut file = File::open(path)?;
    let meta = file.metadata()?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let fs_modified_utc = meta
        .modified()
        .ok()
        .and_then(|t| OffsetDateTime::from(t).format(&Rfc3339).ok());
    Ok(FileInfo {
        path: path.display().to_string(),
        size_bytes: meta.len(),
        sha256: hex::encode(hasher.finalize()),
        fs_modified_utc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha256_streams() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"abc").unwrap();
        let info = file_info(f.path()).unwrap();
        assert_eq!(
            info.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(info.size_bytes, 3);
        assert!(info.fs_modified_utc.is_some());
    }
}
