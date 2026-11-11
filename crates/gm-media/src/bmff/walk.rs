//! A seek-only walker over ISO-BMFF boxes. It never reads payloads unless asked,
//! never panics on malformed sizes, and records what it had to skip as warnings.

use std::io::{Read, Seek};

/// Deepest nesting the walker will descend into.
pub const MAX_DEPTH: u8 = 32;
/// Largest payload `payload` will ever return.
pub const MAX_PAYLOAD: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxHeader {
    pub fourcc: [u8; 4],
    pub offset: u64,
    pub header_len: u8,
    /// Total size including the header.
    pub size: u64,
}

pub struct Walker<R: Read + Seek> {
    reader: R,
    file_len: u64,
    warnings: Vec<String>,
}

impl<R: Read + Seek> Walker<R> {
    pub fn new(reader: R, file_len: u64) -> Self {
        Self {
            reader,
            file_len,
            warnings: Vec::new(),
        }
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Children of the range [start, end); never reads payloads.
    pub fn children(&mut self, start: u64, end: u64, depth: u8) -> Vec<BoxHeader> {
        let mut out = Vec::new();
        if depth > MAX_DEPTH {
            self.warn(format!(
                "nesting deeper than {MAX_DEPTH} levels at offset {start}; not descending"
            ));
            return out;
        }
        if self.file_len == 0 {
            self.warn("file is empty".to_string());
            return out;
        }
        let end = end.min(self.file_len);
        let mut at = start;
        while at < end {
            if end - at < 8 {
                self.warn(format!("{} stray bytes at offset {at}", end - at));
                break;
            }
            let Some(b) = self.header_at(at, end) else {
                break;
            };
            at = b.offset + b.size;
            out.push(b);
        }
        out
    }

    /// The byte range holding a container box's children. `meta` is a full box
    /// (4 bytes of version/flags) in ISO files but a plain box in QuickTime files.
    pub fn child_range(&mut self, b: &BoxHeader) -> (u64, u64) {
        let start = b.offset + u64::from(b.header_len);
        let end = b.offset + b.size;
        if &b.fourcc == b"meta" && !self.plausible_header_at(start, end) {
            return ((start + 4).min(end), end);
        }
        (start, end)
    }

    /// Reads at most `max` payload bytes of a box.
    pub fn payload(&mut self, b: &BoxHeader, max: usize) -> Option<Vec<u8>> {
        let body = b.size.saturating_sub(u64::from(b.header_len));
        let n = body.min(max.min(MAX_PAYLOAD) as u64) as usize;
        let at = b.offset + u64::from(b.header_len);
        match self.read_at(at, n) {
            Some(v) => Some(v),
            None => {
                self.warn(format!(
                    "could not read payload of '{}' at {}",
                    fourcc_str(&b.fourcc),
                    b.offset
                ));
                None
            }
        }
    }

    /// Depth-first path lookup, e.g. `&[*b"moov", *b"udta", *b"\xa9xyz"]`.
    pub fn find(&mut self, path: &[[u8; 4]]) -> Option<BoxHeader> {
        let (mut start, mut end) = (0, self.file_len);
        let mut found = None;
        for (depth, want) in path.iter().enumerate() {
            let depth = u8::try_from(depth).unwrap_or(u8::MAX);
            let hit = self
                .children(start, end, depth)
                .into_iter()
                .find(|b| &b.fourcc == want)?;
            (start, end) = self.child_range(&hit);
            found = Some(hit);
        }
        found
    }

    fn warn(&mut self, msg: String) {
        self.warnings.push(msg);
    }

    fn read_at(&mut self, at: u64, n: usize) -> Option<Vec<u8>> {
        self.reader.seek(std::io::SeekFrom::Start(at)).ok()?;
        let mut buf = vec![0u8; n];
        self.reader.read_exact(&mut buf).ok()?;
        Some(buf)
    }

    /// Parse the box header at `at`, clamping it to `end` with a warning if it overruns.
    fn header_at(&mut self, at: u64, end: u64) -> Option<BoxHeader> {
        let Some(h) = self.read_at(at, 8) else {
            self.warn(format!("could not read box header at offset {at}"));
            return None;
        };
        let size32 = u32::from_be_bytes([h[0], h[1], h[2], h[3]]);
        let fourcc = [h[4], h[5], h[6], h[7]];
        let (header_len, size) = match size32 {
            0 => (8u8, end - at),
            1 => {
                let Some(l) = self.read_at(at + 8, 8) else {
                    self.warn(format!(
                        "truncated 64-bit size for '{}' at {at}",
                        fourcc_str(&fourcc)
                    ));
                    return None;
                };
                let large = u64::from_be_bytes([l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]]);
                (16u8, large)
            }
            n => (8u8, u64::from(n)),
        };
        if size < u64::from(header_len) {
            self.warn(format!(
                "box '{}' at offset {at} declares size {size}, smaller than its header; stopping",
                fourcc_str(&fourcc)
            ));
            return None;
        }
        let size = if size > end - at {
            self.warn(format!(
                "box '{}' at offset {at} declares size {size} past the end of its parent; truncated to {}",
                fourcc_str(&fourcc),
                end - at
            ));
            end - at
        } else {
            size
        };
        Some(BoxHeader {
            fourcc,
            offset: at,
            header_len,
            size,
        })
    }

    /// Whether the 8 bytes at `at` look like a real child box header.
    fn plausible_header_at(&mut self, at: u64, end: u64) -> bool {
        if end.saturating_sub(at) < 8 {
            return false;
        }
        let Some(h) = self.read_at(at, 8) else {
            return false;
        };
        let size = u64::from(u32::from_be_bytes([h[0], h[1], h[2], h[3]]));
        size >= 8
            && size <= end - at
            && h[4..8]
                .iter()
                .all(|c| c.is_ascii_graphic() || *c == b' ' || *c == 0xA9)
    }

    pub fn into_warnings(self) -> Vec<String> {
        self.warnings
    }
}

/// A printable rendering of a fourcc ("©xyz" for the 0xA9 prefix).
pub fn fourcc_str(f: &[u8; 4]) -> String {
    f.iter()
        .map(|&c| match c {
            0xA9 => '©',
            c if c.is_ascii_graphic() || c == b' ' => c as char,
            _ => '?',
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Cursor, Seek, SeekFrom, Write};
    use std::time::Instant;

    /// A box with a 32-bit size header.
    pub(crate) fn bx(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut v = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(fourcc);
        v.extend_from_slice(payload);
        v
    }

    fn walker(bytes: Vec<u8>) -> Walker<Cursor<Vec<u8>>> {
        let len = bytes.len() as u64;
        Walker::new(Cursor::new(bytes), len)
    }

    #[test]
    fn walks_nested() {
        let xyz = bx(b"\xa9xyz", b"\0\x12\x15\xc7+51.5072-000.1276/");
        let file = [
            bx(b"ftyp", b"qt  \0\0\0\0"),
            bx(b"moov", &bx(b"udta", &xyz)),
        ]
        .concat();
        let mut w = walker(file);
        let found = w.find(&[*b"moov", *b"udta", *b"\xa9xyz"]).expect("found");
        assert_eq!(found.fourcc, *b"\xa9xyz");
        assert_eq!(found.size, xyz.len() as u64);
        assert!(w.into_warnings().is_empty());
    }

    #[test]
    fn size_zero_runs_to_eof() {
        let mut last = 0u32.to_be_bytes().to_vec();
        last.extend_from_slice(b"mdat");
        last.extend_from_slice(&[7u8; 20]);
        let file = [bx(b"ftyp", b"isom\0\0\0\0"), last].concat();
        let len = file.len() as u64;
        let mut w = walker(file);
        let kids = w.children(0, len, 0);
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].fourcc, *b"mdat");
        assert_eq!(kids[1].size, len - kids[1].offset);
    }

    #[test]
    fn largesize_sparse_4gib() {
        let t0 = Instant::now();
        let mut f = tempfile::tempfile().unwrap();
        let ftyp = bx(b"ftyp", b"isom\0\0\0\0");
        f.write_all(&ftyp).unwrap();
        let mdat_size: u64 = (4u64 << 30) + 16;
        f.write_all(&1u32.to_be_bytes()).unwrap();
        f.write_all(b"mdat").unwrap();
        f.write_all(&mdat_size.to_be_bytes()).unwrap();
        let moov_at = ftyp.len() as u64 + mdat_size;
        f.set_len(moov_at).unwrap();
        f.seek(SeekFrom::Start(moov_at)).unwrap();
        f.write_all(&bx(b"moov", &bx(b"mvhd", &[0u8; 100])))
            .unwrap();
        let len = f.metadata().unwrap().len();
        f.seek(SeekFrom::Start(0)).unwrap();
        let mut w = Walker::new(f, len);
        let moov = w.find(&[*b"moov"]).expect("moov after 4 GiB mdat");
        assert_eq!(moov.offset, moov_at);
        assert!(t0.elapsed().as_secs_f64() < 1.0);
    }

    #[test]
    fn malformed_no_panic() {
        // size 3 (smaller than a header)
        let mut tiny = 3u32.to_be_bytes().to_vec();
        tiny.extend_from_slice(b"moov");
        // size past EOF
        let mut past = 1000u32.to_be_bytes().to_vec();
        past.extend_from_slice(b"moov");
        past.extend_from_slice(&[0u8; 8]);
        // 40-deep nesting
        let mut deep = bx(b"free", b"");
        for _ in 0..40 {
            deep = bx(b"moov", &deep);
        }
        let deep_path = [*b"moov"; 40];
        for (name, bytes, path) in [
            ("tiny", tiny, &[*b"moov"][..]),
            ("past", past, &[*b"moov", *b"udta"][..]),
            ("deep", deep, &deep_path[..]),
            ("empty", Vec::new(), &[*b"moov"][..]),
        ] {
            let mut w = walker(bytes);
            let _ = w.find(path);
            assert!(!w.into_warnings().is_empty(), "{name}: expected warnings");
        }
    }

    #[test]
    fn payload_capped() {
        let file = bx(b"free", &[9u8; 92]);
        let mut w = walker(file);
        let b = w.children(0, 100, 0)[0];
        assert_eq!(w.payload(&b, 16).unwrap(), vec![9u8; 16]);
        assert_eq!(w.payload(&b, 1000).unwrap().len(), 92);
    }
}
