//! Identify a file's format from its first bytes, never from its extension.

use crate::model::Format;

/// Classify `head` (the first 64 bytes or fewer of a file).
pub fn sniff(head: &[u8]) -> Format {
    if head.len() >= 12 && &head[4..8] == b"ftyp" {
        return if ftyp_is_heif(head) {
            Format::Heif
        } else {
            Format::Bmff
        };
    }
    if head.len() >= 12 && head.starts_with(b"RIFF") && &head[8..12] == b"WAVE" {
        return Format::Wav;
    }
    if head.len() >= 12 && head.starts_with(b"FORM") && matches!(&head[8..12], b"AIFF" | b"AIFC") {
        return Format::Aiff;
    }
    if head.starts_with(b"ID3") {
        return Format::Mp3;
    }
    if head.starts_with(b"fLaC") {
        return Format::Flac;
    }
    if head.starts_with(b"OggS") {
        return Format::Ogg;
    }
    if head.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Format::Matroska;
    }
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Format::Png;
    }
    if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Format::Jpeg;
    }
    // MPEG audio frame sync: 11 set bits, a valid layer (not 00).
    if head.len() >= 2 && head[0] == 0xFF && head[1] & 0xE0 == 0xE0 && head[1] & 0x06 != 0 {
        return Format::Mp3;
    }
    Format::Unrecognised
}

const HEIF_BRANDS: [&[u8; 4]; 5] = [b"heic", b"heix", b"mif1", b"msf1", b"avif"];

/// Whether the `ftyp` box at the start of `head` names a HEIF/AVIF brand.
fn ftyp_is_heif(head: &[u8]) -> bool {
    let size = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let end = size.clamp(12, head.len());
    let major = &head[8..12];
    // Compatible brands start after the 4-byte minor version.
    let compatible = head.get(16..end).unwrap_or(&[]);
    let (brands, _) = compatible.as_chunks::<4>();
    std::iter::once(major)
        .chain(brands.iter().map(|b| &b[..]))
        .any(|b| HEIF_BRANDS.iter().any(|h| b == &h[..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_detects_magic() {
        let cases: &[(&[u8], Format)] = &[
            (b"\0\0\0\x18ftypqt  \0\0\0\0", Format::Bmff),
            (b"RIFF\x24\0\0\0WAVEfmt ", Format::Wav),
            (b"ID3\x04\0\0", Format::Mp3),
            (&[0xFF, 0xFB, 0x90, 0x64], Format::Mp3),
            (b"fLaC\0\0\0\x22", Format::Flac),
            (b"OggS\0\x02", Format::Ogg),
            (&[0x1A, 0x45, 0xDF, 0xA3, 0x01], Format::Matroska),
            (b"FORM\0\0\0\x10AIFFCOMM", Format::Aiff),
            (&[0xFF, 0xD8, 0xFF, 0xE0], Format::Jpeg),
            (b"\x89PNG\r\n\x1a\n\0\0", Format::Png),
            (b"\0\0\0\x18ftypheic\0\0\0\0", Format::Heif),
            (b"\0\0\0\x1cftypmp42\0\0\0\0isomavif", Format::Heif),
            (b"hello", Format::Unrecognised),
            (b"", Format::Unrecognised),
        ];
        for (bytes, want) in cases {
            assert_eq!(sniff(bytes), *want, "bytes {bytes:02x?}");
        }
    }
}
