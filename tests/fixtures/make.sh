#!/bin/sh
# Regenerates the committed test fixtures. Needs ffmpeg and python3 (developer tools only; CI uses the committed files).
set -eu
cd "$(dirname "$0")"
FF="ffmpeg -hide_banner -loglevel error -y"

# Phone-style QuickTime with Apple mdta keys and a udta ©xyz location.
$FF -f lavfi -i testsrc=size=64x64:rate=1 -f lavfi -i sine=frequency=440:sample_rate=48000 -t 1 \
  -c:v mpeg4 -c:a aac -movflags use_metadata_tags \
  -metadata com.apple.quicktime.location.ISO6709="+51.5072-000.1276+011.000/" \
  -metadata com.apple.quicktime.make=Apple \
  -metadata com.apple.quicktime.model="iPhone 15" \
  -metadata com.apple.quicktime.creationdate=2026-05-14T10:02:00-0700 \
  phone_gps.mov
$FF -i phone_gps.mov -c copy -metadata location="+51.5072-000.1276/" -f mov phone_udta.tmp
mv phone_udta.tmp phone_udta.mov

# The same clip with all metadata stripped (what a re-upload often looks like).
$FF -f lavfi -i testsrc=size=64x64:rate=1 -f lavfi -i sine=frequency=440:sample_rate=48000 -t 1 \
  -c:v mpeg4 -c:a aac -map_metadata -1 reencoded.mp4

# Audio-only files.
$FF -f lavfi -i sine=frequency=440:sample_rate=48000 -t 1 -c:a libmp3lame -b:a 32k \
  -metadata date=2026-05-14 -metadata encoded_by=TestRig tagged.mp3
$FF -f lavfi -i sine=frequency=440:sample_rate=48000 -t 1 \
  -metadata date=2026-05-14 -metadata encoded_by=TestRig tagged.flac
$FF -f lavfi -i sine=frequency=440:sample_rate=8000 -t 1 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact -flags:a +bitexact plain.wav

# Broadcast WAV: plain.wav with an EBU Tech 3285 bext chunk inserted after "WAVE".
python3 -I - <<'PY'
import struct
d = open("plain.wav", "rb").read()
desc = b"".ljust(256, b"\0")
originator = b"TestRig".ljust(32, b"\0")
ref = b"".ljust(32, b"\0")
bext = desc + originator + ref + b"2026-05-14" + b"10:02:00" + struct.pack("<QH", 0, 1) + b"\0" * 64 + b"\0" * 190
chunk = b"bext" + struct.pack("<I", len(bext)) + bext
body = d[12:]
out = b"RIFF" + struct.pack("<I", 4 + len(chunk) + len(body)) + b"WAVE" + chunk + body
open("bwf.wav", "wb").write(out)
PY

# Images: an 8x8 PNG without EXIF, and a JPEG with a known EXIF block (GPS 51.5072 N, 0.1276 W, 11 m).
$FF -f lavfi -i color=c=gray:size=8x8 -frames:v 1 nogps.png
$FF -f lavfi -i color=c=gray:size=8x8 -frames:v 1 plain.jpg
python3 -I - <<'PY'
import struct

def ifd(entries, next_off, base):
    """entries: list of (tag, type, count, value_bytes). Returns bytes with data area appended."""
    n = len(entries)
    head = struct.pack(">H", n)
    data = b""
    data_off = base + 2 + n * 12 + 4
    body = b""
    for tag, typ, count, val in sorted(entries):
        if len(val) <= 4:
            body += struct.pack(">HHI", tag, typ, count) + val.ljust(4, b"\0")
        else:
            body += struct.pack(">HHII", tag, typ, count, data_off + len(data))
            data += val + (b"\0" if len(val) % 2 else b"")
    return head + body + struct.pack(">I", next_off) + data

def ascii(s): return s.encode() + b"\0"
def rats(*pairs): return b"".join(struct.pack(">II", a, b) for a, b in pairs)
ASCII, SHORT, LONG, RAT, BYTE = 2, 3, 4, 5, 1

# Layout: TIFF header (8) | IFD0 | EXIF IFD | GPS IFD. Offsets are from the TIFF header.
make, model = ascii("TestCam"), ascii("T1")
ifd0_len = lambda e: 2 + len(e) * 12 + 4 + sum(len(v) + len(v) % 2 for *_, v in e if len(v) > 4)
ifd0_entries = [(0x010F, ASCII, len(make), make), (0x0110, ASCII, len(model), model),
                (0x8769, LONG, 1, struct.pack(">I", 0)), (0x8825, LONG, 1, struct.pack(">I", 0))]
exif_entries = [(0x9003, ASCII, 20, ascii("2026:05:14 10:02:00")), (0x9011, ASCII, 7, ascii("-07:00"))]
gps_entries = [(0x0001, ASCII, 2, b"N\0"), (0x0002, RAT, 3, rats((51, 1), (30, 1), (2592, 100))),
               (0x0003, ASCII, 2, b"W\0"), (0x0004, RAT, 3, rats((0, 1), (7, 1), (3936, 100))),
               (0x0005, BYTE, 1, b"\0"), (0x0006, RAT, 1, rats((11, 1)))]
exif_at = 8 + ifd0_len(ifd0_entries)
gps_at = exif_at + ifd0_len(exif_entries)
ifd0_entries[2] = (0x8769, LONG, 1, struct.pack(">I", exif_at))
ifd0_entries[3] = (0x8825, LONG, 1, struct.pack(">I", gps_at))
tiff = b"MM\0*" + struct.pack(">I", 8)
tiff += ifd(ifd0_entries, 0, 8) + ifd(exif_entries, 0, exif_at) + ifd(gps_entries, 0, gps_at)
app1 = b"Exif\0\0" + tiff
seg = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
jpg = open("plain.jpg", "rb").read()
open("gps.jpg", "wb").write(jpg[:2] + seg + jpg[2:])
PY
rm plain.jpg

# Time-base fixtures: a 300 Hz tone that starts exactly at 1.000 s (AAC) / 20.000 s (MP3, then corrupted).
$FF -f lavfi -i "aevalsrc=if(gte(t\,1)\,0.5*sin(2*PI*300*t)\,0):s=8000:d=3" -c:a aac -b:a 32k onset.m4a
$FF -f lavfi -i "aevalsrc=if(gte(t\,20)\,0.5*sin(2*PI*300*t)\,0):s=8000:d=30" -c:a libmp3lame -b:a 16k onset.mp3
python3 -I - <<'PY'
d = bytearray(open("onset.mp3", "rb").read())
lo, hi = len(d) // 10, len(d) * 6 // 10
for i in range(lo, hi, 97):
    d[i] ^= 0xFF
open("onset_corrupt.mp3", "wb").write(bytes(d))
PY
rm onset.mp3

# A 3 s Opus/WebM tone (the codec YouTube audio uses; decoded via the ffmpeg fallback).
$FF -f lavfi -i "sine=frequency=300:sample_rate=48000" -t 3 -c:a libopus -b:a 24k opus.webm
