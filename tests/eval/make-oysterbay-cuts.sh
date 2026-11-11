#!/bin/sh
# 300 s WAV cuts of the Oyster Bay hour for tests/eval/oysterbay-short.toml (media stay local).
# usage: tests/eval/make-oysterbay-cuts.sh   (needs ffmpeg)
set -eu
M="$HOME/Library/Application Support/geomedia/media/oysterbay"
mkdir -p "$M/cuts"
ffmpeg -v error -y -ss 0 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0000-ade9c0.wav"
ffmpeg -v error -y -ss 30 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0030-ab20e7.wav"
ffmpeg -v error -y -ss 141 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0141-a1286c.wav"
ffmpeg -v error -y -ss 167 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0167-aa9093.wav"
ffmpeg -v error -y -ss 221 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0221-a613e7.wav"
ffmpeg -v error -y -ss 312 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0312-a3915a.wav"
ffmpeg -v error -y -ss 382 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0382-a09590.wav"
ffmpeg -v error -y -ss 452 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0452-a189f2.wav"
ffmpeg -v error -y -ss 542 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0542-0d113a.wav"
ffmpeg -v error -y -ss 655 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0655-c08543.wav"
ffmpeg -v error -y -ss 982 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut0982-a8222c.wav"
ffmpeg -v error -y -ss 1808 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut1808-a53d80.wav"
ffmpeg -v error -y -ss 2005 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut2005-a1dfeb.wav"
ffmpeg -v error -y -ss 2930 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut2930-ace391.wav"
ffmpeg -v error -y -ss 3135 -t 300 -i "$M/raw.webm" -ac 1 "$M/cuts/cut3135-a34e01.wav"
