#!/bin/sh
set -eu
# No speed experiments here; run benchmark separately after foreground approval.
run_root=${1:-runs/reproduce}
test ! -e "$run_root" || { echo "Choose a fresh output root" >&2; exit 1; }
CARGO_BUILD_JOBS=2 cargo test --locked
CARGO_BUILD_JOBS=2 cargo build --release --locked
./target/release/highband generate --seed 42 --samples 16384 --out "$run_root/generated"
./target/release/highband train --seed 20000 --steps 400 --samples 8192 --learning-rate 0.001 --out "$run_root/train"
./target/release/highband evaluate --model "$run_root/train/model.json" --seed 100000 --count 48 --samples 8192 --out "$run_root/eval"
