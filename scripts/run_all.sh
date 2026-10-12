#!/usr/bin/env bash
# End-to-end reproduction (Git Bash on Windows or Linux). Data/outputs go to $DATA_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-D:/driving-3dgs/env/python.exe}
DATA_ROOT=${DATA_ROOT:-D:/driving-3dgs}
LOG=15ec0778-826e-3ed7-9775-54fbf66997f4
D=$DATA_ROOT/data/$LOG

# 1) download (~135 MB: 160 front-centre JPEGs, all 80 lidar sweeps of the window, calibration, poses, annotations)
"$PY" scripts/download_av2.py --out "$DATA_ROOT/data" --split val --log $LOG --start 20 --num-frames 160 --lidar-every 1
cp "$D/manifest.json" results/data_manifest.json   # provenance: file list + bytes

# 2) prepare
#   full / sparse4: the original setup (init from every 2nd sweep = 40 sweeps; these include the held-out
#                   frames' nearest sweeps -> not a fully isolated held-out input protocol)
#   strict        : held-out frames' nearest sweeps reserved for evaluation; masks + train depth maps
"$PY" scripts/prepare.py --log-dir "$D" --out "$DATA_ROOT/work/full"
"$PY" scripts/prepare.py --log-dir "$D" --out "$DATA_ROOT/work/sparse4" --train-stride 4
"$PY" scripts/prepare.py --log-dir "$D" --out "$DATA_ROOT/work/strict" --strict-lidar --masks

# 3) train + evaluate (sequential so timings / VRAM are not shared)
run () {  # name work steps [train args...]
  local name=$1 work=$2 steps=$3; shift 3
  "$PY" scripts/train.py --work "$DATA_ROOT/work/$work" --out "$DATA_ROOT/runs/$name" --steps $steps "$@"
  "$PY" scripts/evaluate.py --work "$DATA_ROOT/work/$work" --run "$DATA_ROOT/runs/$name" --aux "$DATA_ROOT/work/strict" \
      --results results/runs/$name.json --figs "$DATA_ROOT/runs/$name/figs"
}
run full_7k    full    7000
run full_30k   full    30000
run sparse4_7k sparse4 7000
run strict_7k            strict 7000
run strict_7k_seed1      strict 7000 --seed 1
run strict_mask_7k       strict 7000 --mask moving
run strict_depth_7k      strict 7000 --depth-lambda 0.05
run strict_mask_depth_7k strict 7000 --mask moving --depth-lambda 0.05

# 4) web viewer (pruned .splat + viewer.html in docs/splat) and its measured quality
#    (full_7k with opacity >= 0.05; the 30k model pruned to a similar size renders worse, see README)
"$PY" scripts/export.py --run "$DATA_ROOT/runs/full_7k" --work "$DATA_ROOT/work/full" --out "$DATA_ROOT/export/full_7k" \
    --skip-ply --skip-preview --splat-dir docs/splat --splat-max 600000 --splat-min-opacity 0.05 --web-results results/web_export.json
node scripts/check_viewer.cjs "${PLAYWRIGHT_MODULE:-playwright}" results/current_viewer_check.json docs/img/current_viewer.jpg \
    --record-in results/web_export.json

# 5) off-path (laterally shifted) views: coverage + depth vs lidar, and a small figure
"$PY" scripts/offpath.py --work "$DATA_ROOT/work/strict" --runs "$DATA_ROOT/runs/strict_7k" "$DATA_ROOT/runs/strict_depth_7k" \
    --results results/offpath.json --fig docs/img/offpath.jpg

# 6) README tables <-> JSON consistency, tests, mutation check
# This reproduction writes fresh measurements, including wall-clock timings. Regenerate
# their tables before checking consistency; use a separate checkout to keep old results.
"$PY" scripts/check_readme.py --write
"$PY" scripts/check_readme.py --check
"$PY" -m pytest -q
"$PY" scripts/mutation_check.py
