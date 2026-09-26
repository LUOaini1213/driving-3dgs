#!/usr/bin/env bash
# End-to-end reproduction (Git Bash on Windows or Linux). Data/outputs go to $DATA_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-D:/driving-3dgs/env/python.exe}
DATA_ROOT=${DATA_ROOT:-D:/driving-3dgs}
LOG=15ec0778-826e-3ed7-9775-54fbf66997f4

# 1) download (~94 MB: 160 front-centre JPEGs, 40 lidar sweeps, calibration, poses)
$PY scripts/download_av2.py --out $DATA_ROOT/data --split val --log $LOG --start 20 --num-frames 160 --lidar-every 2
cp $DATA_ROOT/data/$LOG/manifest.json results/data_manifest.json   # provenance: file list + bytes

# 2) prepare: full train set (140 frames) and a sparse variant (every 4th frame -> 20 train frames)
$PY scripts/prepare.py --log-dir $DATA_ROOT/data/$LOG --out $DATA_ROOT/work/full
$PY scripts/prepare.py --log-dir $DATA_ROOT/data/$LOG --out $DATA_ROOT/work/sparse4 --train-stride 4

# 3) train + evaluate (sequential so timings / VRAM are not shared)
run () {  # name work steps
  $PY scripts/train.py --work $DATA_ROOT/work/$2 --out $DATA_ROOT/runs/$1 --steps $3
  $PY scripts/evaluate.py --work $DATA_ROOT/work/$2 --run $DATA_ROOT/runs/$1 \
      --results results/runs/$1.json --figs $DATA_ROOT/runs/$1/figs
}
run full_7k full 7000
run full_30k full 30000
run sparse4_7k sparse4 7000

# 4) export PLY + browser preview of the main model
$PY scripts/export.py --run $DATA_ROOT/runs/full_30k --work $DATA_ROOT/work/full --out $DATA_ROOT/export/full_30k

# 5) README table <-> JSON consistency
$PY scripts/check_readme.py --check
