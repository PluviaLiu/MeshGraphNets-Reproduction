#!/bin/bash
# Driver for the MeshGraphNets reproduction.
#
#   bash run.sh smoke    # official demo: flag_minimal, 10 steps -> eval -> plot
#   bash run.sh train    # flag_simple training (default 1M steps)
#   bash run.sh eval     # rollouts + MSE curve + GIF from the trained checkpoint
#   bash run.sh plot     # xvfb-run the official plot_cloth.py on an existing pkl
#
# Everything runs against the official, unmodified meshgraphnets sources. The
# only additions are mgn_shim.py (TF2 compat) and analyze_rollout.py (renders
# the artifacts the official plot script never saves).

set -euo pipefail

ROOT=/data/HOI/LiuSiyu/MeshGraphNets
REPO="$ROOT/deepmind-research"
DATA="$ROOT/data"
PY="$ROOT/env/bin/python"
STEPS="${STEPS:-1000000}"
WORKERS="${WORKERS:-12}"

# GPU 3 is the idle A30; 0/1/2 are 74-93% busy with other users' jobs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
# Reduce allocator fragmentation contention on a shared card.
export TF_GPU_ALLOCATOR=cuda_malloc_async

cd "$REPO"   # meshgraphnets/ must be importable as a namespace package

case "${1:-}" in
  smoke)
    bash "$ROOT/fetch_dataset.py" flag_minimal "$DATA" >/dev/null 2>&1 || \
      python3 "$ROOT/fetch_dataset.py" flag_minimal "$DATA"
    $PY -m meshgraphnets.run_model --model=cloth --mode=train \
        --checkpoint_dir="$DATA/chk_min" --dataset_dir="$DATA/flag_minimal" \
        --num_training_steps=10
    $PY -m meshgraphnets.run_model --model=cloth --mode=eval \
        --checkpoint_dir="$DATA/chk_min" --dataset_dir="$DATA/flag_minimal" \
        --rollout_path="$DATA/rollout_min.pkl" --num_rollouts=1
    $PY "$ROOT/analyze_rollout.py" --rollout_path="$DATA/rollout_min.pkl" \
        --mse_out="$DATA/mse_min.png" --gif_out="$DATA/rollout_min.gif"
    ;;

  train)
    $PY -m meshgraphnets.run_model --model=cloth --mode=train \
        --checkpoint_dir="$DATA/chk_simple" --dataset_dir="$DATA/flag_simple" \
        --num_training_steps="$STEPS"
    ;;

  eval)
    $PY -m meshgraphnets.run_model --model=cloth --mode=eval \
        --checkpoint_dir="$DATA/chk_simple" --dataset_dir="$DATA/flag_simple" \
        --rollout_path="$DATA/rollout_flag.pkl" --num_rollouts=10
    $PY "$ROOT/analyze_rollout.py" --rollout_path="$DATA/rollout_flag.pkl" \
        --mse_out="$DATA/mse_flag.png" --gif_out="$DATA/rollout_flag.gif"
    ;;

  plot)
    # The official script only calls plt.show(block=True) -- it saves nothing,
    # so it needs a display. xvfb-run gives it one.
    xvfb-run -a $PY -m meshgraphnets.plot_cloth \
        --rollout_path="${2:-$DATA/rollout_flag.pkl}"
    ;;

  *)
    echo "usage: bash run.sh {smoke|train|eval|plot [rollout.pkl]}" >&2
    exit 2
    ;;
esac
