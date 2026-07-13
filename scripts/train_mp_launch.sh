#!/usr/bin/env bash
# LFHV multi-process trainer launcher: one GPU per process, host-staged NCCL.
#
# Why this exists: on the LFHV workstation (8x RTX 6000 Ada, no NVLink) the AMD
# IOMMU silently corrupts GPU-to-GPU DMA, so single-process multi-GPU XLA
# deadlocks or trains NaN. Multi-process JAX + NCCL_P2P_DISABLE=1 keeps all
# cross-GPU traffic host-staged and is verified correct
# (LFHV docs/sim_quality_log.md 坑#26 实证矩阵).
#
# Usage:
#   GPUS="1,2,3,4" LOGDIR=/path/to/logs scripts/train_mp_launch.sh <config> [args...]
# GPU ids are PCI_BUS_ID order (= nvidia-smi order). Pass OPENPI_DATA_HOME /
# HF_LEROBOT_HOME etc. through the environment as usual.
#
# NCCL_P2P_DISABLE defaults to 1 (host-staged, always safe). After the 2026-07-13
# reboot cleared the P2P corruption (sim_quality_log 坑#26 闭环), direct P2P can be
# enabled with NCCL_P2P_DISABLE=0 — run scripts in LFHV scripts/gpu_diag/ first if
# the machine has been up for long.
set -euo pipefail
GPUS="${GPUS:-1,2,3,4}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
IFS=',' read -ra ARR <<< "$GPUS"
N=${#ARR[@]}
PORT="${PORT:-29500}"
LOGDIR="${LOGDIR:-/tmp/openpi_mp_logs}"
mkdir -p "$LOGDIR"
pids=()
for i in "${!ARR[@]}"; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${ARR[$i]}" \
  NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" PYTHONUNBUFFERED=1 \
  JAX_COORDINATOR_ADDRESS="localhost:$PORT" JAX_NUM_PROCESSES="$N" JAX_PROCESS_ID="$i" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  uv run scripts/train.py "$@" > "$LOGDIR/proc$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $N processes on GPUs $GPUS (progress in $LOGDIR/proc0.log)"
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
exit $rc
