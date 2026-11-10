#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1,3%100
#SBATCH --output=/project/logs/train_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=USR2@120
set -euo pipefail
cd /project
export PYTHONUNBUFFERED=1
export EXAMPLE='a b'
TMPDIR=/tmp/tslab-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export TMPDIR
export WANDB_CACHE_DIR=$TMPDIR/wandb-cache
export WANDB_CONFIG_DIR=$TMPDIR/wandb-config
if [ ! -d /snapshots/ab12cd34/src ]; then
    echo "tslab: code snapshot /snapshots/ab12cd34/src is missing; not running the live code" >&2
    exit 1
fi
export TSLAB_CODE_SNAPSHOT=/snapshots/ab12cd34
export TSLAB_CODE_SNAPSHOT_HASH=ab12cd34ef56ab12
export TSLAB_CODE_SNAPSHOT_COMMIT=c0ffee00
export TSLAB_CODE_SNAPSHOT_DIRTY=false
export TSLAB_CODE_SNAPSHOT_UNTRACKED_COPIED=false
export PYTHONPATH=/snapshots/ab12cd34/src${PYTHONPATH:+:$PYTHONPATH}
for _ in $(seq 60); do [ -s /project/jobs/0.tasks ] && break; sleep 1; done
TASK_LINE=$(sed -n "s/^${SLURM_ARRAY_TASK_ID} //p" /project/jobs/0.tasks)
if [ -z "$TASK_LINE" ]; then
    echo "no task ${SLURM_ARRAY_TASK_ID} in "/project/jobs/0.tasks >&2
    exit 1
fi
ATTEMPT=${TASK_LINE%% *}
TASK_LINE=${TASK_LINE#* }
echo "=== tslab attempt ${ATTEMPT} | job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | $(hostname) | $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
srun --unbuffered --kill-on-bad-exit=1 bash -c "exec $TASK_LINE"
