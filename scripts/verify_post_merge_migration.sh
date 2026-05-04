#!/bin/bash
#
# Post-merge verification: regenerate the per-spec InferenceData cache under
# data/specs/ and run a 1-step train_ppo to confirm the migrated example
# scripts wire end-to-end.
#
# Why this exists: master removed the JSON-driven config.json + monolithic
# data/offline/<config-hash>/ pipeline and replaced it with a per-spec
# content-addressed cache (data/specs/<spec-hash>/_manifest.json).  The
# rl_controller-branch examples were migrated to the new APIs in commits
# 4362c2dd / 6d5d3ec1 / da8b6ec1 / f1c5f3ce, but the spec cache hasn't been
# regenerated on most machines.  InferenceData.ensure() does the regen
# automatically on first call, but the extraction step needs ~32-64 GB of
# RAM, which exceeds login-node limits.  Run this script via sbatch (not on
# the login node) so the regen has enough memory.
#
# Output: a populated data/specs/ tree (one subdir per InferenceModelSpec
# cache_hash) and a confirmation that train_ppo can ingest it.
#
#SBATCH --job-name=verify_post_merge
#SBATCH --account=zhirui0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zhirui_root/zhirui0/zhirui/openg2g/slurm_verify_post_merge_%j.out

set -euo pipefail

echo "=== environment ==="
date; hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

module load python/3.10.4
source /home/zhirui/venvs/pt_mod/bin/activate

REPO_ROOT="/gpfs/accounts/zhirui_root/zhirui0/zhirui/openg2g"
cd "${REPO_ROOT}/examples/offline"

echo
echo "=== step 1: regenerate per-spec InferenceData cache ==="
echo "    (downloads from HF Hub, extracts timelines, fits ITL distributions)"
python -u -c "
import sys, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s', force=True)
sys.path.insert(0, '.')
import systems
from openg2g.controller.ofo import LogisticModelStore
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace

print(f'SPECS_CACHE_DIR = {systems.SPECS_CACHE_DIR}', flush=True)
print(f'TRAINING_TRACE_PATH = {systems.TRAINING_TRACE_PATH}', flush=True)
print(f'ALL_MODEL_SPECS = {[s.model_label for s in systems.ALL_MODEL_SPECS]}', flush=True)
print(flush=True)

inference_data = InferenceData.ensure(
    systems.SPECS_CACHE_DIR, systems.ALL_MODEL_SPECS, plot=False, dt_s=1.0,
)
print(f'InferenceData ready: {len(inference_data.models)} models', flush=True)

training_trace = TrainingTrace.ensure(systems.TRAINING_TRACE_PATH)
print(f'TrainingTrace ready: {systems.TRAINING_TRACE_PATH}', flush=True)

logistic_models = LogisticModelStore.ensure(
    systems.SPECS_CACHE_DIR, systems.ALL_MODEL_SPECS, plot=False,
)
print(f'LogisticModelStore ready: {len(systems.ALL_MODEL_SPECS)} models', flush=True)
"

echo
echo "=== step 2: 1-step train_ppo on ieee13 ==="
echo "    (verifies make_sim_factory, OfflineWorkload.replica_schedules,"
echo "     grid.attach_dc, Coordinator(datacenters=[...]) all wire end-to-end)"
SMOKE_OUT="/tmp/ppo_smoke_${SLURM_JOB_ID:-$$}"
python -u train_ppo.py --system ieee13 --total-timesteps 1 --output-dir "${SMOKE_OUT}"

echo
echo "=== step 3: 1-scenario evaluate_controllers on the just-trained model ==="
echo "    (verifies the eval pipeline: scenario randomization, baseline/OFO/PPO"
echo "     simulation, voltage/throughput metrics)"
EVAL_OUT="/tmp/eval_smoke_${SLURM_JOB_ID:-$$}"
python -u evaluate_controllers.py \
    --system ieee13 \
    --n-scenarios 1 \
    --ppo-models "${SMOKE_OUT}/ppo_model.zip" \
    --ppo-labels ppo_smoke \
    --no-per-scenario-plots \
    --no-aggregate-plots \
    --output-dir "${EVAL_OUT}"

echo
echo "=== summary ==="
echo "data/specs/ contents:"
ls -la "${REPO_ROOT}/data/specs/" | head
echo
echo "manifest count:"
find "${REPO_ROOT}/data/specs/" -name '_manifest.json' | wc -l
echo
echo "train smoke output:"
ls -la "${SMOKE_OUT}/" 2>&1 | head
echo
echo "eval smoke output:"
ls -la "${EVAL_OUT}" 2>&1 | head -20

echo
echo "verification done"
date
