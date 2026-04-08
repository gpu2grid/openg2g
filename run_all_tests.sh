#!/usr/bin/env bash
# Run all offline example scripts across all three IEEE systems.
# Usage: bash run_all_tests.sh 2>&1 | tee run_all_tests.log
#
# Each command is annotated with:
#   - Estimated run time (if known)
#   - Whether Gurobi is required
#
# The script exits on the first failure (set -e). Remove if you want to run all regardless.

set -e

SYSTEMS="ieee13 ieee34 ieee123"

echo "============================================="
echo " openg2g — Full Regression Test Suite"
echo " $(date)"
echo "============================================="

# ─────────────────────────────────────────────────
# 1. plot_topology (fast, no simulation)
#    SKIPPED: requires bus coordinate CSV files not yet in the repo
# ─────────────────────────────────────────────────
# for sys in $SYSTEMS; do
#     echo ""
#     echo ">>> plot_topology --system $sys"
#     # ~5 seconds each
#     python examples/offline/plot_topology.py --system "$sys"
# done

# ─────────────────────────────────────────────────
# 2. run_baseline — mode: both (runs no-tap + tap-change)
# ─────────────────────────────────────────────────
for sys in $SYSTEMS; do
    echo ""
    echo ">>> run_baseline --system $sys --mode both"
    # ~2-5 min each (1-hour sim at 0.1s resolution)
    python examples/offline/run_baseline.py --system "$sys" --mode both
done

# ─────────────────────────────────────────────────
# 3. run_ofo — mode: all (baseline no-tap, baseline tap, OFO no-tap, OFO tap)
# ─────────────────────────────────────────────────
for sys in $SYSTEMS; do
    echo ""
    echo ">>> run_ofo --system $sys --mode all"
    # ~5-15 min each (4 scenarios × 1-hour sim)
    python examples/offline/run_ofo.py --system "$sys" --mode all
done

# ─────────────────────────────────────────────────
# 4. analyze_different_controllers (baseline + rule-based + OFO)
# ─────────────────────────────────────────────────
for sys in $SYSTEMS; do
    echo ""
    echo ">>> analyze_different_controllers --system $sys"
    # ~5-10 min each (baseline + rule-based + OFO)
    python examples/offline/analyze_different_controllers.py --system "$sys"
done

# ─────────────────────────────────────────────────
# 5. sweep_ofo_parameters
# ─────────────────────────────────────────────────
for sys in $SYSTEMS; do
    echo ""
    echo ">>> sweep_ofo_parameters --system $sys"
    # ieee13: 1-D sweep, ~30-60 min
    # ieee34: 2-D sweep (auto), ~1-3 hours
    # ieee123: 2-D sweep (auto), ~2-5 hours
    python examples/offline/sweep_ofo_parameters.py --system "$sys"
done

# ─────────────────────────────────────────────────
# 6. sweep_dc_locations
# ─────────────────────────────────────────────────
for sys in $SYSTEMS; do
    echo ""
    echo ">>> sweep_dc_locations --system $sys"
    # ieee13: 1-D sweep, ~30-60 min
    # ieee34: 2-D zone-constrained, ~1-3 hours
    # ieee123: 2-D zone-constrained, ~2-5 hours
    python examples/offline/sweep_dc_locations.py --system "$sys"
done

# ─────────────────────────────────────────────────
# 7. sweep_hosting_capacities — 1d mode
# ─────────────────────────────────────────────────
for sys in $SYSTEMS; do
    echo ""
    echo ">>> sweep_hosting_capacities --system $sys --mode 1d"
    # ~30-60 min each (binary search over buses)
    python examples/offline/sweep_hosting_capacities.py --system "$sys" --mode 1d
done

# ─────────────────────────────────────────────────
# 8. sweep_hosting_capacities — 2d mode
# ─────────────────────────────────────────────────
for sys in $SYSTEMS; do
    echo ""
    echo ">>> sweep_hosting_capacities --system $sys --mode 2d"
    # ~1-5 hours each (pairwise bus sweep)
    python examples/offline/sweep_hosting_capacities.py --system "$sys" --mode 2d
done

# ─────────────────────────────────────────────────
# 9. analyze_LLM_load_shifting (ieee123 only)
# ─────────────────────────────────────────────────
echo ""
echo ">>> analyze_LLM_load_shifting --system ieee123"
# ~5-15 min (multi-DC with cross-site shifting comparison)
python examples/offline/analyze_LLM_load_shifting.py --system ieee123

# ─────────────────────────────────────────────────
# 10. optimize_pv_locations_and_capacities (Gurobi required)
#     Supports ieee34 and ieee123 only
# ─────────────────────────────────────────────────
for sys in ieee34 ieee123; do
    echo ""
    echo ">>> optimize_pv_locations_and_capacities --system $sys  [REQUIRES GUROBI]"
    # ~5-15 min each (MILP with 300s solver time limit)
    python examples/offline/optimize_pv_locations_and_capacities.py --system "$sys"
done

# ─────────────────────────────────────────────────
# 11. optimize_pv_and_dc_locations (Gurobi required)
#     Supports ieee34 and ieee123 only
# ─────────────────────────────────────────────────
for sys in ieee34 ieee123; do
    echo ""
    echo ">>> optimize_pv_and_dc_locations --system $sys  [REQUIRES GUROBI]"
    # ~10-20 min each (MILP with 600s solver time limit)
    python examples/offline/optimize_pv_and_dc_locations.py --system "$sys"
done

echo ""
echo "============================================="
echo " All tests completed successfully!"
echo " $(date)"
echo "============================================="
