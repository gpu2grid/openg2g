"""Joint PV + DC Location Co-Optimisation via Sensitivity-Based MILP.

Extends PV expansion planning to simultaneously optimise:
  - PV placement: binary x[j], continuous capacity s[j], curtailment
  - DC placement: binary y[k,b] assigning each DC site to a candidate bus
  - Tap co-optimisation: integer delta_tap variables

Uses OpenDSS-derived voltage sensitivity matrices and representative
operating scenarios.  A Gurobi MILP minimises annualised investment cost
minus operational savings plus voltage-violation penalty and switching
cost, subject to linearised voltage constraints.

Formulation
-----------
Sets:
    B_pv  candidate buses for PV placement
    B_dc  candidate buses for DC placement
    K     DC sites (to be assigned to buses)
    R     voltage regulators
    I     monitored bus-phase pairs
    T     time steps within each scenario
    S     scenarios

Decision variables:
    x[j]   in {0,1}      PV placement at bus j
    s[j]   >= 0           installed PV capacity (kW 3-phase) at bus j
    y[k,b] in {0,1}       assign DC site k to bus b (within its zone)
    c[j,t,s] >= 0         curtailed PV power (kW)
    dt[r,s,h] in Z        tap change for regulator r, scenario s, hour h
    slack_o/u >= 0         voltage violation slacks

Objective:
    min  C_inv * sum_j(s[j])
       - annual_savings(s - c, scenarios, TOU_price)
       + C_viol * sum_{i,t,s} w_s * (slack_o + slack_u)
       + C_switch * sum tap_switches

Constraints:
    sum_j(x[j]) <= N_pv
    s[j] <= S_max * x[j]
    c[j,t,s] <= s[j] * pv_frac[t,s]
    sum_b y[k,b] == 1  for each DC site k (assignment)
    sum_k y[k,b] <= 1  for each bus b   (mutual exclusion)
    v[i,t,s] = v0[i] + PV_effect + DC_effect + tap_effect + load_shift
    v[i,t,s] in [v_min - slack_u, v_max + slack_o]

Usage:
    python optimize_pv_and_dc_locations.py --config config_ieee34_pv_optimization.json --system ieee34
    python optimize_pv_and_dc_locations.py --config config_ieee123.json --system ieee123 --n-pv 5
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from optimize_pv_locations_and_capacities import (
    MILPResult,
    Scenario,
    TAP_STEP,
    _DC_DEMAND_ARCHETYPES,
    compute_load_sensitivities,
    compute_pv_sensitivities,
    compute_tap_sensitivities,
    generate_scenarios,
    plot_sensitivity_heatmap,
    precompute_load_v_shift,
    validate_with_opendss,
)
from sweep_dc_locations import DCSiteConfig, SweepConfig, discover_candidate_buses

logger = logging.getLogger("pv_dc_coopt")


# ── CoOptMILPResult ──────────────────────────────────────────────────────────


@dataclass
class CoOptMILPResult:
    """Result of the joint PV + DC co-optimisation MILP."""

    status: str
    objective: float
    pv_locations: list[str]
    pv_capacities_kw: list[float]
    dc_assignments: dict[str, str]  # site_id -> assigned bus
    investment_cost: float
    operational_savings: float
    violation_penalty: float
    switching_cost: float
    curtailment_mwh: float
    solve_time_s: float
    all_x: dict[str, float]
    all_s: dict[str, float]
    all_y: dict[tuple[str, str], float]  # (site_id, bus) -> y value
    tap_schedules: dict[str, dict[int, dict[str, int]]] | None = None


# ── DC demand profile computation ───────────────────────────────────────────


def compute_dc_demand_profiles(
    dc_site_configs: dict[str, DCSiteConfig],
    model_gpu_map: dict[str, int],
    scenarios: list[Scenario],
    typical_gpu_w: float = 300.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute time-varying DC demand for each site.

    Cycles through ``_DC_DEMAND_ARCHETYPES`` for each site.  For each site,
    computes the peak demand (base infrastructure + GPU inference) and scales
    the archetype profile accordingly.

    Returns:
        dc_demand_kw: shape (n_sites, T, n_scenarios), kW per phase
        site_ids: ordered list of site IDs
    """
    site_ids = list(dc_site_configs.keys())
    n_sites = len(site_ids)
    if n_sites == 0:
        raise ValueError("No DC sites provided")

    # Use the first scenario to determine T (all scenarios should have same T)
    T = len(scenarios[0].hours)
    n_sc = len(scenarios)
    hours = scenarios[0].hours

    dc_demand_kw = np.zeros((n_sites, T, n_sc))

    for k, site_id in enumerate(site_ids):
        site_cfg = dc_site_configs[site_id]

        # Compute total GPU count for this site
        if site_cfg.total_gpu_capacity is not None:
            total_gpus = site_cfg.total_gpu_capacity
        else:
            total_gpus = sum(
                model_gpu_map.get(m, 0) for m in (site_cfg.models or [])
            )

        gpu_kw_per_phase = total_gpus * typical_gpu_w / 1000.0 / 3.0
        peak_kw_per_phase = site_cfg.base_kw_per_phase + gpu_kw_per_phase

        # Select archetype by cycling
        archetype = _DC_DEMAND_ARCHETYPES[k % len(_DC_DEMAND_ARCHETYPES)]

        # Compute normalised profile (same across scenarios, but demand
        # level can vary by scenario via load multiplier — we keep it
        # simple and use the same profile for all scenarios)
        profile = np.array([archetype(h) for h in hours])

        for sc_idx in range(n_sc):
            dc_demand_kw[k, :, sc_idx] = peak_kw_per_phase * profile

        logger.info(
            "DC site %s: %d GPUs, peak %.0f kW/ph (base %.0f + GPU %.0f), "
            "archetype %d (%s)",
            site_id, total_gpus, peak_kw_per_phase,
            site_cfg.base_kw_per_phase, gpu_kw_per_phase,
            k % len(_DC_DEMAND_ARCHETYPES),
            archetype.__name__,
        )

    return dc_demand_kw, site_ids


# ── DC sensitivity computation ──────────────────────────────────────────────


def compute_dc_sensitivities(
    case_dir: Path,
    master_file: str,
    dc_candidate_buses: list[str],
    v_index: list[tuple[str, str]],
    *,
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    bus_kv: float = 24.9,
    perturbation_kw: float = 100.0,
) -> np.ndarray:
    """Compute voltage sensitivity to DC load at each candidate bus.

    This is a wrapper around ``compute_load_sensitivities`` — DC loads
    are positive loads (consuming power), so the sensitivity is dv/dP_load.

    Returns:
        H_dc: shape (n_monitored, n_dc_candidates) — dv per kW per phase (pu/kW)
    """
    return compute_load_sensitivities(
        case_dir=case_dir,
        master_file=master_file,
        load_buses=dc_candidate_buses,
        v_index=v_index,
        source_pu=source_pu,
        initial_taps=initial_taps,
        bus_kv=bus_kv,
        perturbation_kw=perturbation_kw,
    )


# ── MILP formulation ────────────────────────────────────────────────────────


def solve_pv_dc_milp(
    pv_candidate_buses: list[str],
    dc_candidate_buses: list[str],
    dc_site_ids: list[str],
    dc_zone_indices: dict[str, list[int]],  # site_id -> indices into dc_candidate_buses
    dc_demand_kw: np.ndarray,  # (n_sites, T, n_scenarios) kW per phase
    H_pv: np.ndarray,
    H_dc: np.ndarray,  # (n_mon, n_dc_cand) pu/kW per phase
    v0: np.ndarray,
    v_index: list[tuple[str, str]],
    scenarios: list[Scenario],
    *,
    # PV params
    n_pv: int = 5,
    s_max_kw: float = 2000.0,
    s_total_max_kw: float | None = None,
    # Voltage limits
    v_min: float = 0.95,
    v_max: float = 1.05,
    # Cost params
    c_inv: float = 1.0,
    c_viol: float = 1000.0,
    time_limit_s: float = 600.0,
    mip_gap: float = 0.02,
    # Tap params
    H_tap: np.ndarray | None = None,
    regulator_names: list[str] | None = None,
    initial_tap_ints: dict[str, int] | None = None,
    tap_range: tuple[int, int] = (-16, 16),
    max_tap_delta: int = 8,
    days_per_year: float = 365.0,
    c_switch: float = 0.0,
    load_v_shift: list[np.ndarray] | None = None,
    # Zone constraints
    pv_zones: dict[str, list[str]] | None = None,
    max_pv_per_zone: int = 1,
) -> CoOptMILPResult:
    """Formulate and solve the joint PV + DC co-optimisation MILP.

    Args:
        pv_candidate_buses: bus names for PV placement (columns of H_pv)
        dc_candidate_buses: union of all DC candidate buses (columns of H_dc)
        dc_site_ids: ordered list of DC site identifiers
        dc_zone_indices: site_id -> list of column indices into dc_candidate_buses
        dc_demand_kw: time-varying DC demand (n_sites, T, n_scenarios), kW/phase
        H_pv: PV sensitivity matrix (n_mon x n_pv_cand), pu/kW per phase
        H_dc: DC load sensitivity matrix (n_mon x n_dc_cand), pu/kW per phase
        v0: base voltages (n_mon,)
        v_index: (bus, phase) pairs for rows
        scenarios: list of operating scenarios
        n_pv: max number of PV sites
        s_max_kw: max PV capacity per site (kW, 3-phase)
        s_total_max_kw: total PV capacity budget (kW)
        v_min, v_max: voltage limits (pu)
        c_inv: annualised investment cost per kW ($/kW/year)
        c_viol: violation penalty per pu slack
        time_limit_s: solver time limit
        mip_gap: acceptable MIP gap
        H_tap: tap sensitivity matrix
        regulator_names: regulator names
        initial_tap_ints: initial tap positions as integers
        tap_range: absolute tap limits (min, max)
        max_tap_delta: max tap change from initial
        days_per_year: for annualising savings
        c_switch: tap switching cost ($/switch)
        load_v_shift: precomputed load voltage shifts
        pv_zones: zone constraints for PV
        max_pv_per_zone: max PV per zone
    """
    import gurobipy as gp
    from gurobipy import GRB

    n_mon = len(v_index)
    n_pv_cand = len(pv_candidate_buses)
    n_dc_cand = len(dc_candidate_buses)
    n_sites = len(dc_site_ids)
    n_regs = len(regulator_names) if regulator_names else 0

    model = gp.Model("PV_DC_CoOpt")
    model.Params.TimeLimit = time_limit_s
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1
    model.Params.MIPFocus = 1  # focus on finding good feasible solutions

    # ── PV decision variables ──
    x = model.addVars(n_pv_cand, vtype=GRB.BINARY, name="x")
    s = model.addVars(n_pv_cand, lb=0.0, ub=s_max_kw, name="s")

    for j in range(n_pv_cand):
        model.addConstr(s[j] <= s_max_kw * x[j], name=f"pv_link_{j}")

    model.addConstr(
        gp.quicksum(x[j] for j in range(n_pv_cand)) <= n_pv, name="n_pv"
    )

    # Total PV capacity budget
    if s_total_max_kw is not None:
        model.addConstr(
            gp.quicksum(s[j] for j in range(n_pv_cand)) <= s_total_max_kw,
            name="s_total_max",
        )

    # Per-zone PV count limit
    if pv_zones is not None:
        cand_lower = [b.lower() for b in pv_candidate_buses]
        for zid, zone_buses in pv_zones.items():
            zone_set = {b.lower() for b in zone_buses}
            zone_indices = [j for j, b in enumerate(cand_lower) if b in zone_set]
            if zone_indices:
                model.addConstr(
                    gp.quicksum(x[j] for j in zone_indices) <= max_pv_per_zone,
                    name=f"pv_zone_{zid}",
                )
                logger.info(
                    "  PV zone %s: %d candidates, max %d PV",
                    zid, len(zone_indices), max_pv_per_zone,
                )

    # ── DC placement decision variables ──
    y: dict[tuple[int, int], gp.Var] = {}
    for k, site_id in enumerate(dc_site_ids):
        for b in dc_zone_indices[site_id]:
            y[(k, b)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y_{site_id}_{dc_candidate_buses[b]}",
            )

    # DC assignment: each site to exactly one bus
    for k, site_id in enumerate(dc_site_ids):
        model.addConstr(
            gp.quicksum(y[(k, b)] for b in dc_zone_indices[site_id]) == 1,
            name=f"dc_assign_{site_id}",
        )

    # DC mutual exclusion: at most one DC per bus
    for b in range(n_dc_cand):
        sites_at_b = [
            k for k in range(n_sites) if b in dc_zone_indices[dc_site_ids[k]]
        ]
        if len(sites_at_b) > 1:
            model.addConstr(
                gp.quicksum(y[(k, b)] for k in sites_at_b) <= 1,
                name=f"dc_excl_{dc_candidate_buses[b]}",
            )

    # ── Tap decision variables (2-hour resolution) ──
    TAP_INTERVAL_H = 2
    N_TAP_SLOTS = 24 // TAP_INTERVAL_H  # 12 slots per day
    delta_tap: dict[tuple[int, int, int], gp.Var] = {}
    if n_regs > 0 and initial_tap_ints is not None:
        for sc_idx in range(len(scenarios)):
            for h in range(N_TAP_SLOTS):
                for r, reg_name in enumerate(regulator_names):
                    init = initial_tap_ints[reg_name]
                    abs_lb = tap_range[0] - init
                    abs_ub = tap_range[1] - init
                    lb = max(abs_lb, -max_tap_delta)
                    ub = min(abs_ub, max_tap_delta)
                    delta_tap[(r, sc_idx, h)] = model.addVar(
                        vtype=GRB.INTEGER, lb=lb, ub=ub,
                        name=f"dtap_{reg_name}_{sc_idx}_{h}",
                    )

    # ── Precompute DC voltage sensitivity expressions ──
    # dc_sens_expr[(k, i)] = sum_b H_dc[i, b] * y[k, b]
    # This is the effective sensitivity for site k at monitored bus-phase i,
    # weighted by the placement decision.
    dc_sens_expr: dict[tuple[int, int], gp.LinExpr] = {}
    for k, site_id in enumerate(dc_site_ids):
        for i in range(n_mon):
            dc_sens_expr[(k, i)] = gp.quicksum(
                H_dc[i, b] * y[(k, b)] for b in dc_zone_indices[site_id]
            )

    # ── Curtailment variables and voltage constraints ──
    curtail: dict[tuple[int, int, int], gp.Var] = {}
    all_slack_o: list[tuple[gp.Var, float]] = []
    all_slack_u: list[tuple[gp.Var, float]] = []

    for sc_idx, sc in enumerate(scenarios):
        T = len(sc.hours)
        for t in range(T):
            pv_frac = sc.pv_profile[t]

            # Skip timesteps with no PV AND no tap variables AND no DC effect
            # DC always has an effect, so we only skip if truly nothing changes.
            if pv_frac < 1e-6 and n_regs == 0 and n_sites == 0:
                continue

            # Create curtailment variables (only when PV active)
            if pv_frac >= 1e-6:
                for j in range(n_pv_cand):
                    c_var = model.addVar(lb=0.0, name=f"curt_{j}_{sc_idx}_{t}")
                    curtail[(j, sc_idx, t)] = c_var
                    model.addConstr(
                        c_var <= s[j] * pv_frac,
                        name=f"curt_ub_{j}_{sc_idx}_{t}",
                    )

            slack_o = model.addVars(n_mon, lb=0.0, name=f"so_{sc_idx}_{t}")
            slack_u = model.addVars(n_mon, lb=0.0, name=f"su_{sc_idx}_{t}")

            # Map timestep to 2-hour tap slot
            tap_hour = min(int(sc.hours[t]) // TAP_INTERVAL_H, N_TAP_SLOTS - 1)

            for i in range(n_mon):
                # PV voltage effect (using actual output after curtailment)
                if pv_frac >= 1e-6:
                    pv_effect = gp.quicksum(
                        H_pv[i, j]
                        * (s[j] * pv_frac - curtail[(j, sc_idx, t)])
                        / 3.0
                        for j in range(n_pv_cand)
                    )
                else:
                    pv_effect = 0

                # DC effect: sum over sites of demand * sensitivity expression
                # dc_demand_kw[k, t, sc_idx] is kW per phase (scalar for this t, sc)
                # dc_sens_expr[(k, i)] is the effective sensitivity (LinExpr of y vars)
                dc_effect = gp.quicksum(
                    dc_demand_kw[k, t, sc_idx] * dc_sens_expr[(k, i)]
                    for k in range(n_sites)
                )

                # Tap voltage effect
                if n_regs > 0:
                    tap_effect = gp.quicksum(
                        H_tap[i, r] * delta_tap[(r, sc_idx, tap_hour)]
                        for r in range(n_regs)
                    )
                else:
                    tap_effect = 0

                # Load voltage shift (known constant)
                load_shift = (
                    load_v_shift[sc_idx][t, i] if load_v_shift is not None else 0.0
                )

                v_expr = v0[i] + pv_effect + dc_effect + tap_effect + load_shift

                model.addConstr(
                    v_expr >= v_min - slack_u[i],
                    name=f"vmin_{sc_idx}_{t}_{i}",
                )
                model.addConstr(
                    v_expr <= v_max + slack_o[i],
                    name=f"vmax_{sc_idx}_{t}_{i}",
                )

            all_slack_o.extend([(slack_o[i], sc.weight) for i in range(n_mon)])
            all_slack_u.extend([(slack_u[i], sc.weight) for i in range(n_mon)])

    # ── Objective ──

    # 1. Annualised investment cost ($/year) — PV only, no DC cost
    investment = c_inv * gp.quicksum(s[j] for j in range(n_pv_cand))

    # 2. Annual operational savings from PV generation ($/year)
    savings_expr = 0
    for sc_idx, sc in enumerate(scenarios):
        for t in range(len(sc.hours)):
            pv_frac = sc.pv_profile[t]
            if pv_frac < 1e-6:
                continue
            price = sc.price_per_kwh[t]
            actual_gen = gp.quicksum(
                s[j] * pv_frac - curtail[(j, sc_idx, t)]
                for j in range(n_pv_cand)
            )
            savings_expr += sc.weight * price * actual_gen * sc.hours_per_step

    annual_savings = savings_expr * days_per_year

    # 3. Voltage violation penalty
    violation = c_viol * days_per_year * (
        gp.quicksum(w * sv for sv, w in all_slack_o)
        + gp.quicksum(w * sv for sv, w in all_slack_u)
    )

    # 4. Tap switching cost
    switching_cost_expr = 0
    if n_regs > 0 and c_switch > 0 and delta_tap:
        all_switch_vars: list[tuple[gp.Var, float]] = []
        for sc_idx, sc in enumerate(scenarios):
            for h in range(N_TAP_SLOTS - 1):
                for r in range(n_regs):
                    d = model.addVar(lb=0, name=f"dswitch_{sc_idx}_{h}_{r}")
                    diff = delta_tap[(r, sc_idx, h + 1)] - delta_tap[(r, sc_idx, h)]
                    model.addConstr(d >= diff, name=f"sw_pos_{sc_idx}_{h}_{r}")
                    model.addConstr(d >= -diff, name=f"sw_neg_{sc_idx}_{h}_{r}")
                    all_switch_vars.append((d, sc.weight))
        switching_cost_expr = c_switch * days_per_year * gp.quicksum(
            w * d for d, w in all_switch_vars
        )

    model.setObjective(
        investment - annual_savings + violation + switching_cost_expr,
        GRB.MINIMIZE,
    )

    model.update()
    logger.info(
        "MILP: %d PV cand, %d DC cand, %d DC sites, %d monitored, "
        "%d scenarios, %d PVs, %d regulators",
        n_pv_cand, n_dc_cand, n_sites, n_mon, len(scenarios), n_pv, n_regs,
    )
    logger.info(
        "MILP: s_max=%.0f kW, s_total_max=%s kW, c_inv=%.2f $/kW/yr, "
        "c_viol=%.0f, c_switch=%.0f, days/yr=%.0f",
        s_max_kw,
        f"{s_total_max_kw:.0f}" if s_total_max_kw else "None",
        c_inv, c_viol, c_switch, days_per_year,
    )
    logger.info("MILP: %d variables, %d constraints", model.NumVars, model.NumConstrs)

    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0

    if model.SolCount > 0:
        status = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract PV results
        pv_locs: list[str] = []
        pv_caps: list[float] = []
        all_x_vals: dict[str, float] = {}
        all_s_vals: dict[str, float] = {}
        for j in range(n_pv_cand):
            all_x_vals[pv_candidate_buses[j]] = x[j].X
            all_s_vals[pv_candidate_buses[j]] = s[j].X
            if x[j].X > 0.5:
                pv_locs.append(pv_candidate_buses[j])
                pv_caps.append(s[j].X)

        # Extract DC assignments
        dc_assignments: dict[str, str] = {}
        all_y_vals: dict[tuple[str, str], float] = {}
        for k, site_id in enumerate(dc_site_ids):
            for b in dc_zone_indices[site_id]:
                val = y[(k, b)].X
                all_y_vals[(site_id, dc_candidate_buses[b])] = val
                if val > 0.5:
                    dc_assignments[site_id] = dc_candidate_buses[b]

        # Extract tap schedules
        tap_schedules_val = None
        if n_regs > 0 and initial_tap_ints is not None:
            tap_schedules_val = {}
            for sc_idx, sc in enumerate(scenarios):
                schedule: dict[int, dict[str, int]] = {}
                for h_slot in range(N_TAP_SLOTS):
                    hour_taps = {}
                    for r, reg_name in enumerate(regulator_names):
                        delta = int(round(delta_tap[(r, sc_idx, h_slot)].X))
                        hour_taps[reg_name] = initial_tap_ints[reg_name] + delta
                    for hh in range(
                        h_slot * TAP_INTERVAL_H, (h_slot + 1) * TAP_INTERVAL_H
                    ):
                        if hh < 24:
                            schedule[hh] = hour_taps
                tap_schedules_val[sc.name] = schedule

        # Compute cost components
        inv_val = sum(all_s_vals.values()) * c_inv
        savings_val = 0.0
        curtail_kwh = 0.0
        for sc_idx, sc in enumerate(scenarios):
            for t in range(len(sc.hours)):
                pv_frac = sc.pv_profile[t]
                if pv_frac < 1e-6:
                    continue
                price = sc.price_per_kwh[t]
                actual_gen = 0.0
                for j, b in enumerate(pv_candidate_buses):
                    c_val = (
                        curtail[(j, sc_idx, t)].X
                        if (j, sc_idx, t) in curtail
                        else 0.0
                    )
                    gen_j = all_s_vals[b] * pv_frac - c_val
                    actual_gen += gen_j
                    curtail_kwh += sc.weight * c_val * sc.hours_per_step
                savings_val += sc.weight * price * actual_gen * sc.hours_per_step
        savings_val *= days_per_year
        curtail_mwh_annual = curtail_kwh * days_per_year / 1000.0

        # Compute switching cost from solution
        switch_val = 0.0
        if n_regs > 0 and c_switch > 0 and tap_schedules_val:
            for sc_idx, sc in enumerate(scenarios):
                for h in range(N_TAP_SLOTS - 1):
                    for r in range(n_regs):
                        d0 = int(round(delta_tap[(r, sc_idx, h)].X))
                        d1 = int(round(delta_tap[(r, sc_idx, h + 1)].X))
                        switch_val += sc.weight * abs(d1 - d0)
            switch_val *= c_switch * days_per_year

        viol_val = model.ObjVal - inv_val + savings_val - switch_val

        return CoOptMILPResult(
            status=status,
            objective=model.ObjVal,
            pv_locations=pv_locs,
            pv_capacities_kw=pv_caps,
            dc_assignments=dc_assignments,
            investment_cost=inv_val,
            operational_savings=savings_val,
            violation_penalty=viol_val,
            switching_cost=switch_val,
            curtailment_mwh=curtail_mwh_annual,
            solve_time_s=solve_time,
            all_x=all_x_vals,
            all_s=all_s_vals,
            all_y=all_y_vals,
            tap_schedules=tap_schedules_val,
        )
    else:
        return CoOptMILPResult(
            status=(
                "infeasible"
                if model.Status == GRB.INFEASIBLE
                else f"status_{model.Status}"
            ),
            objective=float("inf"),
            pv_locations=[],
            pv_capacities_kw=[],
            dc_assignments={},
            investment_cost=0.0,
            operational_savings=0.0,
            violation_penalty=0.0,
            switching_cost=0.0,
            curtailment_mwh=0.0,
            solve_time_s=solve_time,
            all_x={},
            all_s={},
            all_y={},
        )


# ── Validation wrapper ──────────────────────────────────────────────────────


def validate_coopt(
    *,
    case_dir: Path,
    master_file: str,
    pv_locations: list[str],
    pv_capacities_kw: list[float],
    dc_assignments: dict[str, str],
    dc_demand_kw: np.ndarray,
    dc_site_ids: list[str],
    scenario: Scenario,
    sc_idx: int,
    v_index: list[tuple[str, str]],
    source_pu: float = 1.05,
    initial_taps: dict[str, float] | None = None,
    bus_kv: float = 24.9,
    tap_schedule: dict[int, dict[str, int]] | None = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> dict:
    """Validate co-optimisation result using OpenDSS power flow.

    Embeds DC demand into the scenario's ``load_kw_per_bus`` and calls
    ``validate_with_opendss`` with ``dc_sites=None`` (no fixed DC loads).
    """
    import copy

    # Create a modified scenario with DC demands embedded as time-varying loads
    mod_sc = copy.deepcopy(scenario)

    for k, site_id in enumerate(dc_site_ids):
        assigned_bus = dc_assignments.get(site_id)
        if assigned_bus is None:
            continue
        # dc_demand_kw[k, :, sc_idx] is the total demand in kW per phase
        dc_profile = dc_demand_kw[k, :, sc_idx]
        if assigned_bus in mod_sc.load_kw_per_bus:
            mod_sc.load_kw_per_bus[assigned_bus] = (
                mod_sc.load_kw_per_bus[assigned_bus] + dc_profile
            )
        else:
            mod_sc.load_kw_per_bus[assigned_bus] = dc_profile.copy()

    return validate_with_opendss(
        case_dir=case_dir,
        master_file=master_file,
        pv_locations=pv_locations,
        pv_capacities_kw=pv_capacities_kw,
        scenario=mod_sc,
        v_index=v_index,
        source_pu=source_pu,
        initial_taps=initial_taps,
        bus_kv=bus_kv,
        dc_sites=None,  # DC loads are in load_kw_per_bus
        tap_schedule=tap_schedule,
        v_min=v_min,
        v_max=v_max,
    )


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_coopt_results(
    pv_candidate_buses: list[str],
    dc_candidate_buses: list[str],
    result: CoOptMILPResult,
    save_dir: Path,
    system: str,
) -> None:
    """Bar chart of PV capacities + DC assignment annotations."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PV placements
    ax = axes[0]
    caps = [result.all_s.get(b, 0.0) for b in pv_candidate_buses]
    colors = [
        "#DD8452" if result.all_x.get(b, 0) > 0.5 else "#4C72B0"
        for b in pv_candidate_buses
    ]
    xp = np.arange(len(pv_candidate_buses))
    ax.bar(xp, caps, color=colors)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Installed PV Capacity (kW)")
    ax.set_title(f"PV Placements — {system}")
    ax.set_xticks(xp)
    ax.set_xticklabels(pv_candidate_buses, rotation=45, ha="right", fontsize=8)
    ax.legend(handles=[
        Patch(color="#DD8452", label="Selected"),
        Patch(color="#4C72B0", label="Not selected"),
    ])

    # DC assignments
    ax = axes[1]
    # Show which buses got DC sites
    assigned_buses = set(result.dc_assignments.values())
    dc_colors = [
        "#55A868" if b in assigned_buses else "#CCCCCC"
        for b in dc_candidate_buses
    ]
    xd = np.arange(len(dc_candidate_buses))
    ax.bar(xd, [1 if b in assigned_buses else 0.1 for b in dc_candidate_buses],
           color=dc_colors)
    ax.set_xlabel("Bus")
    ax.set_ylabel("DC Assigned")
    ax.set_title(f"DC Assignments — {system}")
    ax.set_xticks(xd)
    ax.set_xticklabels(dc_candidate_buses, rotation=45, ha="right", fontsize=8)

    # Annotate with site IDs
    for site_id, bus in result.dc_assignments.items():
        if bus in dc_candidate_buses:
            idx = dc_candidate_buses.index(bus)
            ax.annotate(
                site_id, (idx, 1.05), ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

    ax.legend(handles=[
        Patch(color="#55A868", label="DC assigned"),
        Patch(color="#CCCCCC", label="Not assigned"),
    ])

    fig.suptitle(f"PV + DC Co-Optimisation — {system}", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        save_dir / f"pv_dc_placements_{system}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    logger.info("Plot saved: %s", save_dir / f"pv_dc_placements_{system}.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(
    *,
    config_path: Path,
    system: str,
    n_pv: int = 3,
    s_max_kw: float = 500.0,
    s_total_max_kw: float | None = None,
    c_inv: float = 200.0,
    c_viol: float = 5_000.0,
    c_switch: float = 10.0,
    max_tap_delta: int = 8,
    time_limit_s: float = 600.0,
    validate: bool = True,
    v_min: float = 0.95,
    v_max: float = 1.05,
    days_per_year: float = 365.0,
    no_dc_zones: bool = False,
) -> None:
    """Run joint PV + DC location co-optimisation.

    1. Load config, parse taps
    2. Discover PV candidate buses (all 3-phase, zone-filtered)
    3. Discover DC candidate buses (per-zone from config.zones)
    4. Estimate DC peak demand per site
    5. Generate scenarios (WITHOUT dc_sites_info since DC locations are variable)
    6. Compute sensitivities (H_pv, H_dc, H_tap, H_load) on bare circuit
    7. Compute DC demand profiles
    8. Solve MILP
    9. Validate
    10. Save results
    """
    config_path = config_path.resolve()

    config = SweepConfig.model_validate_json(config_path.read_bytes())
    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    assert config.dc_sites is not None, "Config must define dc_sites"
    default_site = next(iter(config.dc_sites.values()))

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "pv_dc_coopt"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.getLogger().addHandler(file_handler)

    # ── Step 1: Discover PV candidate buses ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: Discovering candidate buses")
    logger.info("=" * 70)

    all_candidate_buses = discover_candidate_buses(
        config.ieee_case_dir,
        config.dss_master_file,
        default_site.bus_kv,
        exclude=set(config.exclude_buses),
    )

    # PV candidates: zone-filtered if zones are defined
    pv_candidate_buses = list(all_candidate_buses)
    if config.zones:
        zoned_buses = {
            b.lower() for buses_list in config.zones.values() for b in buses_list
        }
        before = len(pv_candidate_buses)
        pv_candidate_buses = [
            b for b in pv_candidate_buses if b.lower() in zoned_buses
        ]
        logger.info(
            "PV: Filtered to zoned buses: %d -> %d candidates", before, len(pv_candidate_buses)
        )
    logger.info(
        "PV candidate buses (%d): %s", len(pv_candidate_buses), pv_candidate_buses
    )

    # ── Step 2: Build DC candidate buses per zone ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Building DC candidate buses per zone")
    logger.info("=" * 70)

    # DC candidate buses: union of all zone buses that are valid 3-phase buses
    all_cand_lower = {b.lower() for b in all_candidate_buses}
    dc_candidate_buses: list[str] = []
    dc_zone_indices: dict[str, list[int]] = {}

    if config.zones and not no_dc_zones:
        # Each DC site gets candidates from its zone
        zone_names = list(config.zones.keys())
        dc_site_ids = list(config.dc_sites.keys())

        # Collect all unique DC candidate buses across zones
        dc_bus_set: set[str] = set()
        for zone_buses in config.zones.values():
            for b in zone_buses:
                if b.lower() in all_cand_lower:
                    dc_bus_set.add(b)

        dc_candidate_buses = sorted(dc_bus_set)
        dc_cand_lower = [b.lower() for b in dc_candidate_buses]

        # Build zone indices: each site gets candidates from its assigned zone
        # If there are more sites than zones, cycle through zones
        for k, site_id in enumerate(dc_site_ids):
            zone_name = zone_names[k % len(zone_names)]
            zone_set = {b.lower() for b in config.zones[zone_name]}
            indices = [
                j for j, b in enumerate(dc_cand_lower) if b in zone_set
            ]
            dc_zone_indices[site_id] = indices
            logger.info(
                "  DC site %s -> zone %s: %d candidate buses",
                site_id, zone_name, len(indices),
            )
    else:
        # No zones: all candidate buses are available for all DC sites
        dc_candidate_buses = list(all_candidate_buses)
        dc_site_ids = list(config.dc_sites.keys())
        all_indices = list(range(len(dc_candidate_buses)))
        for site_id in dc_site_ids:
            dc_zone_indices[site_id] = list(all_indices)

    logger.info(
        "DC candidate buses (%d): %s", len(dc_candidate_buses), dc_candidate_buses
    )

    # ── Step 3: Parse taps and compute DC demand info ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Parsing taps and DC demand")
    logger.info("=" * 70)

    # Parse initial taps
    tap_pu: dict[str, float] = {}
    initial_tap_ints: dict[str, int] = {}
    regulator_names: list[str] = []
    if config.initial_taps:
        for name, val in config.initial_taps.items():
            if isinstance(val, str):
                tap_int = int(val)
                tap_pu[name] = 1.0 + tap_int * TAP_STEP
                initial_tap_ints[name] = tap_int
            else:
                tap_pu[name] = float(val)
                initial_tap_ints[name] = round((float(val) - 1.0) / TAP_STEP)
            regulator_names.append(name)

    # Estimate GPU counts per model
    TYPICAL_GPU_W = 300.0
    model_gpu_map = {
        m.model_label: m.initial_num_replicas * m.gpus_per_replica
        for m in config.models
    }

    # ── Build load_configs from config ──
    load_configs: list[tuple[str, float]] = []
    load_buses: list[str] = []
    if config.time_varying_loads:
        for lc in config.time_varying_loads:
            load_configs.append((lc.bus, lc.peak_kw))
            load_buses.append(lc.bus)
        logger.info(
            "Load configs: %s",
            ", ".join(f"{b}={kw:.0f}kW" for b, kw in load_configs),
        )

    # ── Step 4: Generate scenarios (WITHOUT dc_sites_info) ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: Generating operating scenarios")
    logger.info("=" * 70)

    scenarios = generate_scenarios(
        load_configs=load_configs if load_configs else None,
        dc_sites_info=None,  # DC locations are variable — no fixed DC in scenarios
    )
    total_weight = sum(sc.weight for sc in scenarios)
    logger.info("Total scenario weight: %.4f (should be 1.0)", total_weight)
    for sc in scenarios:
        logger.info(
            "  %s: %d steps, weight=%.2f (~%d days/yr), peak PV=%.2f, loads=%d buses",
            sc.name, len(sc.hours), sc.weight,
            int(sc.weight * 365),
            sc.pv_profile.max(),
            len(sc.load_kw_per_bus),
        )

    # ── Step 5: Compute sensitivities on bare circuit (no DC) ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: Computing voltage sensitivities")
    logger.info("=" * 70)

    # 5a: PV sensitivities
    logger.info("  5a: PV sensitivities...")
    t0 = time.time()
    H_pv, v0, v_index = compute_pv_sensitivities(
        case_dir=config.ieee_case_dir,
        master_file=config.dss_master_file,
        candidate_buses=pv_candidate_buses,
        exclude_buses=set(config.exclude_buses),
        source_pu=config.source_pu or 1.05,
        initial_taps=tap_pu if tap_pu else None,
        bus_kv=default_site.bus_kv,
        dc_sites=None,  # bare circuit
    )
    logger.info("  PV sensitivity: %.1f s, %d monitored x %d candidates",
                time.time() - t0, H_pv.shape[0], H_pv.shape[1])
    logger.info("  Base voltage range: %.4f to %.4f pu", v0.min(), v0.max())

    plot_sensitivity_heatmap(H_pv, v_index, pv_candidate_buses, save_dir, system)
    # Rename the default output
    default_heatmap = save_dir / f"pv_sensitivity_{system}.png"
    pv_heatmap = save_dir / "sensitivity_heatmap_pv.png"
    if default_heatmap.exists():
        default_heatmap.replace(pv_heatmap)

    # 5b: DC sensitivities
    logger.info("  5b: DC sensitivities...")
    t0 = time.time()
    H_dc = compute_dc_sensitivities(
        case_dir=config.ieee_case_dir,
        master_file=config.dss_master_file,
        dc_candidate_buses=dc_candidate_buses,
        v_index=v_index,
        source_pu=config.source_pu or 1.05,
        initial_taps=tap_pu if tap_pu else None,
        bus_kv=default_site.bus_kv,
    )
    logger.info("  DC sensitivity: %.1f s, %d monitored x %d candidates",
                time.time() - t0, H_dc.shape[0], H_dc.shape[1])

    # Plot DC sensitivity heatmap
    plot_sensitivity_heatmap(H_dc, v_index, dc_candidate_buses, save_dir, system)
    default_heatmap = save_dir / f"pv_sensitivity_{system}.png"
    dc_heatmap = save_dir / "sensitivity_heatmap_dc.png"
    if default_heatmap.exists():
        default_heatmap.replace(dc_heatmap)

    # 5c: Tap sensitivities
    H_tap = None
    if regulator_names:
        logger.info("  5c: Tap sensitivities...")
        logger.info("    Regulators: %s", regulator_names)
        t0 = time.time()
        H_tap = compute_tap_sensitivities(
            case_dir=config.ieee_case_dir,
            master_file=config.dss_master_file,
            regulator_names=regulator_names,
            v_index=v_index,
            source_pu=config.source_pu or 1.05,
            initial_taps=tap_pu if tap_pu else None,
            bus_kv=default_site.bus_kv,
            dc_sites=None,
        )
        logger.info("  Tap sensitivity: %.1f s", time.time() - t0)

    # 5d: Load sensitivities
    load_v_shift = None
    if load_buses:
        logger.info("  5d: Load sensitivities...")
        t0 = time.time()
        H_load = compute_load_sensitivities(
            case_dir=config.ieee_case_dir,
            master_file=config.dss_master_file,
            load_buses=load_buses,
            v_index=v_index,
            source_pu=config.source_pu or 1.05,
            initial_taps=tap_pu if tap_pu else None,
            bus_kv=default_site.bus_kv,
            dc_sites=None,
        )
        logger.info("  Load sensitivity: %.1f s", time.time() - t0)
        load_v_shift = precompute_load_v_shift(H_load, load_buses, scenarios)
        logger.info("  Precomputed load voltage shifts for %d scenarios", len(load_v_shift))

    # ── Step 6: Compute DC demand profiles ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 6: Computing DC demand profiles")
    logger.info("=" * 70)

    dc_demand_kw, site_ids_ordered = compute_dc_demand_profiles(
        config.dc_sites, model_gpu_map, scenarios,
        typical_gpu_w=TYPICAL_GPU_W,
    )
    logger.info(
        "DC demand shape: %s (sites x timesteps x scenarios)", dc_demand_kw.shape
    )
    for k, sid in enumerate(site_ids_ordered):
        logger.info(
            "  %s: mean=%.0f, min=%.0f, max=%.0f kW/ph",
            sid,
            dc_demand_kw[k].mean(),
            dc_demand_kw[k].min(),
            dc_demand_kw[k].max(),
        )

    # ── Step 7: Solve MILP ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 7: Solving PV + DC co-optimisation MILP")
    logger.info("=" * 70)
    logger.info(
        "  n_pv=%d, s_max=%.0f kW, s_total_max=%s kW, "
        "c_inv=%.1f $/kW/yr, c_viol=%.0f, c_switch=%.0f",
        n_pv, s_max_kw,
        f"{s_total_max_kw:.0f}" if s_total_max_kw else "None",
        c_inv, c_viol, c_switch,
    )
    logger.info("  v_min=%.2f, v_max=%.2f, days_per_year=%.0f", v_min, v_max, days_per_year)

    result = solve_pv_dc_milp(
        pv_candidate_buses=pv_candidate_buses,
        dc_candidate_buses=dc_candidate_buses,
        dc_site_ids=site_ids_ordered,
        dc_zone_indices=dc_zone_indices,
        dc_demand_kw=dc_demand_kw,
        H_pv=H_pv,
        H_dc=H_dc,
        v0=v0,
        v_index=v_index,
        scenarios=scenarios,
        n_pv=n_pv,
        s_max_kw=s_max_kw,
        s_total_max_kw=s_total_max_kw,
        v_min=v_min,
        v_max=v_max,
        c_inv=c_inv,
        c_viol=c_viol,
        time_limit_s=time_limit_s,
        mip_gap=0.02,
        H_tap=H_tap,
        regulator_names=regulator_names if H_tap is not None else None,
        initial_tap_ints=initial_tap_ints if H_tap is not None else None,
        max_tap_delta=max_tap_delta,
        days_per_year=days_per_year,
        c_switch=c_switch,
        load_v_shift=load_v_shift,
        pv_zones=config.zones,
    )

    # ── Report results ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("CO-OPT MILP RESULT")
    logger.info("=" * 70)
    logger.info("  Status: %s", result.status)
    logger.info("  Objective: $%.2f/year", result.objective)
    logger.info("  Investment cost: $%.2f/year", result.investment_cost)
    logger.info("  Operational savings: $%.2f/year", result.operational_savings)
    logger.info("  Violation penalty: $%.2f/year", result.violation_penalty)
    logger.info("  Switching cost: $%.2f/year", result.switching_cost)
    logger.info("  Curtailment: %.1f MWh/year", result.curtailment_mwh)
    logger.info("  Solve time: %.1f s", result.solve_time_s)
    logger.info("")
    logger.info("  PV Placements:")
    if result.pv_locations:
        for bus, cap in zip(result.pv_locations, result.pv_capacities_kw):
            logger.info(
                "    Bus %-10s  Capacity: %8.1f kW (%.1f kW/phase)", bus, cap, cap / 3
            )
    else:
        logger.info("    (none)")

    logger.info("")
    logger.info("  DC Assignments:")
    for site_id, bus in result.dc_assignments.items():
        logger.info("    Site %-10s -> Bus %s", site_id, bus)

    if result.tap_schedules:
        logger.info("")
        logger.info("  Tap Schedules (time-varying):")
        for sc_name, schedule in result.tap_schedules.items():
            unique_settings: dict[tuple, list[int]] = {}
            for hour, taps in sorted(schedule.items()):
                key = tuple(taps[r] for r in regulator_names)
                if key not in unique_settings:
                    unique_settings[key] = []
                unique_settings[key].append(hour)
            logger.info(
                "    Scenario: %s (%d distinct tap settings)",
                sc_name, len(unique_settings),
            )
            for taps_tuple, hours_list in unique_settings.items():
                hour_range = (
                    f"h{hours_list[0]}-{hours_list[-1]}"
                    if len(hours_list) > 1
                    else f"h{hours_list[0]}"
                )
                taps_str = ", ".join(
                    f"{r}={t:+d}" for r, t in zip(regulator_names, taps_tuple)
                )
                logger.info("      %s: %s", hour_range, taps_str)

    # ── Save outputs ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("Saving outputs")
    logger.info("=" * 70)

    # PV placements CSV
    pv_rows = []
    for bus in pv_candidate_buses:
        pv_rows.append({
            "bus": bus,
            "selected": int(result.all_x.get(bus, 0) > 0.5),
            "capacity_kw": result.all_s.get(bus, 0.0),
        })
    pv_csv_path = save_dir / f"pv_placements_{system}.csv"
    with open(pv_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pv_rows)
    logger.info("  PV placements: %s", pv_csv_path)

    # DC assignments CSV
    dc_rows = []
    for site_id in site_ids_ordered:
        assigned_bus = result.dc_assignments.get(site_id, "")
        # Determine zone name
        zone_name = ""
        if config.zones:
            zone_names = list(config.zones.keys())
            k = site_ids_ordered.index(site_id)
            zone_name = zone_names[k % len(zone_names)]
        # Peak demand
        peak_kw = float(dc_demand_kw[site_ids_ordered.index(site_id)].max())
        dc_rows.append({
            "site_id": site_id,
            "assigned_bus": assigned_bus,
            "zone": zone_name,
            "peak_kw_per_phase": f"{peak_kw:.1f}",
        })
    dc_csv_path = save_dir / f"dc_assignments_{system}.csv"
    with open(dc_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dc_rows)
    logger.info("  DC assignments: %s", dc_csv_path)

    # Tap schedule CSV
    if result.tap_schedules:
        tap_csv_path = save_dir / f"tap_schedule_{system}.csv"
        with open(tap_csv_path, "w", newline="") as f:
            fields = (
                ["scenario", "hour"]
                + [f"{r}_tap" for r in regulator_names]
                + [f"{r}_pu" for r in regulator_names]
            )
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for sc_name, schedule in result.tap_schedules.items():
                for hour in sorted(schedule.keys()):
                    taps = schedule[hour]
                    row: dict = {"scenario": sc_name, "hour": hour}
                    for r in regulator_names:
                        row[f"{r}_tap"] = taps[r]
                        row[f"{r}_pu"] = f"{1.0 + taps[r] * TAP_STEP:.5f}"
                    writer.writerow(row)
        logger.info("  Tap schedule: %s", tap_csv_path)

    # Summary JSON
    summary = {
        "status": result.status,
        "objective_dollar_per_year": result.objective,
        "investment_cost_dollar_per_year": result.investment_cost,
        "operational_savings_dollar_per_year": result.operational_savings,
        "violation_penalty": result.violation_penalty,
        "switching_cost_dollar_per_year": result.switching_cost,
        "curtailment_mwh_per_year": result.curtailment_mwh,
        "solve_time_s": result.solve_time_s,
        "pv_locations": result.pv_locations,
        "pv_capacities_kw": result.pv_capacities_kw,
        "total_pv_kw": sum(result.pv_capacities_kw),
        "dc_assignments": result.dc_assignments,
        "tap_schedules": result.tap_schedules,
        "parameters": {
            "n_pv": n_pv,
            "s_max_kw": s_max_kw,
            "s_total_max_kw": s_total_max_kw,
            "c_inv": c_inv,
            "c_viol": c_viol,
            "c_switch": c_switch,
            "v_min": v_min,
            "v_max": v_max,
            "max_tap_delta": max_tap_delta,
            "days_per_year": days_per_year,
        },
    }
    summary_path = save_dir / f"milp_summary_{system}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("  Summary: %s", summary_path)

    # Plots
    plot_coopt_results(
        pv_candidate_buses, dc_candidate_buses, result, save_dir, system
    )

    # ── Plot optimized topology ──
    try:
        from plot_topology import plot_topology

        # Build a modified config dict with optimized locations
        cfg_dict = json.loads(config_path.read_bytes())
        # Update DC sites to optimized buses
        for site_id, assigned_bus in result.dc_assignments.items():
            if site_id in cfg_dict.get("dc_sites", {}):
                cfg_dict["dc_sites"][site_id]["bus"] = assigned_bus
        # Update PV systems to optimized locations and capacities
        cfg_dict["pv_systems"] = [
            {"bus": bus, "bus_kv": default_site.bus_kv,
             "peak_kw": cap / 3.0, "power_factor": 1.0}
            for bus, cap in zip(result.pv_locations, result.pv_capacities_kw)
        ]
        # Use absolute ieee_case_dir so the temp config works from any location
        cfg_dict["ieee_case_dir"] = str(config.ieee_case_dir)
        # Write temporary config for topology plot
        tmp_config = save_dir / f"_optimized_config_{system}.json"
        tmp_config.write_text(json.dumps(cfg_dict, indent=2))
        plot_topology(
            config_path=tmp_config, system=system,
            output_dir=save_dir,
        )
        # Rename to a more descriptive name
        topo_default = save_dir / f"{system}_topology.png"
        topo_final = save_dir / f"optimized_topology_{system}.png"
        if topo_default.exists():
            topo_default.replace(topo_final)
            logger.info("  Topology plot: %s", topo_final)
        tmp_config.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("  Topology plot failed: %s", e)

    # ── Step 8: Validate with full OpenDSS simulation ──
    if validate and (result.pv_locations or result.dc_assignments):
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 8: Validating with full OpenDSS simulation")
        logger.info("=" * 70)

        val_results = []
        for sc_idx, sc in enumerate(scenarios):
            val_schedule = None
            if result.tap_schedules and sc.name in result.tap_schedules:
                val_schedule = result.tap_schedules[sc.name]
                unique = len(set(
                    tuple(sorted(taps.items()))
                    for taps in val_schedule.values()
                ))
                logger.info(
                    "  Validating %s with %d distinct tap settings across %d hours",
                    sc.name, unique, len(val_schedule),
                )

            vr = validate_coopt(
                case_dir=config.ieee_case_dir,
                master_file=config.dss_master_file,
                pv_locations=result.pv_locations,
                pv_capacities_kw=result.pv_capacities_kw,
                dc_assignments=result.dc_assignments,
                dc_demand_kw=dc_demand_kw,
                dc_site_ids=site_ids_ordered,
                scenario=sc,
                sc_idx=sc_idx,
                v_index=v_index,
                source_pu=config.source_pu or 1.05,
                initial_taps=tap_pu if tap_pu else None,
                bus_kv=default_site.bus_kv,
                tap_schedule=val_schedule,
                v_min=v_min,
                v_max=v_max,
            )
            val_results.append(vr)

        val_csv_path = save_dir / f"validation_{system}.csv"
        with open(val_csv_path, "w", newline="") as f:
            fields = list(val_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for vr in val_results:
                row = dict(vr)
                row["pv_locations"] = ";".join(row["pv_locations"])
                row["pv_capacities_kw"] = ";".join(
                    f"{c:.1f}" for c in row["pv_capacities_kw"]
                )
                writer.writerow(row)
        logger.info("  Validation CSV: %s", val_csv_path)

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        system: str = "ieee123"
        """System name for output directory."""
        n_pv: int = 3
        """Max number of PV sites to place."""
        s_max_kw: float = 500.0
        """Maximum PV capacity per site (kW, 3-phase total)."""
        s_total_max_kw: float | None = None
        """Total PV capacity budget across all sites (kW). None = no limit."""
        c_inv: float = 200.0
        """Annualised investment cost ($/kW/year)."""
        c_viol: float = 5_000.0
        """Voltage violation penalty per pu slack per bus-phase per hour."""
        c_switch: float = 10.0
        """Tap switching cost per switch per regulator per hour ($/switch)."""
        max_tap_delta: int = 8
        """Max tap change from initial (+-steps)."""
        time_limit: float = 600.0
        """Solver time limit (seconds)."""
        validate: bool = True
        """Run full OpenDSS validation after MILP solve."""
        no_dc_zones: bool = False
        """Allow DC sites to be placed at any bus (ignore zone constraints)."""
        log_level: str = "INFO"
        """Logging verbosity."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("gurobipy").setLevel(logging.WARNING)
    logging.getLogger("openg2g.coordinator").setLevel(logging.WARNING)
    logging.getLogger("openg2g.datacenter").setLevel(logging.WARNING)
    logging.getLogger("openg2g.grid").setLevel(logging.WARNING)

    main(
        config_path=Path(args.config),
        system=args.system,
        n_pv=args.n_pv,
        s_max_kw=args.s_max_kw,
        s_total_max_kw=args.s_total_max_kw,
        c_inv=args.c_inv,
        c_viol=args.c_viol,
        c_switch=args.c_switch,
        max_tap_delta=args.max_tap_delta,
        time_limit_s=args.time_limit,
        validate=args.validate,
        no_dc_zones=args.no_dc_zones,
    )
