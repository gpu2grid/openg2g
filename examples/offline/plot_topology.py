"""Plot distribution system topology from DSS files with DC locations, PV, and regulators.

Automatically parses bus coordinates, line/transformer connections, and regulator
definitions from OpenDSS files. Overlays DC sites, PV systems, and zone coloring
from experiment definitions in run_ofo.py.

Usage:
    python plot_topology.py --system ieee13
    python plot_topology.py --system ieee34
    python plot_topology.py --system ieee123
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ── Parsing helpers ──────────────────────────────────────────────────────────


def parse_bus_coords(path: Path) -> dict[str, tuple[float, float]]:
    """Parse bus coordinates from a CSV or DAT file.

    Supports both comma-separated and whitespace-separated formats.
    """
    coords: dict[str, tuple[float, float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try comma-separated first, then whitespace
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
            else:
                parts = line.split()
            if len(parts) >= 3:
                try:
                    coords[parts[0].lower()] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    continue
    return coords


def _strip_bus_phases(bus_spec: str) -> str:
    """Strip phase suffixes from a bus specification (e.g., '632.1.2.3' -> '632')."""
    return bus_spec.split(".")[0].strip().lower()


def parse_dss_edges(dss_dir: Path, master_file: str) -> list[tuple[str, str]]:
    """Parse line and transformer connections from DSS files.

    Follows `redirect` directives to find all Line and Transformer definitions.
    Returns a list of (bus_a, bus_b) tuples with lowercase bus names.
    """
    edges: list[tuple[str, str]] = []
    visited: set[str] = set()

    def _parse_file(fpath: Path) -> None:
        if not fpath.exists():
            return
        key = str(fpath.resolve()).lower()
        if key in visited:
            return
        visited.add(key)

        text = fpath.read_text(errors="replace")
        # Join continuation lines (~ at start of line)
        joined_lines: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("~"):
                if joined_lines:
                    joined_lines[-1] += " " + stripped[1:]
                continue
            joined_lines.append(stripped)

        for line in joined_lines:
            lower = line.lower()
            # Follow redirect/compile directives
            for directive in ("redirect", "compile"):
                if lower.startswith(directive):
                    ref = line[len(directive) :].strip().strip('"').strip("'")
                    _parse_file(fpath.parent / ref)

            # Parse New Line.xxx Bus1=... Bus2=...
            if re.match(r"new\s+line\.", lower):
                m1 = re.search(r"bus1\s*=\s*(\S+)", lower)
                m2 = re.search(r"bus2\s*=\s*(\S+)", lower)
                if m1 and m2:
                    a = _strip_bus_phases(m1.group(1))
                    b = _strip_bus_phases(m2.group(1))
                    edges.append((a, b))

            # Parse New Transformer.xxx ... Buses=[bus1 bus2] or Buses=(bus1 bus2) or bus=bus1 ... bus=bus2
            if re.match(r"new\s+transformer\.", lower):
                # Try Buses=[...] or Buses=(...) format
                m_buses = re.search(r"buses\s*=\s*[\[\(]([^\]\)]+)[\]\)]", lower)
                if m_buses:
                    bus_list = [_strip_bus_phases(b) for b in m_buses.group(1).split()]
                    for i in range(len(bus_list) - 1):
                        edges.append((bus_list[i], bus_list[i + 1]))
                else:
                    # Try multiple bus=xxx patterns (winding-based)
                    bus_matches = re.findall(r"bus\s*=\s*(\S+)", lower)
                    for i in range(len(bus_matches) - 1):
                        a = _strip_bus_phases(bus_matches[i])
                        b = _strip_bus_phases(bus_matches[i + 1])
                        edges.append((a, b))

    _parse_file(dss_dir / master_file)
    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for e in edges:
        key = (min(e), max(e))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


@dataclass
class RegulatorInfo:
    name: str
    from_bus: str
    to_bus: str
    phases: str
    ctrl_name: str


def parse_dss_regulators(dss_dir: Path, master_file: str) -> list[RegulatorInfo]:
    """Parse voltage regulators from DSS files.

    Identifies transformers that have an associated RegControl, extracts their
    bus connections and phase information.
    """
    transformers: dict[str, dict] = {}  # name -> {buses, phases, bank}
    regcontrols: dict[str, str] = {}  # ctrl_name -> transformer_name
    visited: set[str] = set()

    def _parse_file(fpath: Path) -> None:
        if not fpath.exists():
            return
        key = str(fpath.resolve()).lower()
        if key in visited:
            return
        visited.add(key)

        text = fpath.read_text(errors="replace")
        joined_lines: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("~"):
                if joined_lines:
                    joined_lines[-1] += " " + stripped[1:]
                continue
            joined_lines.append(stripped)

        for line in joined_lines:
            lower = line.lower()
            for directive in ("redirect", "compile"):
                if lower.startswith(directive):
                    ref = line[len(directive) :].strip().strip('"').strip("'")
                    _parse_file(fpath.parent / ref)

            # Parse transformers
            m_xfm = re.match(r"new\s+transformer\.(\S+)", lower)
            if m_xfm:
                xfm_name = m_xfm.group(1)
                m_buses = re.search(r"buses\s*=\s*[\[\(]([^\]\)]+)[\]\)]", lower)
                if m_buses:
                    buses = [_strip_bus_phases(b) for b in m_buses.group(1).split()]
                else:
                    buses = [_strip_bus_phases(b) for b in re.findall(r"bus\s*=\s*(\S+)", lower)]

                m_phases = re.search(r"phases\s*=\s*(\d+)", lower)
                phases = int(m_phases.group(1)) if m_phases else 3

                m_bank = re.search(r"bank\s*=\s*(\S+)", lower)
                bank = m_bank.group(1) if m_bank else None

                transformers[xfm_name] = {"buses": buses, "phases": phases, "bank": bank}

            # Parse regcontrols
            m_reg = re.match(r"new\s+regcontrol\.(\S+)", lower)
            if m_reg:
                ctrl_name = m_reg.group(1)
                m_xfm_ref = re.search(r"transformer\s*=\s*(\S+)", lower)
                if m_xfm_ref:
                    regcontrols[ctrl_name] = m_xfm_ref.group(1)

    _parse_file(dss_dir / master_file)

    # Group regcontrols by transformer bank
    bank_ctrls: dict[str, list[str]] = {}  # bank -> [ctrl_names]
    for ctrl_name, xfm_name in regcontrols.items():
        xfm = transformers.get(xfm_name)
        if xfm and xfm["bank"]:
            bank_ctrls.setdefault(xfm["bank"], []).append(ctrl_name)

    # Build regulator info
    regs: list[RegulatorInfo] = []
    seen_banks: set[str] = set()

    for ctrl_name, xfm_name in regcontrols.items():
        xfm = transformers.get(xfm_name)
        if not xfm or len(xfm["buses"]) < 2:
            continue

        bank = xfm["bank"]
        if bank and bank in seen_banks:
            continue
        if bank:
            seen_banks.add(bank)

        from_bus = xfm["buses"][0]
        to_bus = xfm["buses"][1]
        n_phases = xfm["phases"]

        if bank and bank in bank_ctrls:
            ctrls = sorted(bank_ctrls[bank])
            ctrl_label = ", ".join(ctrls)
            total_phases = len(ctrls) * n_phases
            if total_phases >= 3 and n_phases == 1:
                phase_label = "ABC"
            else:
                phase_label = f"{n_phases}ph x{len(ctrls)}"
            display_name = bank.upper() if bank else ctrl_name
        else:
            ctrl_label = ctrl_name
            phase_label = {1: "1ph", 2: "2ph", 3: "3ph"}.get(n_phases, f"{n_phases}ph")
            display_name = ctrl_name.upper()

        regs.append(
            RegulatorInfo(
                name=display_name,
                from_bus=from_bus,
                to_bus=to_bus,
                phases=phase_label,
                ctrl_name=ctrl_label,
            )
        )

    return regs


# ── Zone color palette ───────────────────────────────────────────────────────

ZONE_PALETTE = [
    "#2196F3",
    "#4CAF50",
    "#FF9800",
    "#E91E63",
    "#9C27B0",
    "#00BCD4",
    "#795548",
    "#607D8B",
]


# ── Main plotting function ───────────────────────────────────────────────────


def plot_topology(
    *,
    system: str,
    output_dir: Path | None = None,
    large_font: bool = False,
) -> Path:
    """Plot system topology and return the output file path."""
    from run_ofo import _EXPERIMENTS
    from systems import SYSTEMS, DT_DC, load_data_sources

    sys = SYSTEMS[system]()
    dss_dir = sys["dss_case_dir"]
    master_file = sys["dss_master_file"]

    # Build experiment to get DC sites, PV, zones
    # We don't need real data — just the experiment structure
    experiment_fn = _EXPERIMENTS[system]
    experiment = experiment_fn(sys, None, None, None)

    # Find bus coordinate file
    coord_file = None
    for candidate in dss_dir.iterdir():
        if candidate.suffix.lower() in (".csv", ".dat") and "coord" in candidate.name.lower():
            coord_file = candidate
            break
        if "busxy" in candidate.name.lower():
            coord_file = candidate
            break
    if coord_file is None:
        raise FileNotFoundError(f"No bus coordinate file found in {dss_dir}")

    coords = parse_bus_coords(coord_file)
    edges = parse_dss_edges(dss_dir, master_file)
    parse_dss_regulators(dss_dir, master_file)

    # Overlays from experiment
    zones_cfg = experiment.get("zones") or sys.get("zones") or {}
    dc_sites_obj = experiment.get("dc_sites", {})
    pv_systems_list = experiment.get("pv_systems") or []

    # Convert DCSite objects to dicts for plotting
    dc_sites = {sid: {"bus": site.bus} for sid, site in dc_sites_obj.items()}
    dc_buses = {site.bus.lower(): sid for sid, site in dc_sites_obj.items()}
    pv_buses = {pv.bus.lower(): f"{pv.peak_kw:.0f} kW/ph" for pv in pv_systems_list}

    # Zone coloring
    zone_ids = list(zones_cfg.keys()) if zones_cfg else list(dc_sites.keys())
    zone_colors = {zid: ZONE_PALETTE[i % len(ZONE_PALETTE)] for i, zid in enumerate(zone_ids)}

    bus_zone: dict[str, str] = {}
    for zid, buses in zones_cfg.items():
        for b in buses:
            bus_zone[b.lower()] = zid

    # ── Font scale ──
    fs = 2.5 if large_font else 1.0  # font scale factor

    # ── Plot ──
    # fig, ax = plt.subplots(figsize=(20, 15))
    fig, ax = plt.subplots(figsize=(30, 10))

    # Draw edges
    for a, b in edges:
        if a in coords and b in coords:
            x = [coords[a][0], coords[b][0]]
            y = [coords[a][1], coords[b][1]]
            za, zb = bus_zone.get(a), bus_zone.get(b)
            if za and za == zb:
                color, alpha = zone_colors[za], 0.5
            else:
                color, alpha = "#BDBDBD", 0.7
            # ax.plot(x, y, "-", color=color, linewidth=1.2, alpha=alpha, zorder=1)
            ax.plot(x, y, "-", color=color, linewidth=3.0, alpha=alpha, zorder=1)

    # Draw buses
    {b.lower() for b in sys.get("exclude_buses", ())}
    for bus_name, (x, y) in coords.items():
        zid = bus_zone.get(bus_name)
        is_dc = bus_name in dc_buses
        is_pv = bus_name in pv_buses

        if is_dc:
            color = zone_colors.get(zid, "#999999")
            # ax.plot(x, y, "*", color=color, markersize=24, markeredgecolor="black",
            #         markeredgewidth=1.5, zorder=5)
            ax.plot(x, y, "*", color=color, markersize=48, markeredgecolor="black", markeredgewidth=2.0, zorder=5)
        elif is_pv:
            # ax.plot(x, y, "^", color="#FFD600", markersize=14, markeredgecolor="black",
            #         markeredgewidth=1.2, zorder=5)
            ax.plot(x, y, "^", color="#FFD600", markersize=28, markeredgecolor="black", markeredgewidth=1.8, zorder=5)
        elif zid:
            ax.plot(
                x, y, "o", color=zone_colors[zid], markersize=6, markeredgecolor="black", markeredgewidth=0.4, zorder=3
            )
        else:
            ax.plot(x, y, "s", color="#CCCCCC", markersize=5, markeredgecolor="#888", markeredgewidth=0.3, zorder=2)

        fontsize = (9 if (is_dc or is_pv) else 6) * fs
        fontweight = "bold" if (is_dc or is_pv) else "normal"
        ax.annotate(
            bus_name,
            (x, y),
            textcoords="offset points",
            xytext=(0, (8 if (is_dc or is_pv) else 5) * fs),
            ha="center",
            fontsize=fontsize,
            fontweight=fontweight,
            zorder=6,
        )

    # Draw regulators (temporarily disabled)
    # for i, reg in enumerate(regulators):
    #     fb, tb = reg.from_bus, reg.to_bus
    #     if fb in coords and tb in coords:
    #         mx = (coords[fb][0] + coords[tb][0]) / 2
    #         my = (coords[fb][1] + coords[tb][1]) / 2
    #         ax.plot(mx, my, "D", color="red", markersize=14, markeredgecolor="darkred",
    #                 markeredgewidth=2, zorder=10)
    #         # Alternate annotation placement
    #         angle = 45 + (i % 4) * 90
    #         import math
    #         ox = 20 * math.cos(math.radians(angle))
    #         oy = 20 * math.sin(math.radians(angle))
    #         ax.annotate(
    #             f"{reg.name}\n({reg.ctrl_name}, {reg.phases})",
    #             (mx, my), textcoords="offset points",
    #             xytext=(ox, oy), ha="left", fontsize=8 * fs, fontweight="bold",
    #             color="darkred", zorder=11,
    #             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
    #                       edgecolor="darkred", alpha=0.9),
    #             arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
    #         )

    # Legend
    handles = []
    if zones_cfg:
        for zid in zone_ids:
            dc_bus = dc_sites.get(zid, {}).get("bus", "?")
            handles.append(
                mpatches.Patch(
                    color=zone_colors[zid],
                    label=f"Zone {zid} — DC@{dc_bus}",
                )
            )
    elif dc_sites:
        for sid, site in dc_sites.items():
            handles.append(
                mpatches.Patch(
                    color=zone_colors.get(sid, "#999999"),
                    label=f"DC '{sid}' @ bus {site['bus']}",
                )
            )
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markersize=15,
            markeredgecolor="black",
            label="DC Location",
        )
    )
    if pv_buses:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#FFD600",
                markersize=12,
                markeredgecolor="black",
                label="PV System",
            )
        )
    # handles.append(plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
    #                markersize=10, markeredgecolor="darkred", label="Voltage Regulator"))
    handles.append(mpatches.Patch(color="#CCCCCC", label="Unzoned / Source"))
    # ax.legend(handles=handles, loc="best", fontsize=11 * fs, framealpha=0.9)
    # ax.legend(handles=handles, loc="lower right", fontsize=11 * fs, framealpha=0.9)
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=11 * fs,
        framealpha=0.9,
        borderaxespad=0,
    )

    system.upper().replace("IEEE", "IEEE ")
    # ax.set_title(f"{system_upper} Topology with DCs, PV Systems, and Regulators",
    #              fontsize=18 * fs, fontweight="bold")
    ax.set_xlabel("X coordinate", fontsize=13 * fs)
    ax.set_ylabel("Y coordinate", fontsize=13 * fs)
    ax.tick_params(labelsize=10 * fs)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    save_dir = output_dir or (Path(__file__).resolve().parent / "outputs" / system)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{system}_topology.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    from dataclasses import dataclass as dc_cls

    import tyro

    @dc_cls
    class Args:
        system: str = "ieee13"
        """System name (e.g., ieee13, ieee34, ieee123)."""
        output_dir: str | None = None
        """Override output directory."""
        large_font: bool = False
        """Use very large fonts for all text on the plot."""

    args = tyro.cli(Args)
    out_dir = Path(args.output_dir) if args.output_dir else None
    plot_topology(system=args.system, output_dir=out_dir, large_font=args.large_font)
