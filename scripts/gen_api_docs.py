"""Generate API reference markdown stubs and update zensical.toml nav.

Walks the openg2g package tree, groups modules by top-level package,
and writes one docs/api/<name>.md per group. Each file contains an h1
title and a sequence of ::: directives for mkdocstrings. The API
Reference section in zensical.toml is updated to match.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import tomlkit

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "openg2g"
DOCS_API_DIR = Path(__file__).resolve().parent.parent / "docs" / "api"
ZENSICAL_TOML_SRC = Path(__file__).resolve().parent.parent / "_zensical.toml"
ZENSICAL_TOML_OUT = Path(__file__).resolve().parent.parent / "zensical.toml"

# Modules to exclude from API docs (internal helpers, etc.).
EXCLUDE_MODULES: set[str] = set()


def discover_modules() -> dict[str, list[str]]:
    """Discover all public modules and group by top-level component.

    Returns a dict mapping page name to a sorted list of module paths.
    Top-level modules (e.g. `openg2g.clock`) map to a single-element list.
    Packages (e.g. `openg2g.datacenter`) map to all their submodules.
    """
    groups: dict[str, list[str]] = defaultdict(list)

    for py_file in sorted(PACKAGE_DIR.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue

        rel = py_file.relative_to(PACKAGE_DIR.parent)
        module = str(rel.with_suffix("")).replace("/", ".")

        if module in EXCLUDE_MODULES:
            continue

        parts = module.split(".")  # e.g. ["openg2g", "datacenter", "offline"]
        page_name = parts[1]
        groups[page_name].append(module)

    return dict(groups)


def write_api_pages(groups: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Write docs/api/<name>.md for each group.

    Returns a list of (nav_label, relative_path) for zensical.toml nav.
    """
    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old generated files.
    for old in DOCS_API_DIR.glob("*.md"):
        old.unlink()

    nav_entries: list[tuple[str, str]] = []

    for page_name, modules in sorted(groups.items()):
        md_file = DOCS_API_DIR / f"{page_name}.md"
        nav_label = f"openg2g.{page_name}"

        lines = [f"# {nav_label}", ""]
        for mod in modules:
            lines.append(f"::: {mod}")

        md_file.write_text("\n".join(lines) + "\n")
        nav_entries.append((nav_label, f"api/{page_name}.md"))

    return nav_entries


def update_zensical_nav(nav_entries: list[tuple[str, str]]) -> None:
    """Read _zensical.toml, add API Reference nav, write zensical.toml."""
    doc = tomlkit.parse(ZENSICAL_TOML_SRC.read_text())
    nav = doc["project"]["nav"]

    # Build the new API Reference nav entry.
    api_items = tomlkit.array()
    api_items.multiline(True)
    for label, path in nav_entries:
        api_items.append({label: path})
    api_ref = {"API Reference": api_items}

    # Remove any existing API Reference entry, then append fresh.
    for i, entry in enumerate(nav):
        if isinstance(entry, dict) and "API Reference" in entry:
            del nav[i]
            break
    nav.append(api_ref)

    ZENSICAL_TOML_OUT.write_text(tomlkit.dumps(doc))


def main() -> None:
    groups = discover_modules()
    nav_entries = write_api_pages(groups)
    update_zensical_nav(nav_entries)

    print(f"Generated {len(nav_entries)} API pages:")
    for label, path in nav_entries:
        print(f"  {path} -> {label}")


if __name__ == "__main__":
    main()
