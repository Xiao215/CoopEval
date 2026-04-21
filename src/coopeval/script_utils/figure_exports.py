#!/usr/bin/env python3
"""Shared helpers for saving plot outputs into format-specific folders."""

from pathlib import Path
from typing import Any, Sequence

from matplotlib.figure import Figure


def _normalize_formats(formats: Sequence[str]) -> list[str]:
    """Return a deduplicated, lowercase list of requested output formats."""
    ordered: list[str] = []
    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower not in ordered:
            ordered.append(fmt_lower)
    return ordered


def _resolve_relative(
    base_path: Path, root_dir: Path | None
) -> tuple[Path, Path]:
    """Return (root, relative_path) ensuring the base lives under root_dir."""
    path = Path(base_path)
    if root_dir is None:
        return path.parent, Path(path.name)
    root = Path(root_dir)
    try:
        relative = path.relative_to(root)
    except ValueError as err:
        raise ValueError(
            f"Base path {path} must be within root directory {root}"
        ) from err
    return root, relative


def save_matplotlib_figure(
    fig: Figure,
    base_path: Path,
    formats: Sequence[str],
    *,
    dpi: int = 300,
    root_dir: Path | None = None,
    format_subdirs: bool = True,
    **savefig_kwargs,
) -> list[Path]:
    """Save a Matplotlib figure for each requested format.

    By default, files are written under <root>/<format>/<relative_path>.<format>.
    Set format_subdirs=False to write directly beside base_path as
    <base_path>.<format>.
    """
    normalized = _normalize_formats(formats)
    saved_paths: list[Path] = []
    base_parent, relative_path = _resolve_relative(base_path, root_dir)
    base_parent.mkdir(parents=True, exist_ok=True)
    for fmt in normalized:
        out_dir = base_parent / fmt if format_subdirs else base_parent
        dest = (out_dir / relative_path).with_suffix(f".{fmt}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=dpi, **savefig_kwargs)
        saved_paths.append(dest)
    return saved_paths


def save_plotly_figure(
    fig: Any,
    base_path: Path,
    formats: Sequence[str],
    *,
    root_dir: Path | None = None,
) -> list[Path]:
    """Save a Plotly figure to the requested formats (html/pdf/png)."""
    normalized = _normalize_formats(formats)
    saved_paths: list[Path] = []
    base_parent, relative_path = _resolve_relative(base_path, root_dir)
    base_parent.mkdir(parents=True, exist_ok=True)
    for fmt in normalized:
        out_dir = base_parent / fmt
        dest = (out_dir / relative_path).with_suffix(f".{fmt}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "html":
            fig.write_html(dest)
        else:
            try:
                fig.write_image(dest)
            except ValueError as err:
                raise RuntimeError(
                    "Plotly static image export requires the 'kaleido' package. "
                    "Install it (pip install kaleido) or include 'html' in --formats."
                ) from err
        saved_paths.append(dest)
    return saved_paths
