import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def sanitize_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return cleaned or "unnamed"


def source_prefix(sources: list[Path], *, use_stem: bool) -> str:
    parts: list[str] = []
    for source in sources:
        raw = (source.stem if use_stem else source.name) or str(source)
        parts.append(sanitize_text(raw))
    return "_".join(parts)


def default_charts_dir(
    script_dir: Path,
    sources: list[Path],
    *,
    use_stem: bool,
) -> Path:
    return script_dir / "results" / "charts" / source_prefix(sources, use_stem=use_stem)


def source_labels(sources: list[Path], *, use_stem: bool) -> list[str]:
    labels = [((source.stem if use_stem else source.name) or str(source)) for source in sources]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, label in enumerate(labels):
        if counts[label] == 1:
            out.append(label)
            continue
        seen[label] = seen.get(label, 0) + 1
        out.append(f"{label} [{seen[label]}/{counts[label]}]")

    if len(set(out)) != len(out):
        out = [f"{label}#{idx + 1}" for idx, label in enumerate(labels)]
    return out


def plot_grouped_scalar_barh(
    *,
    models: list[str],
    group_labels: list[str],
    grouped_values: list[dict[str, Optional[float]]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    fig_width: float = 14.0,
) -> None:
    if not models:
        return

    y = np.arange(len(models), dtype=float)
    group_height = 0.82
    bar_h = group_height / max(len(grouped_values), 1)
    start = -group_height / 2 + bar_h / 2
    fig_h = max(5.0, 0.45 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    cmap = plt.get_cmap("tab10")

    for idx, (label, source_values) in enumerate(zip(group_labels, grouped_values)):
        x_vals = []
        y_vals = []
        for i, model in enumerate(models):
            value = source_values.get(model)
            if value is None:
                continue
            x_vals.append(float(value))
            y_vals.append(y[i] + start + idx * bar_h)
        if x_vals:
            ax.barh(
                y_vals,
                x_vals,
                height=bar_h * 0.95,
                label=label,
                color=cmap(idx % 10),
            )

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
