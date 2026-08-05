#!/usr/bin/env python3
"""L62: turn one or more benchmark CSV files into self-contained SVG plots.

Only the Python 3.10+ standard library is required.  SVG keeps the figures
readable in Obsidian without adding a plotting dependency to the practice.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import tempfile
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Callable, Iterable


COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def load_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(row for row in csv.DictReader(handle) if row.get("success") == "true")
    return rows


def mean_points(
    rows: list[dict[str, str]],
    *,
    task: str,
    x_key: str,
    y_key: str,
    x_display: Callable[[list[dict[str, str]], float], float] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("task") != task:
            continue
        x = as_float(row.get(x_key))
        y = as_float(row.get(y_key))
        if x is None or y is None:
            continue
        groups[(row.get("label", "default"), x)].append(row)

    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (label, x), group_rows in groups.items():
        values = [as_float(row.get(y_key)) for row in group_rows]
        clean_values = [value for value in values if value is not None]
        shown_x = x_display(group_rows, x) if x_display else x
        result[label].append((shown_x, statistics.fmean(clean_values)))
    return {label: sorted(points) for label, points in result.items()}


def group_metric_points(
    rows: list[dict[str, str]], y_key: str
) -> dict[str, list[tuple[float, float]]]:
    # group_* fields repeat on every request row.  Deduplicate by group_id before
    # averaging repeats, otherwise high-concurrency groups receive extra weight.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("task") == "concurrency":
            unique[(row.get("label", "default"), row.get("group_id", ""))] = row
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for (label, _), row in unique.items():
        x = as_float(row.get("target_concurrency"))
        y = as_float(row.get(y_key))
        if x is not None and y is not None:
            grouped[(label, x)].append(y)
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (label, x), values in grouped.items():
        result[label].append((x, statistics.fmean(values)))
    return {label: sorted(points) for label, points in result.items()}


def xml_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}'
        '.axis{stroke:#111827;stroke-width:1.4}.grid{stroke:#e5e7eb;stroke-width:1}'
        '.tick{fill:#4b5563;font-size:12px}.label{fill:#111827;font-size:14px}'
        '.title{fill:#111827;font-size:18px;font-weight:600}</style>',
    ]


def bounds(series: dict[str, list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    points = [point for values in series.values() for point in values]
    if not points:
        raise ValueError("no data points")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min = max(0.0, y_min - 0.5)
        y_max += 0.5
    y_padding = (y_max - y_min) * 0.08
    return x_min, x_max, max(0.0, y_min - y_padding), y_max + y_padding


def chart_elements(
    series: dict[str, list[tuple[float, float]]],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    x_label: str,
    y_label: str,
) -> list[str]:
    x_min, x_max, y_min, y_max = bounds(series)
    left, right, top, bottom = 78.0, 24.0, 48.0, 58.0
    plot_x = x0 + left
    plot_y = y0 + top
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        return plot_x + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return plot_y + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    out = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 25:.1f}" text-anchor="middle" '
        f'class="title">{escape(title)}</text>'
    ]
    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = sx(x_value)
        out.append(
            f'<line x1="{x:.1f}" y1="{plot_y:.1f}" x2="{x:.1f}" '
            f'y2="{plot_y + plot_h:.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{plot_y + plot_h + 22:.1f}" '
            f'text-anchor="middle" class="tick">{x_value:.4g}</text>'
        )
        y_value = y_min + fraction * (y_max - y_min)
        y = sy(y_value)
        out.append(
            f'<line x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_w:.1f}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{plot_x - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="tick">{y_value:.4g}</text>'
        )
    out.extend(
        [
            f'<line x1="{plot_x:.1f}" y1="{plot_y + plot_h:.1f}" '
            f'x2="{plot_x + plot_w:.1f}" y2="{plot_y + plot_h:.1f}" class="axis"/>',
            f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" '
            f'y2="{plot_y + plot_h:.1f}" class="axis"/>',
            f'<text x="{plot_x + plot_w / 2:.1f}" y="{y0 + height - 10:.1f}" '
            f'text-anchor="middle" class="label">{escape(x_label)}</text>',
            f'<text x="{x0 + 18:.1f}" y="{plot_y + plot_h / 2:.1f}" '
            f'transform="rotate(-90 {x0 + 18:.1f} {plot_y + plot_h / 2:.1f})" '
            f'text-anchor="middle" class="label">{escape(y_label)}</text>',
        ]
    )

    legend_x = plot_x + 8
    legend_y = plot_y + 16
    for index, (label, points) in enumerate(sorted(series.items())):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        out.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="2.5"/>'
        )
        for x, y in points:
            out.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>'
            )
        out.append(
            f'<line x1="{legend_x:.1f}" y1="{legend_y + index * 19:.1f}" '
            f'x2="{legend_x + 22:.1f}" y2="{legend_y + index * 19:.1f}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        out.append(
            f'<text x="{legend_x + 28:.1f}" y="{legend_y + 4 + index * 19:.1f}" '
            f'class="tick">{escape(label)}</text>'
        )
    return out


def write_single_chart(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    parts = xml_header(900, 520)
    parts.extend(
        chart_elements(
            series,
            x0=0,
            y0=0,
            width=900,
            height=520,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_tradeoff_chart(
    path: Path,
    throughput: dict[str, list[tuple[float, float]]],
    tpot: dict[str, list[tuple[float, float]]],
) -> None:
    parts = xml_header(900, 920)
    parts.extend(
        chart_elements(
            throughput,
            x0=0,
            y0=0,
            width=900,
            height=450,
            title="Decode throughput vs concurrency",
            x_label="Concurrent requests",
            y_label="Aggregate decode tok/s",
        )
    )
    parts.extend(
        chart_elements(
            tpot,
            x0=0,
            y0=460,
            width=900,
            height=450,
            title="TPOT cost of concurrency",
            x_label="Concurrent requests",
            y_label="Mean TPOT (ms)",
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def generate(inputs: list[Path], out_dir: Path) -> list[Path]:
    rows = load_rows(inputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def actual_prompt_x(group_rows: list[dict[str, str]], fallback: float) -> float:
        values = [as_float(row.get("prompt_tokens")) for row in group_rows]
        clean = [value for value in values if value is not None]
        return statistics.fmean(clean) if clean else fallback

    length_ttft = mean_points(
        rows,
        task="length",
        x_key="target_prompt_words",
        y_key="ttft_ms",
        x_display=actual_prompt_x,
    )
    length_tpot = mean_points(
        rows,
        task="length",
        x_key="target_prompt_words",
        y_key="tpot_ms",
        x_display=actual_prompt_x,
    )
    if length_ttft:
        path = out_dir / "L62_prompt_length_TTFT.svg"
        write_single_chart(path, length_ttft, "TTFT vs prompt length", "Prompt tokens", "Mean TTFT (ms)")
        written.append(path)
    if length_tpot:
        path = out_dir / "L62_prompt_length_TPOT.svg"
        write_single_chart(path, length_tpot, "TPOT vs prompt length", "Prompt tokens", "Mean TPOT (ms)")
        written.append(path)

    group_decode = group_metric_points(rows, "group_decode_tps")
    concurrency_tpot = mean_points(
        rows,
        task="concurrency",
        x_key="target_concurrency",
        y_key="tpot_ms",
    )
    if group_decode and concurrency_tpot:
        path = out_dir / "L62_concurrency_tradeoff.svg"
        write_tradeoff_chart(path, group_decode, concurrency_tpot)
        written.append(path)

    prefix_ttft = mean_points(
        rows,
        task="prefix",
        x_key="run_id",
        y_key="ttft_ms",
    )
    if prefix_ttft:
        path = out_dir / "L62_prefix_cache_TTFT.svg"
        write_single_chart(path, prefix_ttft, "Prefix replay TTFT", "Replay index (0 = cold)", "TTFT (ms)")
        written.append(path)
    return written


def self_test() -> None:
    fields = [
        "task,label,group_id,run_id,target_prompt_words,target_concurrency,"
        "prompt_tokens,output_tokens,token_count_source,ttft_ms,tpot_ms,e2e_ms,"
        "request_decode_tps,group_output_tps,group_decode_tps,group_makespan_ms,"
        "success,error\n"
    ]
    for label in ("cache_on", "cache_off"):
        for index, prompt in enumerate((128, 512, 2048)):
            fields.append(
                f"length,{label},l-{index},{index},{prompt},1,{prompt + 8},64,/tokenize,"
                f"{20 + index * 10},{5 + index},100,200,,,,true,\n"
            )
        for concurrency in (1, 2, 4):
            fields.append(
                f"concurrency,{label},c-{concurrency},0,512,{concurrency},520,64,"
                f"/tokenize,30,{5 + concurrency},100,200,300,{100 * concurrency},100,true,\n"
            )
        for replay in range(3):
            fields.append(
                f"prefix,{label},p,{replay},2048,1,2050,64,/tokenize,"
                f"{100 - replay * 20},5,200,200,,,,true,\n"
            )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_path = root / "sample.csv"
        csv_path.write_text("".join(fields), encoding="utf-8")
        outputs = generate([csv_path], root / "figures")
        assert len(outputs) == 4
        assert all("<svg" in path.read_text(encoding="utf-8") for path in outputs)
    print("[PASS] CSV aggregation and four SVG outputs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("l62-figures"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.inputs:
        parser.error("provide at least one CSV input, or use --self-test")
    outputs = generate(args.inputs, args.out_dir)
    if not outputs:
        raise SystemExit("no plottable successful rows found")
    for path in outputs:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
