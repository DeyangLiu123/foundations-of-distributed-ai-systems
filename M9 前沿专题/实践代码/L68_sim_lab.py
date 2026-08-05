#!/usr/bin/env python3
"""L68 ASTRA-sim lab: hand model, analytical simulation, and trend plots.

The script has two execution paths:

1. Pure Python 3.9+ (standard library only): calculate the alpha-beta baseline
   and generate CSV/SVG artifacts on any CPU, including Apple Silicon.
2. ``--astra-root``: generate Chakra ET files, run ASTRA-sim's analytical
   backend, parse the reported cycles, and place the simulator curve beside
   the hand model.

The ASTRA-sim integration is pinned by the lesson to upstream commit
518bd513ae110428cd62eb60efc0f3993fd53c70 (checked 2026-08-05).  The script
also records the actual commit supplied by the user, because simulator inputs
and example paths evolve.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


MB = 1_000_000
GB = 1_000_000_000
PINNED_ASTRA_COMMIT = "518bd513ae110428cd62eb60efc0f3993fd53c70"
FINISHED_RE = re.compile(r"sys\[\d+\] finished,\s*(\d+) cycles")


@dataclass(frozen=True)
class Series:
    name: str
    points: Sequence[Tuple[float, float]]
    color: str
    dashed: bool = False


def ring_all_reduce_ns(message_bytes: int, ranks: int, alpha_ns: float, bandwidth_gb_s: float) -> float:
    """Return the L37 ring all-reduce alpha-beta prediction in nanoseconds."""
    if message_bytes <= 0:
        raise ValueError("message_bytes must be positive")
    if ranks < 2:
        raise ValueError("ranks must be at least 2")
    if alpha_ns < 0 or bandwidth_gb_s <= 0:
        raise ValueError("alpha must be non-negative and bandwidth positive")
    # Under SI units, 1 GB/s is exactly 1 B/ns.
    latency_term_ns = 2 * (ranks - 1) * alpha_ns
    bandwidth_term_ns = 2 * message_bytes * (ranks - 1) / ranks / bandwidth_gb_s
    return latency_term_ns + bandwidth_term_ns


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float, logarithmic: bool) -> float:
    if logarithmic:
        if value <= 0 or lo <= 0:
            raise ValueError("log axes require positive values")
        value, lo, hi = math.log10(value), math.log10(lo), math.log10(hi)
    if math.isclose(lo, hi):
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def write_svg(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    *,
    log_x: bool,
    log_y: bool,
) -> None:
    """Render a compact dependency-free line chart."""
    all_points = [point for item in series for point in item.points]
    if not all_points:
        raise ValueError("at least one plot point is required")
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min *= 0.9
        y_max *= 1.1

    width, height = 900, 540
    left, right, top, bottom = 92, 32, 58, 82
    plot_w, plot_h = width - left - right, height - top - bottom

    def px(x: float) -> float:
        return _scale(x, x_min, x_max, left, left + plot_w, log_x)

    def py(y: float) -> float:
        return _scale(y, y_min, y_max, top + plot_h, top, log_y)

    unique_x = sorted(set(xs))
    if len(unique_x) > 8:
        stride = math.ceil(len(unique_x) / 8)
        x_ticks = unique_x[::stride]
        if x_ticks[-1] != unique_x[-1]:
            x_ticks.append(unique_x[-1])
    else:
        x_ticks = unique_x

    if log_y:
        lo_exp = math.floor(math.log10(y_min))
        hi_exp = math.ceil(math.log10(y_max))
        y_ticks = [10.0**exp for exp in range(lo_exp, hi_exp + 1)]
        y_ticks = [tick for tick in y_ticks if y_min <= tick <= y_max]
        if len(y_ticks) < 2:
            y_ticks = [y_min, y_max]
    else:
        y_ticks = [y_min + index * (y_max - y_min) / 4 for index in range(5)]

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]

    for tick in y_ticks:
        y = py(tick)
        label = f"{tick:.3g}"
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>')

    for tick in x_ticks:
        x = px(tick)
        label = f"{tick:g}"
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 26}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="22" y="{top + plot_h / 2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 22 {top + plot_h / 2})">{html.escape(y_label)}</text>',
        ]
    )

    legend_x, legend_y = left + 12, top + 18
    for index, item in enumerate(series):
        dash = ' stroke-dasharray="8 5"' if item.dashed else ""
        points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in item.points)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{item.color}" stroke-width="2.5"{dash}/>')
        for x, y in item.points:
            lines.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3.5" fill="{item.color}"/>')
        ly = legend_y + 22 * index
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 28}" y2="{ly}" stroke="{item.color}" stroke-width="2.5"{dash}/>')
        lines.append(f'<text x="{legend_x + 36}" y="{ly + 4}" font-family="sans-serif" font-size="12">{html.escape(item.name)}</text>')

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class AstraRunner:
    """Generate a one-node Chakra ET per rank and run analytical ASTRA-sim."""

    def __init__(self, root: Path, out_dir: Path, splits: int) -> None:
        self.root = root.resolve()
        self.out_dir = out_dir.resolve()
        self.splits = splits
        if splits < 1:
            raise ValueError("splits must be at least 1")
        self.binary = self.root / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
        if not self.binary.exists():
            raise FileNotFoundError(
                f"ASTRA-sim binary not found: {self.binary}\n"
                "Run: bash build/astra_analytical/build.sh -t congestion_unaware"
            )
        self.commit = self._git_output(["rev-parse", "HEAD"])
        self.dirty = bool(self._git_output(["status", "--porcelain"]))
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        try:
            self.et_def = importlib.import_module(
                "extern.graph_frontend.chakra.schema.protobuf.et_def_pb2"
            )
            protolib = importlib.import_module(
                "extern.graph_frontend.chakra.src.third_party.utils.protolib"
            )
            self.encode_message = protolib.encodeMessage
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Chakra protobuf module is unavailable. The ASTRA-sim build script "
                "must finish before running this lab."
            ) from exc

    def _git_output(self, args: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _write_trace(self, prefix: Path, ranks: int, message_bytes: int) -> None:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for rank in range(ranks):
            path = Path(f"{prefix}.{rank}.et")
            with path.open("wb") as handle:
                self.encode_message(handle, self.et_def.GlobalMetadata(version="0.0.4"))
                node = self.et_def.Node()
                node.id = rank
                node.name = f"l68_all_reduce_{ranks}r_{message_bytes}B"
                node.type = self.et_def.COMM_COLL_NODE
                node.attr.append(self.et_def.AttributeProto(name="is_cpu_op", bool_val=False))
                node.attr.append(
                    self.et_def.AttributeProto(name="comm_type", int64_val=self.et_def.ALL_REDUCE)
                )
                node.attr.append(
                    self.et_def.AttributeProto(name="comm_size", int64_val=message_bytes)
                )
                self.encode_message(handle, node)

    def run(self, label: str, message_bytes: int, ranks: int, bandwidth_gb_s: float, alpha_ns: float) -> int:
        case_dir = self.out_dir / "astra-runs" / label
        case_dir.mkdir(parents=True, exist_ok=True)
        workload_prefix = case_dir / "workload" / "all_reduce"
        self._write_trace(workload_prefix, ranks, message_bytes)

        system = {
            "scheduling-policy": "LIFO",
            "endpoint-delay": 10,
            "active-chunks-per-dimension": 1,
            "preferred-dataset-splits": self.splits,
            "all-reduce-implementation": ["ring"],
            "all-gather-implementation": ["ring"],
            "reduce-scatter-implementation": ["ring"],
            "all-to-all-implementation": ["ring"],
            "collective-optimization": "localBWAware",
            "local-mem-bw": 3350,
            "boost-mode": 0,
            "roofline-enabled": 0,
            "peak-perf": 989,
        }
        system_path = case_dir / "system.json"
        system_path.write_text(json.dumps(system, indent=2) + "\n", encoding="utf-8")

        network_path = case_dir / "network.yml"
        network_path.write_text(
            "topology: [ Ring ]\n"
            f"npus_count: [ {ranks} ]\n"
            f"bandwidth: [ {bandwidth_gb_s:.9g} ]  # GB/s, per direction\n"
            f"latency: [ {alpha_ns:.9g} ]  # ns per modeled hop\n",
            encoding="utf-8",
        )
        memory_path = case_dir / "remote_memory.json"
        memory_path.write_text('{\n  "memory-type": "NO_MEMORY_EXPANSION"\n}\n', encoding="utf-8")

        command = [
            str(self.binary),
            f"--workload-configuration={workload_prefix}",
            f"--system-configuration={system_path}",
            f"--remote-memory-configuration={memory_path}",
            f"--network-configuration={network_path}",
        ]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        combined = result.stdout + "\n" + result.stderr
        (case_dir / "run.log").write_text(combined, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"ASTRA-sim failed for {label}; inspect {case_dir / 'run.log'}")
        cycles = [int(value) for value in FINISHED_RE.findall(combined)]
        if not cycles:
            raise RuntimeError(f"ASTRA-sim emitted no finished-cycle line for {label}")
        return max(cycles)


def run_task_b(out_dir: Path, runner: Optional[AstraRunner], splits: int) -> None:
    ranks = 8
    alpha_ns = 10_000.0
    bandwidth_gb_s = 50.0
    message_mb_values = [1, 4, 16, 64, 256, 1000]
    rows: List[Dict[str, object]] = []
    hand_points: List[Tuple[float, float]] = []
    astra_points: List[Tuple[float, float]] = []

    for message_mb in message_mb_values:
        message_bytes = message_mb * MB
        hand_ns = ring_all_reduce_ns(message_bytes, ranks, alpha_ns, bandwidth_gb_s)
        astra_ns: Optional[int] = None
        if runner:
            astra_ns = runner.run(
                f"task-b-{message_mb}MB",
                message_bytes,
                ranks,
                bandwidth_gb_s,
                alpha_ns,
            )
            astra_points.append((message_mb, astra_ns / 1e6))
        relative_error = (
            100 * abs(astra_ns - hand_ns) / hand_ns if astra_ns is not None else ""
        )
        rows.append(
            {
                "message_bytes": message_bytes,
                "message_mb_si": message_mb,
                "ranks": ranks,
                "alpha_us": alpha_ns / 1000,
                "bandwidth_gb_s_direction": bandwidth_gb_s,
                "splits": splits,
                "hand_ms": f"{hand_ns / 1e6:.9f}",
                "astra_ms": f"{astra_ns / 1e6:.9f}" if astra_ns is not None else "",
                "relative_error_pct": f"{relative_error:.6f}" if relative_error != "" else "",
                "astra_commit": runner.commit if runner else "not-run",
            }
        )
        hand_points.append((message_mb, hand_ns / 1e6))

    write_csv(
        out_dir / "task_b_calibration.csv",
        [
            "message_bytes",
            "message_mb_si",
            "ranks",
            "alpha_us",
            "bandwidth_gb_s_direction",
            "splits",
            "hand_ms",
            "astra_ms",
            "relative_error_pct",
            "astra_commit",
        ],
        rows,
    )
    chart_series = [Series("L37 hand model", hand_points, "#2563eb")]
    if astra_points:
        chart_series.append(Series("ASTRA-sim analytical", astra_points, "#dc2626", dashed=True))
    write_svg(
        out_dir / "task_b_calibration.svg",
        "Task B: ring all-reduce calibration",
        "message size (MB, SI)",
        "completion time (ms)",
        chart_series,
        log_x=True,
        log_y=True,
    )


def run_task_c(out_dir: Path, runner: Optional[AstraRunner], splits: int) -> None:
    # Llama-3-70B teaching shape from 03: b=1, S=8192, d=8192, BF16=2 B.
    message_bytes = 1 * 8192 * 8192 * 2
    alpha_ns = 10_000.0
    bandwidth_cases = [("NVLink ceiling", 450.0, "#059669"), ("400G line rate", 50.0, "#d97706")]
    ranks_values = [2, 4, 8]
    rows: List[Dict[str, object]] = []
    plot_series: List[Series] = []

    for bandwidth_name, bandwidth_gb_s, color in bandwidth_cases:
        hand_points: List[Tuple[float, float]] = []
        astra_points: List[Tuple[float, float]] = []
        for ranks in ranks_values:
            hand_ns = ring_all_reduce_ns(message_bytes, ranks, alpha_ns, bandwidth_gb_s)
            astra_ns: Optional[int] = None
            if runner:
                label = f"task-c-{ranks}r-{int(bandwidth_gb_s)}GBps"
                astra_ns = runner.run(label, message_bytes, ranks, bandwidth_gb_s, alpha_ns)
                astra_points.append((ranks, 4 * astra_ns / 1e6))
            rows.append(
                {
                    "model": "Llama-3-70B teaching shape",
                    "tp": ranks,
                    "message_bytes": message_bytes,
                    "message_mb_si": f"{message_bytes / MB:.6f}",
                    "bandwidth_case": bandwidth_name,
                    "bandwidth_gb_s_direction": bandwidth_gb_s,
                    "alpha_us": alpha_ns / 1000,
                    "splits": splits,
                    "hand_one_allreduce_ms": f"{hand_ns / 1e6:.9f}",
                    "hand_four_per_layer_ms": f"{4 * hand_ns / 1e6:.9f}",
                    "hand_320_per_step_ms_no_overlap": f"{320 * hand_ns / 1e6:.9f}",
                    "astra_one_allreduce_ms": f"{astra_ns / 1e6:.9f}" if astra_ns is not None else "",
                    "astra_four_per_layer_ms": f"{4 * astra_ns / 1e6:.9f}" if astra_ns is not None else "",
                    "astra_commit": runner.commit if runner else "not-run",
                }
            )
            hand_points.append((ranks, 4 * hand_ns / 1e6))
        plot_series.append(Series(f"{bandwidth_name} / hand", hand_points, color))
        if astra_points:
            plot_series.append(Series(f"{bandwidth_name} / ASTRA", astra_points, color, dashed=True))

    write_csv(
        out_dir / "task_c_strategy.csv",
        [
            "model",
            "tp",
            "message_bytes",
            "message_mb_si",
            "bandwidth_case",
            "bandwidth_gb_s_direction",
            "alpha_us",
            "splits",
            "hand_one_allreduce_ms",
            "hand_four_per_layer_ms",
            "hand_320_per_step_ms_no_overlap",
            "astra_one_allreduce_ms",
            "astra_four_per_layer_ms",
            "astra_commit",
        ],
        rows,
    )
    write_svg(
        out_dir / "task_c_strategy.svg",
        "Task C: Llama-3-70B TP communication trend",
        "TP ranks",
        "four all-reduces per layer (ms)",
        plot_series,
        log_x=True,
        log_y=True,
    )


def self_test() -> None:
    one_mb = ring_all_reduce_ns(MB, 8, 10_000, 50)
    one_gb = ring_all_reduce_ns(GB, 8, 10_000, 50)
    assert math.isclose(one_mb / 1e6, 0.175, abs_tol=1e-12)
    assert math.isclose(one_gb / 1e6, 35.14, abs_tol=1e-12)
    assert math.isclose(ring_all_reduce_ns(MB, 4, 0, 25), 60_000, abs_tol=1e-9)
    with tempfile.TemporaryDirectory(prefix="l68-self-test-") as temporary:
        path = Path(temporary) / "chart.svg"
        write_svg(
            path,
            "self-test",
            "x",
            "y",
            [Series("line", [(1, 1), (10, 10)], "#000000")],
            log_x=True,
            log_y=True,
        )
        assert path.read_text(encoding="utf-8").startswith("<svg")
    print("[PASS] alpha-beta accounts and dependency-free SVG rendering")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astra-root", type=Path, help="built ASTRA-sim source root")
    parser.add_argument("--out-dir", type=Path, default=Path("results/l68"))
    parser.add_argument("--task", choices=("all", "b", "c"), default="all")
    parser.add_argument(
        "--splits",
        type=int,
        default=1,
        help="ASTRA preferred-dataset-splits; use 1 for the direct hand-model comparison",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.splits < 1:
        parser.error("--splits must be at least 1")

    out_dir = args.out_dir.resolve()
    runner = AstraRunner(args.astra_root, out_dir, args.splits) if args.astra_root else None
    if runner:
        environment = {
            "pinned_commit": PINNED_ASTRA_COMMIT,
            "actual_commit": runner.commit,
            "dirty_worktree": runner.dirty,
            "splits": args.splits,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "environment.json").write_text(
            json.dumps(environment, indent=2) + "\n", encoding="utf-8"
        )
        if runner.commit != PINNED_ASTRA_COMMIT:
            print(
                f"[WARN] ASTRA-sim commit is {runner.commit[:12]}, not the lesson pin "
                f"{PINNED_ASTRA_COMMIT[:12]}; results remain usable if the report records this drift."
            )
    else:
        print("[INFO] --astra-root omitted: generating the hand-model baseline only.")

    if args.task in ("all", "b"):
        run_task_b(out_dir, runner, args.splits)
    if args.task in ("all", "c"):
        run_task_c(out_dir, runner, args.splits)
    print(f"[PASS] wrote L68 artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
