#!/usr/bin/env python3
"""L48 parallel-plan calculator: memory, communication, and bubble accounts.

The formulas are teaching estimates from L37/L40/L42/L44/L45/L47.  They
screen candidates; they do not predict framework peak memory or wall time.
Only Python 3.9+ and the standard library are required.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


GB = 1_000_000_000


@dataclass(frozen=True)
class Model:
    name: str
    params: float
    layers: int
    hidden: int


@dataclass(frozen=True)
class Cluster:
    name: str
    gpus: int
    gpus_per_node: int
    memory_gb: float
    scaleup_gb_s_direction: float
    scaleout_gbps_direction: float


@dataclass(frozen=True)
class Workload:
    seq: int
    global_batch: int
    micro_batch: int = 1


@dataclass(frozen=True)
class Plan:
    name: str
    tp: int
    pp: int
    cp: int
    dp: int
    zero_stage: int = 0
    recompute: str = "selective"
    sequence_parallel: bool = True


@dataclass(frozen=True)
class Case:
    model: Model
    cluster: Cluster
    workload: Workload
    plan: Plan


@dataclass
class Estimate:
    legal: bool
    errors: List[str]
    warnings: List[str]
    ga: int
    bubble_ratio: float
    static_gb: float
    activation_gb: float
    accounted_gb: float
    headroom_gb: float
    memory_pass: bool
    tp_gb: float
    pp_boundary_gb: float
    dp_zero_gb: float
    cp_gb: float
    tp_lower_bound_s: float
    pp_lower_bound_s: float
    dp_zero_lower_bound_s: float
    cp_lower_bound_s: float


LLAMA3_8B = Model("Llama-3-8B teaching shape", 8e9, 32, 4096)
LLAMA3_70B = Model("Llama-3-70B teaching shape", 70e9, 80, 8192)
H100_8 = Cluster("1 node × 8 H100", 8, 8, 80, 450, 400)
H100_512 = Cluster("64 nodes × 8 H100", 512, 8, 80, 450, 400)
H100_512_100G = Cluster("64 nodes × 8 H100 / 100G", 512, 8, 80, 450, 100)
H800_512 = Cluster("64 nodes × 8 H800", 512, 8, 80, 200, 400)


def case(model: Model, cluster: Cluster, seq: int, gbs: int, **plan: Any) -> Case:
    return Case(model, cluster, Workload(seq, gbs), Plan(**plan))


PRESETS: Dict[str, Case] = {
    "a_zero2": case(LLAMA3_8B, H100_8, 8192, 8, name="A / ZeRO-2 DP8", tp=1, pp=1, cp=1, dp=8, zero_stage=2, sequence_parallel=False),
    "a_zero3": case(LLAMA3_8B, H100_8, 8192, 8, name="A / ZeRO-3 DP8", tp=1, pp=1, cp=1, dp=8, zero_stage=3, sequence_parallel=False),
    "a_tp8": case(LLAMA3_8B, H100_8, 8192, 8, name="A / TP8", tp=8, pp=1, cp=1, dp=1),
    "b_tp8_pp8": case(LLAMA3_70B, H100_512, 8192, 1024, name="B / TP8-PP8-DP8", tp=8, pp=8, cp=1, dp=8),
    "b_tp8_pp4": case(LLAMA3_70B, H100_512, 8192, 1024, name="B / TP8-PP4-DP16", tp=8, pp=4, cp=1, dp=16),
    "b_tp4_pp8": case(LLAMA3_70B, H100_512, 8192, 1024, name="B / TP4-PP8-DP16", tp=4, pp=8, cp=1, dp=16),
    "b_tp8_pp8_100g": case(LLAMA3_70B, H100_512_100G, 8192, 1024, name="B+ / TP8-PP8-DP8 / 100G", tp=8, pp=8, cp=1, dp=8),
    "c_no_cp": case(LLAMA3_70B, H800_512, 131072, 1024, name="C / TP8-PP8-DP8-CP1", tp=8, pp=8, cp=1, dp=8),
    "c_cp8": case(LLAMA3_70B, H800_512, 131072, 1024, name="C / TP8-PP8-CP8-DP1", tp=8, pp=8, cp=8, dp=1),
    "c_tp4_cp16": case(LLAMA3_70B, H800_512, 131072, 1024, name="C / TP4-PP8-CP16-DP1", tp=4, pp=8, cp=16, dp=1),
}


def estimate(c: Case) -> Estimate:
    m, cl, w, p = c.model, c.cluster, c.workload, c.plan
    errors: List[str] = []
    warnings: List[str] = []

    degrees = (p.tp, p.pp, p.cp, p.dp)
    if any(x < 1 for x in degrees):
        errors.append("TP/PP/CP/DP degrees must all be positive")
    if math.prod(degrees) != cl.gpus:
        errors.append(f"degree product {math.prod(degrees)} != cluster GPUs {cl.gpus}")
    if m.layers % p.pp:
        errors.append(f"layers {m.layers} are not evenly divisible by PP={p.pp}")
    if p.zero_stage not in (0, 1, 2, 3):
        errors.append("zero_stage must be 0, 1, 2, or 3")
    if p.recompute not in ("selective", "full"):
        errors.append("recompute must be selective or full")
    batch_denominator = w.micro_batch * p.dp
    if w.global_batch % batch_denominator:
        errors.append("global_batch must be divisible by micro_batch × DP")
    ga = w.global_batch // batch_denominator if batch_denominator else 0
    if ga < 1:
        errors.append("gradient accumulation must be at least 1")
    if p.tp > cl.gpus_per_node:
        warnings.append("TP crosses the node boundary; re-check the scale-up domain")

    # L40/L42: 16 B/parameter; TP and PP shard model states, CP does not.
    local_params = m.params / (p.tp * p.pp)
    if p.zero_stage == 0:
        static_bytes = 16 * local_params
    elif p.zero_stage == 1:
        static_bytes = local_params * (4 + 12 / p.dp)
    elif p.zero_stage == 2:
        static_bytes = local_params * (2 + 14 / p.dp)
    else:
        static_bytes = 16 * local_params / p.dp

    # L40/L47: selective estimate 34SbdL; full recompute keeps layer boundaries.
    activation_shards = p.cp * (p.tp if p.sequence_parallel else 1)
    activation_factor = 34 if p.recompute == "selective" else 2
    activation_bytes = (
        activation_factor * w.seq * w.micro_batch * m.hidden * m.layers
        / activation_shards
    )

    static_gb = static_bytes / GB
    activation_gb = activation_bytes / GB
    accounted_gb = static_gb + activation_gb
    headroom_gb = cl.memory_gb - accounted_gb
    memory_pass = headroom_gb >= 0

    bubble = (p.pp - 1) / (ga + p.pp - 1) if p.pp > 1 else 0.0
    local_seq = w.seq / p.cp
    local_layers = m.layers / p.pp

    # L43/L47: Megatron-SP TP ring-equivalent, per rank per direction per step.
    tp_bytes = 0.0
    if p.tp > 1:
        tp_bytes = (
            16 * (p.tp - 1) / p.tp
            * local_seq * w.micro_batch * m.hidden * local_layers * ga
        )

    # L44: one BF16 hidden tensor per PP boundary and forward direction.
    pp_bytes = 0.0
    if p.pp > 1:
        pp_bytes = 2 * local_seq * w.micro_batch * m.hidden * ga

    # L37/L42/L47: DDP/ZeRO-1/2 are 2W; ZeRO-3 reshard is (2GA+1)W.
    dp_zero_bytes = 0.0
    if p.dp > 1:
        weight_bytes = 2 * local_params
        ring_fraction = (p.dp - 1) / p.dp
        if p.zero_stage == 3:
            dp_zero_bytes = (2 * ga + 1) * weight_bytes * ring_fraction
        else:
            dp_zero_bytes = 2 * weight_bytes * ring_fraction

    # L45: Ring Attention forward KV send bytes; receive has the same volume.
    cp_bytes = 0.0
    if p.cp > 1:
        cp_per_layer_microbatch = 4 * local_seq * w.micro_batch * m.hidden * (p.cp - 1)
        cp_bytes = cp_per_layer_microbatch * local_layers * ga

    scaleout_gb_s = cl.scaleout_gbps_direction / 8
    outer_gb_s = cl.scaleup_gb_s_direction if cl.gpus <= cl.gpus_per_node else scaleout_gb_s
    lower_bound = lambda byte_count, gb_s: byte_count / GB / gb_s if byte_count else 0.0

    warnings.append("accounted memory excludes workspaces, buffers, fragmentation, and ZeRO gathered units")
    warnings.append("communication times are separate payload-only lower bounds; do not add them as step time")
    return Estimate(
        legal=not errors,
        errors=errors,
        warnings=warnings,
        ga=ga,
        bubble_ratio=bubble,
        static_gb=static_gb,
        activation_gb=activation_gb,
        accounted_gb=accounted_gb,
        headroom_gb=headroom_gb,
        memory_pass=memory_pass,
        tp_gb=tp_bytes / GB,
        pp_boundary_gb=pp_bytes / GB,
        dp_zero_gb=dp_zero_bytes / GB,
        cp_gb=cp_bytes / GB,
        tp_lower_bound_s=lower_bound(tp_bytes, cl.scaleup_gb_s_direction),
        pp_lower_bound_s=lower_bound(pp_bytes, outer_gb_s),
        dp_zero_lower_bound_s=lower_bound(dp_zero_bytes, outer_gb_s),
        cp_lower_bound_s=lower_bound(cp_bytes, outer_gb_s),
    )


def print_report(c: Case, e: Estimate) -> None:
    p = c.plan
    print(f"[{p.name}]  model={c.model.name}  cluster={c.cluster.name}")
    print(f"mesh: TP{p.tp} × PP{p.pp} × CP{p.cp} × DP{p.dp}; ZeRO-{p.zero_stage}; legal={e.legal}")
    for item in e.errors:
        print(f"  ERROR: {item}")
    print(f"batch: GBS={c.workload.global_batch}, micro={c.workload.micro_batch}, GA={e.ga}")
    print(f"bubble: {100 * e.bubble_ratio:.2f}%")
    print(
        "memory: "
        f"static={e.static_gb:.2f} GB, activation={e.activation_gb:.2f} GB, "
        f"accounted={e.accounted_gb:.2f} GB, headroom={e.headroom_gb:.2f} GB, "
        f"screen={'PASS' if e.memory_pass else 'FAIL'}"
    )
    print("communication (per rank, per direction, per optimizer step):")
    print(f"  TP={e.tp_gb:.2f} GB  payload-LB={e.tp_lower_bound_s:.3f} s")
    print(f"  PP-boundary={e.pp_boundary_gb:.2f} GB  payload-LB={e.pp_lower_bound_s:.3f} s")
    print(f"  DP/ZeRO={e.dp_zero_gb:.2f} GB  payload-LB={e.dp_zero_lower_bound_s:.3f} s")
    print(f"  CP-Ring-forward={e.cp_gb:.2f} GB  payload-LB={e.cp_lower_bound_s:.3f} s")
    for item in e.warnings:
        print(f"  NOTE: {item}")


def load_case(path: Path) -> Case:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Case(
        Model(**raw["model"]), Cluster(**raw["cluster"]),
        Workload(**raw["workload"]), Plan(**raw["plan"]),
    )


def self_test() -> None:
    expected = {
        "a_zero2": (30.00, 36.51, 0.0),
        "a_zero3": (16.00, 36.51, 0.0),
        "a_tp8": (16.00, 4.56, 0.0),
        "b_tp8_pp8": (17.50, 22.82, 5.19),
        "b_tp8_pp4": (35.00, 22.82, 4.48),
        "c_no_cp": (17.50, 365.07, 5.19),
        "c_cp8": (17.50, 45.63, 0.68),
    }
    for name, values in expected.items():
        e = estimate(PRESETS[name])
        got = (e.static_gb, e.activation_gb, 100 * e.bubble_ratio)
        assert e.legal, (name, e.errors)
        assert all(math.isclose(a, b, abs_tol=0.01) for a, b in zip(got, values)), (name, got)
    assert not estimate(PRESETS["b_tp4_pp8"]).memory_pass
    assert not estimate(PRESETS["c_tp4_cp16"]).memory_pass
    assert math.isclose(estimate(PRESETS["a_zero2"]).dp_zero_gb, 28.00, abs_tol=0.01)
    assert math.isclose(estimate(PRESETS["a_zero3"]).dp_zero_gb, 42.00, abs_tol=0.01)
    assert math.isclose(estimate(PRESETS["a_tp8"]).tp_gb, 120.26, abs_tol=0.01)
    b1 = estimate(PRESETS["b_tp8_pp8"])
    assert math.isclose(b1.tp_gb, 1202.59, abs_tol=0.01)
    assert math.isclose(b1.pp_boundary_gb, 17.18, abs_tol=0.01)
    assert math.isclose(b1.dp_zero_gb, 3.83, abs_tol=0.01)
    assert math.isclose(estimate(PRESETS["c_cp8"]).cp_gb, 38482.91, abs_tol=0.01)
    slow = estimate(PRESETS["b_tp8_pp8_100g"])
    fast = estimate(PRESETS["b_tp8_pp8"])
    assert math.isclose(slow.pp_lower_bound_s / fast.pp_lower_bound_s, 4.0)
    assert math.isclose(slow.dp_zero_lower_bound_s / fast.dp_zero_lower_bound_s, 4.0)
    print("[PASS] preset memory/batch/communication checks matched the hand-derived accounts.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=sorted(PRESETS), default="b_tp8_pp8")
    source.add_argument("--config", type=Path, help="JSON file with model/cluster/workload/plan objects")
    parser.add_argument("--dump-config", type=Path, help="write the selected case as JSON")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    selected = load_case(args.config) if args.config else PRESETS[args.preset]
    if args.dump_config:
        args.dump_config.write_text(json.dumps(asdict(selected), indent=2) + "\n", encoding="utf-8")
    result = estimate(selected)
    print_report(selected, result)
    return 0 if result.legal else 2


if __name__ == "__main__":
    raise SystemExit(main())
