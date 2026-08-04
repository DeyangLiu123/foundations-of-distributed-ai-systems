"""L39：在单机多进程上验证 collective 语义，并测量有效 alpha-beta 曲线。

依赖：Python 3.9-3.12，torch==2.5.1；numpy==1.26.4 只用于避免 PyTorch
可选依赖提示。较新 PyTorch 也可运行。CPU / Apple Silicon 示例：

    torchrun --standalone --nproc_per_node=8 L39_collectives.py --backend gloo

8 张 NVIDIA GPU 示例：

    torchrun --standalone --nproc_per_node=8 L39_collectives.py --backend nccl

Gloo 不原生支持 all-to-all；脚本会明确标注并用 all-gather 加本地选取复现
相同输入输出语义。旧版 Gloo 也不支持 reduce-scatter，脚本会用
all-reduce 加本地切片作兼容后备。后备路径只验证语义，不代表原生算法性能。
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class Measurement:
    nbytes: int
    seconds: float
    repeats: int

    @property
    def microseconds(self) -> float:
        return self.seconds * 1e6

    @property
    def algbw_gb_s(self) -> float:
        """按用户 tensor 大小 n/T 计算；不是 nccl-tests 的 busbw。"""
        return self.nbytes / self.seconds / 1e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument(
        "--task",
        choices=("all", "semantics", "duality", "benchmark", "straggler"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/l39"))
    parser.add_argument("--min-bytes", type=int, default=1_000)
    parser.add_argument("--max-bytes", type=int, default=64_000_000)
    parser.add_argument("--small-max-bytes", type=int, default=16_000)
    parser.add_argument("--large-min-bytes", type=int, default=4_000_000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=50)
    parser.add_argument("--target-bytes", type=int, default=64_000_000)
    parser.add_argument("--straggler-ms", type=float, default=5.0)
    parser.add_argument("--straggler-rank", type=int, default=1)
    parser.add_argument("--straggler-iters", type=int, default=10)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", "0")),
    )
    args = parser.parse_args()

    positive = {
        "min-bytes": args.min_bytes,
        "max-bytes": args.max_bytes,
        "small-max-bytes": args.small_max_bytes,
        "large-min-bytes": args.large_min_bytes,
        "min-repeats": args.min_repeats,
        "max-repeats": args.max_repeats,
        "target-bytes": args.target_bytes,
        "straggler-iters": args.straggler_iters,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error("%s 必须为正数" % name)
    if args.warmup < 0 or args.straggler_ms < 0:
        parser.error("warmup 和 straggler-ms 不能为负数")
    if args.min_bytes > args.max_bytes:
        parser.error("min-bytes 不能大于 max-bytes")
    if args.min_repeats > args.max_repeats:
        parser.error("min-repeats 不能大于 max-repeats")
    if args.min_bytes % 4 or args.max_bytes % 4:
        parser.error("min-bytes 和 max-bytes 必须是 FP32 元素大小 4 B 的整数倍")
    return args


def resolve_backend(requested: str, local_rank: int) -> Tuple[str, torch.device]:
    cuda_ready = (
        torch.cuda.is_available()
        and dist.is_nccl_available()
        and local_rank < torch.cuda.device_count()
    )
    backend = "nccl" if requested == "auto" and cuda_ready else requested
    if backend == "auto":
        backend = "gloo"

    if backend == "nccl":
        if not cuda_ready:
            raise RuntimeError(
                "NCCL 路径要求每个 local rank 有一张可用 NVIDIA GPU；请改用 --backend gloo"
            )
        torch.cuda.set_device(local_rank)
        return backend, torch.device("cuda", local_rank)

    if not dist.is_gloo_available():
        raise RuntimeError("当前 PyTorch 构建不包含 Gloo")
    # Gloo 实验使用 CPU tensor；Apple Silicon 的 MPS 不作为 Gloo 通信设备。
    torch.set_num_threads(1)
    return "gloo", torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ordered_print(message: str) -> None:
    rank = dist.get_rank()
    for owner in range(dist.get_world_size()):
        dist.barrier()
        if rank == owner:
            print(message, flush=True)
        dist.barrier()


def reduce_scatter_sum(input_tensor: Tensor) -> Tuple[Tensor, str]:
    """优先执行原生 RS；旧版 Gloo 不支持时，用 AR+slice 复现语义。"""
    world_size = dist.get_world_size()
    if input_tensor.numel() % world_size:
        raise ValueError("reduce-scatter 输入元素数必须能被 world_size 整除")
    output = torch.empty(
        input_tensor.numel() // world_size,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    try:
        dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM)
        return output, "native reduce_scatter_tensor"
    except RuntimeError as exc:
        message = str(exc).lower()
        if str(dist.get_backend()).lower() != "gloo" or "support" not in message:
            raise

    reduced = input_tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    rank = dist.get_rank()
    chunk = output.numel()
    output.copy_(reduced.narrow(0, rank * chunk, chunk))
    return output, "fallback all_reduce + local slice"


def all_to_all_portable(input_tensor: Tensor) -> Tuple[Tensor, str]:
    """NCCL 走原生 A2A；Gloo 用 AG+本地选取复现等长 split 语义。"""
    world_size = dist.get_world_size()
    if input_tensor.numel() != world_size:
        raise ValueError("本课 all-to-all 示例要求每个目的 rank 恰好接收 1 个元素")
    output = torch.empty_like(input_tensor)
    if str(dist.get_backend()).lower() != "gloo":
        dist.all_to_all_single(output, input_tensor)
        return output, "native all_to_all_single"

    gathered = torch.empty(
        world_size * world_size,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.all_gather_into_tensor(gathered, input_tensor)
    # gathered[src, dst] 是 src 发给 dst 的元素；本 rank 取对应列。
    output.copy_(gathered.view(world_size, world_size)[:, dist.get_rank()])
    return output, "fallback all_gather + local column select"


def task_semantics(device: torch.device) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_sum = world_size * (world_size - 1) // 2

    original = torch.tensor([rank], dtype=torch.float32, device=device)

    broadcasted = original.clone()
    dist.broadcast(broadcasted, src=0)
    torch.testing.assert_close(broadcasted, torch.zeros_like(broadcasted))

    gathered_list = [torch.empty_like(original) for _ in range(world_size)]
    dist.all_gather(gathered_list, original)
    gathered = torch.cat(gathered_list)
    torch.testing.assert_close(
        gathered, torch.arange(world_size, dtype=torch.float32, device=device)
    )

    rs_input = (
        torch.arange(world_size, dtype=torch.float32, device=device)
        + rank * world_size
    )
    rs_output, rs_mode = reduce_scatter_sum(rs_input)
    rs_expected = world_size * rank + world_size * world_size * (world_size - 1) // 2
    torch.testing.assert_close(
        rs_output,
        torch.tensor([rs_expected], dtype=torch.float32, device=device),
    )

    all_reduced = original.clone()
    dist.all_reduce(all_reduced, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(
        all_reduced, torch.tensor([expected_sum], dtype=torch.float32, device=device)
    )

    a2a_input = (
        torch.arange(world_size, dtype=torch.float32, device=device)
        + rank * world_size
    )
    a2a_output, a2a_mode = all_to_all_portable(a2a_input)
    a2a_expected = (
        torch.arange(world_size, dtype=torch.float32, device=device) * world_size + rank
    )
    torch.testing.assert_close(a2a_output, a2a_expected)

    ordered_print(
        "rank=%d input=[%d] broadcast=%s all_gather=%s reduce_scatter=%s "
        "all_reduce=%s all_to_all=%s\n"
        "  modes: %s; %s"
        % (
            rank,
            rank,
            broadcasted.cpu().tolist(),
            gathered.cpu().tolist(),
            rs_output.cpu().tolist(),
            all_reduced.cpu().tolist(),
            a2a_output.cpu().tolist(),
            rs_mode,
            a2a_mode,
        )
    )
    if rank == 0:
        print("[PASS] Task A: 五种 collective 的输入输出语义全部通过断言。", flush=True)


def task_duality(device: torch.device, chunk_elements: int = 4) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    elements = world_size * chunk_elements
    contribution = (
        torch.arange(elements, dtype=torch.float32, device=device) + rank * 100
    )

    reference = contribution.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)

    shard, rs_mode = reduce_scatter_sum(contribution)
    reconstructed = torch.empty_like(reference)
    dist.all_gather_into_tensor(reconstructed, shard)
    torch.testing.assert_close(reconstructed, reference, rtol=0.0, atol=0.0)

    if rank == 0:
        print(
            "[PASS] Task B: reduce-scatter + all-gather 与 all-reduce 完全一致 "
            "(elements=%d, RS=%s)。" % (elements, rs_mode),
            flush=True,
        )
        print("  first_values=%s" % reconstructed[:8].cpu().tolist(), flush=True)


def message_sizes(min_bytes: int, max_bytes: int) -> List[int]:
    sizes = []
    value = min_bytes
    while value <= max_bytes:
        sizes.append(value)
        value *= 2
    if sizes[-1] != max_bytes:
        sizes.append(max_bytes)
    return sizes


def repeats_for_size(nbytes: int, args: argparse.Namespace) -> int:
    target = max(1, args.target_bytes // nbytes)
    return max(args.min_repeats, min(args.max_repeats, target))


def measure_all_reduce(
    nbytes: int,
    repeats: int,
    warmup: int,
    device: torch.device,
) -> Measurement:
    tensor = torch.zeros(nbytes // 4, dtype=torch.float32, device=device)
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    dist.barrier()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    synchronize(device)
    local_seconds = (time.perf_counter() - started) / repeats

    # 同步作业由最慢 rank 决定，报告 ranks 中的最大完成时间。
    maximum = torch.tensor([local_seconds], dtype=torch.float64, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return Measurement(nbytes, float(maximum.item()), repeats)


def linear_fit(points: Sequence[Measurement]) -> Tuple[float, float]:
    """返回 T=a+beta*n 的 a（秒）和 beta（秒/Byte）。"""
    if len(points) < 2:
        raise ValueError("线性拟合至少需要两个点")
    xs = [float(point.nbytes) for point in points]
    ys = [point.seconds for point in points]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    beta = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - beta * mean_x
    return intercept, beta


def write_csv(path: Path, measurements: Sequence[Measurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("bytes", "time_seconds", "time_us", "algbw_GB_s", "repeats"))
        for point in measurements:
            writer.writerow(
                (
                    point.nbytes,
                    "%.9g" % point.seconds,
                    "%.6f" % point.microseconds,
                    "%.6f" % point.algbw_gb_s,
                    point.repeats,
                )
            )


def write_loglog_svg(path: Path, measurements: Sequence[Measurement], title: str) -> None:
    """用标准库生成可直接打开的 log-log SVG，避免额外绘图库依赖。"""
    width, height = 900, 520
    left, right, top, bottom = 95, 30, 45, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    xs = [math.log10(point.nbytes) for point in measurements]
    ys = [math.log10(point.microseconds) for point in measurements]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    polyline = " ".join(
        "%.2f,%.2f" % (x_coord(x), y_coord(y)) for x, y in zip(xs, ys)
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="%d" y="25" text-anchor="middle" font-family="sans-serif" '
        'font-size="18">%s</text>' % (width // 2, html.escape(title)),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>'
        % (left, top + plot_height, left + plot_width, top + plot_height),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333"/>'
        % (left, top, left, top + plot_height),
        '<polyline fill="none" stroke="#2563eb" stroke-width="2" points="%s"/>'
        % polyline,
    ]

    label_indexes = sorted(set(list(range(0, len(measurements), 3)) + [len(measurements) - 1]))
    for index, (point, x, y) in enumerate(zip(measurements, xs, ys)):
        px, py = x_coord(x), y_coord(y)
        parts.append('<circle cx="%.2f" cy="%.2f" r="3.5" fill="#dc2626"/>' % (px, py))
        if index in label_indexes:
            parts.append(
                '<text x="%.2f" y="%d" text-anchor="middle" font-family="monospace" '
                'font-size="11">%g MB</text>'
                % (px, top + plot_height + 24, point.nbytes / 1e6)
            )

    for tick in range(5):
        value = y_min + tick * (y_max - y_min) / 4
        py = y_coord(value)
        parts.append(
            '<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#ddd"/>'
            % (left, py, left + plot_width, py)
        )
        parts.append(
            '<text x="%d" y="%.2f" text-anchor="end" font-family="monospace" '
            'font-size="11">%.1f us</text>' % (left - 8, py + 4, 10**value)
        )

    parts.extend(
        [
            '<text x="%d" y="%d" text-anchor="middle" font-family="sans-serif" '
            'font-size="13">message size (MB, SI; log scale)</text>'
            % (left + plot_width // 2, height - 22),
            '<text transform="translate(22 %d) rotate(-90)" text-anchor="middle" '
            'font-family="sans-serif" font-size="13">all-reduce time (us; log scale)</text>'
            % (top + plot_height // 2),
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def measure_payload_copy(nbytes: int, device: torch.device) -> float:
    source = torch.ones(nbytes // 4, dtype=torch.float32, device=device)
    target = torch.empty_like(source)
    repeats = 5
    for _ in range(2):
        target.copy_(source)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        target.copy_(source)
    synchronize(device)
    seconds = (time.perf_counter() - started) / repeats
    return nbytes / seconds / 1e9


def task_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    rank = dist.get_rank()
    sizes = message_sizes(args.min_bytes, args.max_bytes)
    measurements = []
    for nbytes in sizes:
        point = measure_all_reduce(
            nbytes,
            repeats_for_size(nbytes, args),
            args.warmup,
            device,
        )
        measurements.append(point)
        if rank == 0:
            print(
                "bytes=%9d time=%10.3f us algbw=%8.3f GB/s repeats=%d"
                % (point.nbytes, point.microseconds, point.algbw_gb_s, point.repeats),
                flush=True,
            )

    if rank == 0:
        small = [point for point in measurements if point.nbytes <= args.small_max_bytes]
        large = [point for point in measurements if point.nbytes >= args.large_min_bytes]
        if not small or len(large) < 2:
            raise RuntimeError("消息扫描范围不足以拟合小消息平台和大消息斜率")
        alpha_op = statistics.median(point.seconds for point in small)
        large_intercept, beta_op = linear_fit(large)
        bandwidth = math.inf if beta_op <= 0 else 1.0 / beta_op / 1e9
        crossover = math.inf if beta_op <= 0 else alpha_op / beta_op

        csv_path = args.output_dir / "all_reduce.csv"
        svg_path = args.output_dir / "all_reduce_loglog.svg"
        write_csv(csv_path, measurements)
        write_loglog_svg(
            svg_path,
            measurements,
            "all-reduce: effective alpha plateau and beta slope",
        )
        print(
            "fit: alpha_op=%.3f us beta_op=%.6f ns/B effective_bw=%.3f GB/s "
            "crossover=%.3f MB large_intercept=%.3f us"
            % (
                alpha_op * 1e6,
                beta_op * 1e9,
                bandwidth,
                crossover / 1e6,
                large_intercept * 1e6,
            ),
            flush=True,
        )
        print("csv=%s\nplot=%s" % (csv_path, svg_path), flush=True)

    # 只让 rank 0 做本地 copy，避免 8 个进程同时争抢内存带宽。
    dist.barrier()
    if rank == 0:
        copy_gb_s = measure_payload_copy(args.max_bytes, device)
        print(
            "local_payload_copy=%.3f GB/s (只按一份 payload 计数；读+写物理流量更大)"
            % copy_gb_s,
            flush=True,
        )
        print(
            "[PASS] Task C: 已生成时间-消息大小曲线，并拟合 whole-collective 的有效 alpha/beta。",
            flush=True,
        )
    dist.barrier()


def timed_all_reduce_round(
    tensor: Tensor,
    device: torch.device,
    slow_rank: int,
    sleep_seconds: float,
) -> float:
    dist.barrier()
    synchronize(device)
    started = time.perf_counter()
    if dist.get_rank() == slow_rank and sleep_seconds:
        time.sleep(sleep_seconds)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    synchronize(device)
    local = time.perf_counter() - started
    maximum = torch.tensor([local], dtype=torch.float64, device=device)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def task_straggler(args: argparse.Namespace, device: torch.device) -> None:
    world_size = dist.get_world_size()
    if not 0 <= args.straggler_rank < world_size:
        raise ValueError("straggler-rank 必须落在 [0, world_size) 内")
    tensor = torch.zeros(250_000, dtype=torch.float32, device=device)  # 1 MB（SI）
    baseline = []
    delayed = []
    for _ in range(args.straggler_iters):
        baseline.append(timed_all_reduce_round(tensor, device, args.straggler_rank, 0.0))
        delayed.append(
            timed_all_reduce_round(
                tensor,
                device,
                args.straggler_rank,
                args.straggler_ms / 1e3,
            )
        )
    if dist.get_rank() == 0:
        base_ms = statistics.mean(baseline) * 1e3
        delayed_ms = statistics.mean(delayed) * 1e3
        print(
            "[PASS] Challenge: baseline=%.3f ms straggler=%.3f ms delta=%.3f ms "
            "(rank=%d sleep=%.3f ms)"
            % (
                base_ms,
                delayed_ms,
                delayed_ms - base_ms,
                args.straggler_rank,
                args.straggler_ms,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    backend, device = resolve_backend(args.backend, args.local_rank)
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            print(
                "torch=%s backend=%s device=%s world_size=%d"
                % (torch.__version__, backend, device, world_size),
                flush=True,
            )
            if backend == "gloo":
                print("note=Gloo 使用 CPU tensor；Apple Silicon 也走 CPU，不走 MPS。", flush=True)

        if args.task in ("all", "semantics"):
            task_semantics(device)
        if args.task in ("all", "duality"):
            task_duality(device)
        if args.task in ("all", "benchmark"):
            task_benchmark(args, device)
        if args.task in ("all", "straggler"):
            task_straggler(args, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
