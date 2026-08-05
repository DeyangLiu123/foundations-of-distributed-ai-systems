#!/usr/bin/env python3
"""L62: benchmark an OpenAI-compatible streaming inference endpoint.

Only the Python 3.10+ standard library is required.  The script records raw
per-request rows so that medians/percentiles can be recomputed later instead of
being hidden behind a console summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CSV_FIELDS = [
    "task",
    "label",
    "group_id",
    "run_id",
    "target_prompt_words",
    "target_concurrency",
    "prompt_tokens",
    "output_tokens",
    "token_count_source",
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "request_decode_tps",
    "group_output_tps",
    "group_decode_tps",
    "group_makespan_ms",
    "success",
    "error",
]


@dataclass
class Measurement:
    task: str
    label: str
    group_id: str
    run_id: int
    target_prompt_words: int
    target_concurrency: int
    prompt_tokens: int | None
    output_tokens: int | None
    token_count_source: str
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float | None
    request_decode_tps: float | None
    success: bool
    error: str
    start_s: float
    first_token_s: float | None
    end_s: float
    group_output_tps: float | None = None
    group_decode_tps: float | None = None
    group_makespan_ms: float | None = None

    def csv_row(self) -> dict[str, Any]:
        row = {
            name: getattr(self, name)
            for name in CSV_FIELDS
            if hasattr(self, name)
        }
        # CSV should be easy to inspect and should not contain Python booleans.
        row["success"] = "true" if self.success else "false"
        return {name: row.get(name, "") for name in CSV_FIELDS}


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def make_prompt(target_words: int, marker: str, shared_prefix: str = "") -> str:
    """Build a deterministic synthetic prompt; actual token count is measured."""
    vocabulary = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "network system queue cache tensor token request latency throughput "
    ).split()
    offset = sum(ord(ch) for ch in marker) % len(vocabulary)
    words = [vocabulary[(offset + i) % len(vocabulary)] for i in range(target_words)]
    body = " ".join(words)
    return (
        f"Benchmark marker {marker}. {shared_prefix}"
        "Read the following context, then continue with short numbered items.\n"
        f"Context: {body}\nAnswer: 1."
    )


def post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tokenize_count(
    base_url: str,
    model: str,
    text: str,
    api_key: str,
    timeout: float,
    add_special: bool,
) -> int | None:
    """Try vLLM then llama.cpp /tokenize request shapes."""
    url = f"{base_url.rstrip('/')}/tokenize"
    payloads = [
        {
            "model": model,
            "prompt": text,
            "add_special_tokens": add_special,
        },
        {
            "content": text,
            "add_special": add_special,
        },
    ]
    for payload in payloads:
        try:
            result = post_json(url, payload, api_key, timeout)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        if isinstance(result.get("count"), int):
            return int(result["count"])
        if isinstance(result.get("tokens"), list):
            return len(result["tokens"])
        if isinstance(result.get("input_tokens"), int):
            return int(result["input_tokens"])
    return None


def chunk_text(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    if isinstance(choice.get("text"), str):
        return choice["text"]
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return ""


def error_text(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive only
            body = ""
        return f"HTTP {exc.code}: {body[:500]}"
    return f"{type(exc).__name__}: {exc}"


def measure_one(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    prompt: str,
    output_tokens: int,
    ignore_eos: bool,
    task: str,
    label: str,
    group_id: str,
    run_id: int,
    target_prompt_words: int,
    target_concurrency: int,
    start_barrier: threading.Barrier | None = None,
) -> Measurement:
    if start_barrier is not None:
        start_barrier.wait()

    start_s = time.perf_counter()
    first_token_s: float | None = None
    end_s = start_s
    pieces: list[str] = []
    event_count = 0
    usage_prompt: int | None = None
    usage_output: int | None = None

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if ignore_eos:
        # Both vLLM and llama.cpp accept this extension to the OpenAI schema.
        payload["ignore_eos"] = True

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    if isinstance(usage.get("prompt_tokens"), int):
                        usage_prompt = int(usage["prompt_tokens"])
                    if isinstance(usage.get("completion_tokens"), int):
                        usage_output = int(usage["completion_tokens"])
                text = chunk_text(event)
                if text:
                    now = time.perf_counter()
                    if first_token_s is None:
                        first_token_s = now
                    pieces.append(text)
                    event_count += 1
        end_s = time.perf_counter()

        output_text = "".join(pieces)
        prompt_count = tokenize_count(
            base_url, model, prompt, api_key, timeout, add_special=True
        )
        retokenized_output_count = tokenize_count(
            base_url, model, output_text, api_key, timeout, add_special=False
        )

        if prompt_count is None:
            prompt_count = usage_prompt
        if usage_output is not None:
            actual_output = usage_output
            count_source = "stream usage"
        elif retokenized_output_count is not None:
            actual_output = retokenized_output_count
            count_source = "/tokenize (retokenized)"
        else:
            actual_output = event_count
            count_source = "SSE event count (approximate)"

        if first_token_s is None or actual_output <= 0:
            raise RuntimeError("stream completed without a non-empty output token")

        ttft_s = first_token_s - start_s
        e2e_s = end_s - start_s
        decode_s = end_s - first_token_s
        tpot_s = decode_s / (actual_output - 1) if actual_output > 1 else math.nan
        request_decode_tps = (
            (actual_output - 1) / decode_s
            if actual_output > 1 and decode_s > 0
            else math.nan
        )
        return Measurement(
            task=task,
            label=label,
            group_id=group_id,
            run_id=run_id,
            target_prompt_words=target_prompt_words,
            target_concurrency=target_concurrency,
            prompt_tokens=prompt_count,
            output_tokens=actual_output,
            token_count_source=count_source,
            ttft_ms=ttft_s * 1000.0,
            tpot_ms=tpot_s * 1000.0,
            e2e_ms=e2e_s * 1000.0,
            request_decode_tps=request_decode_tps,
            success=True,
            error="",
            start_s=start_s,
            first_token_s=first_token_s,
            end_s=end_s,
        )
    except Exception as exc:
        end_s = time.perf_counter()
        return Measurement(
            task=task,
            label=label,
            group_id=group_id,
            run_id=run_id,
            target_prompt_words=target_prompt_words,
            target_concurrency=target_concurrency,
            prompt_tokens=None,
            output_tokens=None,
            token_count_source="",
            ttft_ms=None,
            tpot_ms=None,
            e2e_ms=None,
            request_decode_tps=None,
            success=False,
            error=error_text(exc),
            start_s=start_s,
            first_token_s=None,
            end_s=end_s,
        )


def common_measure_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "base_url": args.base_url,
        "model": args.model,
        "api_key": args.api_key,
        "timeout": args.timeout,
        "output_tokens": args.output_tokens,
        "ignore_eos": args.ignore_eos,
        "label": args.label,
    }


def warm_up(args: argparse.Namespace) -> None:
    for index in range(args.warmup):
        result = measure_one(
            **common_measure_kwargs(args),
            prompt=make_prompt(32, f"warmup-{index}-{time.time_ns()}"),
            task="warmup",
            group_id=f"warmup-{index}",
            run_id=index,
            target_prompt_words=32,
            target_concurrency=1,
        )
        if not result.success:
            raise RuntimeError(f"warmup failed: {result.error}")


def run_length(args: argparse.Namespace) -> list[Measurement]:
    warm_up(args)
    rows: list[Measurement] = []
    for target_words in args.lengths:
        for repeat in range(args.repeats):
            marker = f"length-{target_words}-{repeat}-{time.time_ns()}"
            rows.append(
                measure_one(
                    **common_measure_kwargs(args),
                    prompt=make_prompt(target_words, marker),
                    task="length",
                    group_id=f"length-{target_words}-{repeat}",
                    run_id=repeat,
                    target_prompt_words=target_words,
                    target_concurrency=1,
                )
            )
    return rows


def attach_group_metrics(rows: list[Measurement]) -> None:
    successful = [row for row in rows if row.success]
    if not successful:
        return
    group_start = min(row.start_s for row in successful)
    group_end = max(row.end_s for row in successful)
    makespan = group_end - group_start
    total_output = sum(row.output_tokens or 0 for row in successful)
    first_tokens = [row.first_token_s for row in successful if row.first_token_s]
    decode_tokens = sum(max((row.output_tokens or 0) - 1, 0) for row in successful)
    decode_span = group_end - min(first_tokens) if first_tokens else math.nan
    output_tps = total_output / makespan if makespan > 0 else math.nan
    decode_tps = (
        decode_tokens / decode_span
        if first_tokens and decode_span > 0
        else math.nan
    )
    for row in rows:
        row.group_output_tps = output_tps
        row.group_decode_tps = decode_tps
        row.group_makespan_ms = makespan * 1000.0


def run_concurrency(args: argparse.Namespace) -> list[Measurement]:
    warm_up(args)
    all_rows: list[Measurement] = []
    for concurrency in args.concurrency:
        for repeat in range(args.repeats):
            group_id = f"concurrency-{concurrency}-{repeat}"
            barrier = threading.Barrier(concurrency + 1)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = []
                for request_index in range(concurrency):
                    # The unique marker appears before the repeated body.  It prevents
                    # prefix-cache hits from contaminating the batching sweep.
                    marker = (
                        f"batch-{concurrency}-{repeat}-{request_index}-{time.time_ns()}"
                    )
                    futures.append(
                        pool.submit(
                            measure_one,
                            **common_measure_kwargs(args),
                            prompt=make_prompt(args.prompt_words, marker),
                            task="concurrency",
                            group_id=group_id,
                            run_id=repeat * concurrency + request_index,
                            target_prompt_words=args.prompt_words,
                            target_concurrency=concurrency,
                            start_barrier=barrier,
                        )
                    )
                barrier.wait()
                group_rows = [future.result() for future in futures]
            attach_group_metrics(group_rows)
            all_rows.extend(group_rows)
    return all_rows


def run_prefix(args: argparse.Namespace) -> list[Measurement]:
    warm_up(args)
    # A long, block-aligned shared history makes the cache signal visible.  The
    # newest "turn" changes, so this tests prefix reuse rather than response reuse.
    shared = make_prompt(
        args.prefix_words,
        f"shared-history-{args.prefix_id}",
        shared_prefix="This is a multi-turn conversation transcript. ",
    )
    rows: list[Measurement] = []
    for repeat in range(args.repeats):
        prompt = (
            f"{shared}\nUser turn {repeat}: give one more short numbered item.\nAssistant:"
        )
        rows.append(
            measure_one(
                **common_measure_kwargs(args),
                prompt=prompt,
                task="prefix",
                group_id=f"prefix-{args.prefix_id}",
                run_id=repeat,
                target_prompt_words=args.prefix_words,
                target_concurrency=1,
            )
        )
    return rows


def write_csv(path: Path, rows: list[Measurement], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not append or not exists:
            writer.writeheader()
        writer.writerows(row.csv_row() for row in rows)


def print_summary(rows: list[Measurement]) -> None:
    successful = [row for row in rows if row.success]
    failed = [row for row in rows if not row.success]
    print(f"completed={len(successful)} failed={len(failed)}")
    if successful:
        ttfts = [row.ttft_ms for row in successful if row.ttft_ms is not None]
        tpots = [
            row.tpot_ms
            for row in successful
            if row.tpot_ms is not None and math.isfinite(row.tpot_ms)
        ]
        print(
            "TTFT ms: "
            f"mean={statistics.fmean(ttfts):.2f} "
            f"p50={percentile(ttfts, 50):.2f} p99={percentile(ttfts, 99):.2f}"
        )
        if tpots:
            print(
                "TPOT ms: "
                f"mean={statistics.fmean(tpots):.2f} "
                f"p50={percentile(tpots, 50):.2f} p99={percentile(tpots, 99):.2f}"
            )
        group_tps = [
            row.group_decode_tps
            for row in successful
            if row.group_decode_tps is not None
            and math.isfinite(row.group_decode_tps)
        ]
        if group_tps:
            print(f"group decode throughput: max={max(group_tps):.2f} tok/s")
    for row in failed[:5]:
        print(f"ERROR {row.group_id}/{row.run_id}: {row.error}", file=sys.stderr)


class _SelfTestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/tokenize":
            text = payload.get("prompt", payload.get("content", ""))
            result = json.dumps(
                {"count": len(str(text).split()), "tokens": list(range(len(str(text).split())))}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)
            return
        if self.path == "/v1/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for piece in ("alpha", " beta", " gamma"):
                time.sleep(0.005)
                event = json.dumps({"choices": [{"text": piece}]})
                self.wfile.write(f"data: {event}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        self.send_error(404)


def self_test() -> None:
    assert parse_int_list("1,2,4") == [1, 2, 4]
    assert chunk_text({"choices": [{"text": "x"}]}) == "x"
    assert chunk_text({"choices": [{"delta": {"content": "y"}}]}) == "y"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SelfTestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = measure_one(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="self-test",
            api_key="",
            timeout=2.0,
            prompt="one two three four",
            output_tokens=3,
            ignore_eos=False,
            task="self-test",
            label="self-test",
            group_id="self-test",
            run_id=0,
            target_prompt_words=4,
            target_concurrency=1,
        )
        assert result.success, result.error
        assert result.prompt_tokens == 4
        assert result.output_tokens == 3
        assert result.ttft_ms is not None and result.ttft_ms > 0
        assert result.tpot_ms is not None and result.tpot_ms > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    print("[PASS] SSE parsing, /tokenize counting, TTFT and TPOT formulas")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("smoke", "length", "concurrency", "prefix", "all", "self-test"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="l62-model")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--label", default="default")
    parser.add_argument("--out", type=Path, default=Path("l62-results.csv"))
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lengths", type=parse_int_list, default=[128, 512, 2048, 4096])
    parser.add_argument("--concurrency", type=parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--prompt-words", type=int, default=512)
    parser.add_argument("--prefix-words", type=int, default=2048)
    parser.add_argument("--prefix-id", default="fixed")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "self-test":
        self_test()
        return 0
    if args.output_tokens <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise SystemExit("output-tokens/repeats must be positive; warmup cannot be negative")

    if args.mode == "smoke":
        rows = [
            measure_one(
                **common_measure_kwargs(args),
                prompt=make_prompt(32, f"smoke-{time.time_ns()}"),
                task="smoke",
                group_id="smoke",
                run_id=0,
                target_prompt_words=32,
                target_concurrency=1,
            )
        ]
    elif args.mode == "length":
        rows = run_length(args)
    elif args.mode == "concurrency":
        rows = run_concurrency(args)
    elif args.mode == "prefix":
        rows = run_prefix(args)
    else:
        rows = run_length(args) + run_concurrency(args) + run_prefix(args)

    write_csv(args.out, rows, args.append)
    print_summary(rows)
    print(f"wrote {args.out}")
    return 0 if all(row.success for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
