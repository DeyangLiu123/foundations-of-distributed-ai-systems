"""L19: 在 CPU、Apple Silicon 或 CUDA 上训练并解剖一个字符级迷你 GPT。

依赖：Python 3.9-3.12，torch==2.5.1、numpy==1.26.4。
常用命令：
    python L19_mini_gpt.py train --device cpu
    python L19_mini_gpt.py benchmark --checkpoint outputs/l19/model.pt --device cpu
    python L19_mini_gpt.py generate --checkpoint outputs/l19/model.pt --temperature 0.8 --top-p 0.9
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# 无网络时的公版文本后备。重复只为了凑够可采样长度，不应拿它评估模型质量。
FALLBACK_TEXT = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

KING RICHARD III:
Now is the winter of our discontent
Made glorious summer by this sun of York.

HAMLET:
To be, or not to be: that is the question.

JULIET:
Good night, good night! parting is such sweet sorrow.
"""

PastKeyValue = Tuple[Tensor, Tensor]


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 2048
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

    def validate(self) -> None:
        if min(
            self.vocab_size,
            self.max_seq_len,
            self.n_layer,
            self.n_head,
            self.n_kv_head,
            self.n_embd,
        ) < 1:
            raise ValueError("模型维度必须为正数")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n-embd 必须能被 n-head 整除")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n-head 必须能被 n-kv-head 整除")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须在 [0, 1) 内")


class CausalSelfAttention(nn.Module):
    """支持 MHA/GQA/MQA 和 KV cache 的 causal self-attention。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        kv_width = self.n_kv_head * self.head_dim

        # 分开 Q/K/V，便于把 n_kv_head 改成 1 观察 MQA 参数和缓存变化。
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, kv_width, bias=False)
        self.v_proj = nn.Linear(config.n_embd, kv_width, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(
        self,
        x: Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[PastKeyValue]]:
        batch, steps, width = x.shape
        q = self.q_proj(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, steps, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, steps, self.n_kv_head, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            if steps != 1:
                raise ValueError("本课缓存路径每次 decode 只接收 1 个新 token")
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_key_value = (k, v) if use_cache else None

        # cache 中仍只存 n_kv_head 组；计算 attention 时才在 head 维度广播。
        repeat = self.n_head // self.n_kv_head
        k_for_attention = k.repeat_interleave(repeat, dim=1)
        v_for_attention = v.repeat_interleave(repeat, dim=1)

        # prefill 需要下三角 mask；单 token decode 的所有 cache 位置都在过去，可全部看见。
        is_prefill = past_key_value is None and steps > 1
        y = F.scaled_dot_product_attention(
            q,
            k_for_attention,
            v_for_attention,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_prefill,
        )
        y = y.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.resid_dropout(self.out_proj(y)), new_key_value


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[PastKeyValue]]:
        attention, new_key_value = self.attn(self.ln_1(x), past_key_value, use_cache)
        x = x + attention
        x = x + self.mlp(self.ln_2(x))
        return x, new_key_value


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.apply(self._init_weights)
        # weight tying：输入 embedding 与输出 lm_head 共用同一张参数表。
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: Tensor,
        targets: Optional[Tensor] = None,
        past_key_values: Optional[Sequence[PastKeyValue]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[PastKeyValue]]]:
        batch, steps = token_ids.shape
        if past_key_values is not None and len(past_key_values) != self.config.n_layer:
            raise ValueError("past_key_values 的层数与模型不一致")
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if past_len + steps > self.config.max_seq_len:
            raise ValueError(
                f"序列长度 {past_len + steps} 超过 max_seq_len={self.config.max_seq_len}"
            )

        positions = torch.arange(past_len, past_len + steps, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)

        new_key_values: Optional[List[PastKeyValue]] = [] if use_cache else None
        for layer_index, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            x, layer_cache = block(x, layer_past, use_cache)
            if new_key_values is not None:
                assert layer_cache is not None
                new_key_values.append(layer_cache)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_key_values


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    available = {
        "cpu": True,
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        for name in ("cuda", "mps", "cpu"):
            if available[name]:
                return torch.device(name)
    if not available[requested]:
        raise RuntimeError(f"请求的设备 {requested!r} 不可用；可改用 --device cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def load_corpus(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"downloading={TINY_SHAKESPEARE_URL}")
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as response:
            text = response.read().decode("utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"dataset={path} characters={len(text):,}")
        return text
    except Exception as exc:
        text = (FALLBACK_TEXT + "\n") * 512
        path.write_text(text, encoding="utf-8")
        print(f"download_failed={type(exc).__name__} using_public_domain_fallback={path}")
        return text


def make_vocabulary(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> Tensor:
    unknown_id = stoi.get("\n", 0)
    return torch.tensor([stoi.get(char, unknown_id) for char in text], dtype=torch.long)


def decode(token_ids: Tensor, itos: Dict[int, str]) -> str:
    return "".join(itos[int(token)] for token in token_ids.detach().cpu().tolist())


def get_batch(data: Tensor, batch_size: int, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    if len(data) <= seq_len:
        raise ValueError(f"数据长度 {len(data)} 必须大于 train-seq-len={seq_len}")
    starts = torch.randint(0, len(data) - seq_len, (batch_size,))
    x = torch.stack([data[start : start + seq_len] for start in starts])
    y = torch.stack([data[start + 1 : start + seq_len + 1] for start in starts])
    return x.to(device), y.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def theoretical_parameter_count(config: GPTConfig) -> int:
    d = config.n_embd
    head_dim = d // config.n_head
    embeddings = config.vocab_size * d + config.max_seq_len * d
    attention = 2 * d * d + 2 * d * config.n_kv_head * head_dim
    mlp = 8 * d * d
    two_layer_norms = 4 * d
    final_layer_norm = 2 * d
    return embeddings + config.n_layer * (attention + mlp + two_layer_norms) + final_layer_norm


def parameter_breakdown(model: MiniGPT) -> Dict[str, int]:
    return {
        "token_embedding_tied_lm_head": model.token_embedding.weight.numel(),
        "position_embedding": model.position_embedding.weight.numel(),
        "attention_all_layers": sum(
            parameter.numel() for block in model.blocks for parameter in block.attn.parameters()
        ),
        "mlp_all_layers": sum(
            parameter.numel() for block in model.blocks for parameter in block.mlp.parameters()
        ),
        "layer_norms": sum(
            parameter.numel()
            for block in model.blocks
            for norm in (block.ln_1, block.ln_2)
            for parameter in norm.parameters()
        )
        + sum(parameter.numel() for parameter in model.ln_f.parameters()),
    }


def sample_next_token(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    if temperature < 0.0:
        raise ValueError("temperature 不能为负数")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top-p 必须在 (0, 1] 内")
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.inference_mode()
def generate_naive(
    model: MiniGPT,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tensor:
    model.eval()
    output = prompt
    for _ in range(max_new_tokens):
        logits, _, _ = model(output)
        next_token = sample_next_token(logits[:, -1, :], temperature, top_p)
        output = torch.cat((output, next_token), dim=1)
    return output


@torch.inference_mode()
def generate_cached(
    model: MiniGPT,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[Tensor, List[PastKeyValue]]:
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens 必须为正数")
    model.eval()
    logits, _, cache = model(prompt, use_cache=True)
    assert cache is not None
    next_token = sample_next_token(logits[:, -1, :], temperature, top_p)
    output = torch.cat((prompt, next_token), dim=1)

    for _ in range(max_new_tokens - 1):
        logits, _, cache = model(next_token, past_key_values=cache, use_cache=True)
        assert cache is not None
        next_token = sample_next_token(logits[:, -1, :], temperature, top_p)
        output = torch.cat((output, next_token), dim=1)
    return output, cache


@torch.inference_mode()
def estimate_loss(
    model: MiniGPT,
    train_data: Tensor,
    val_data: Tensor,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    result: Dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, seq_len, device)
            _, loss, _ = model(x, targets=y)
            assert loss is not None
            losses.append(loss.item())
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    chars: List[str],
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "chars": chars,
            "step": step,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> Tuple[MiniGPT, List[str], int]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, list(checkpoint["chars"]), int(checkpoint.get("step", 0))


def run_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    text = load_corpus(args.data)
    chars, stoi, itos = make_vocabulary(text)
    all_tokens = encode(text, stoi)
    split = int(0.9 * len(all_tokens))
    train_data, val_data = all_tokens[:split], all_tokens[split:]

    config = GPTConfig(
        vocab_size=len(chars),
        max_seq_len=args.max_seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    actual = count_parameters(model)
    expected = theoretical_parameter_count(config)
    if actual != expected:
        raise AssertionError(f"参数量不符：实测 {actual:,}，公式 {expected:,}")
    print(f"device={device} vocabulary={len(chars)} parameters={actual:,}")
    for name, value in parameter_breakdown(model).items():
        print(f"parameter_group={name} count={value:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sample_prompt = encode(args.prompt, stoi)[None, :].to(device)
    if sample_prompt.size(1) + args.sample_tokens > config.max_seq_len:
        raise ValueError("prompt + sample-tokens 超过 max-seq-len")

    started = time.perf_counter()
    metric_rows = []
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.train_seq_len,
                args.eval_iters,
                device,
            )
            elapsed = time.perf_counter() - started
            print(
                f"step={step:04d}/{args.steps} train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} elapsed={elapsed:.1f}s"
            )
            metric_rows.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "elapsed_seconds": elapsed,
                }
            )
            sample, _ = generate_cached(
                model,
                sample_prompt,
                args.sample_tokens,
                args.temperature,
                args.top_p,
            )
            print("--- sample ---")
            print(decode(sample[0], itos))
            print("--- /sample ---")
        if step == args.steps:
            break

        model.train()
        x, y = get_batch(train_data, args.batch_size, args.train_seq_len, device)
        _, loss, _ = model(x, targets=y)
        assert loss is not None
        if not math.isfinite(loss.item()):
            raise RuntimeError("loss 变为 NaN/Inf，请减小 learning rate")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    save_checkpoint(args.output, model, optimizer, chars, args.steps)
    args.loss_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.loss_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"checkpoint={args.output}")
    print(f"loss_csv={args.loss_csv}")


def kv_cache_bytes(cache: Sequence[PastKeyValue]) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)


@torch.inference_mode()
def measure_prefill_and_decode(
    model: MiniGPT,
    length: int,
    device: torch.device,
) -> Tuple[float, float]:
    if length + 1 > model.config.max_seq_len:
        raise ValueError("prefill-length + 1 超过模型 max_seq_len")
    token_ids = (torch.arange(length, device=device) % model.config.vocab_size)[None, :]
    model(token_ids[:, : min(8, length)], use_cache=True)  # 预热 kernel / allocator
    synchronize(device)
    start = time.perf_counter()
    _, _, cache = model(token_ids, use_cache=True)
    synchronize(device)
    prefill_seconds = time.perf_counter() - start
    assert cache is not None

    next_token = token_ids[:, -1:]
    synchronize(device)
    start = time.perf_counter()
    model(next_token, past_key_values=cache, use_cache=True)
    synchronize(device)
    decode_seconds = time.perf_counter() - start
    return prefill_seconds, decode_seconds


def run_benchmark(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, chars, step = load_checkpoint(args.checkpoint, device)
    stoi = {char: index for index, char in enumerate(chars)}
    prompt = encode(args.prompt, stoi)[None, :].to(device)
    lengths = sorted(set(args.lengths))
    if not lengths or min(lengths) < 1:
        raise ValueError("lengths 必须是正数")
    if prompt.size(1) + max(lengths) > model.config.max_seq_len:
        raise ValueError("prompt 长度 + 最大生成长度超过 max_seq_len")

    prefill_seconds, one_decode_seconds = measure_prefill_and_decode(
        model, args.prefill_length, device
    )
    print(
        f"device={device} checkpoint_step={step} parameters={count_parameters(model):,} "
        f"prefill_tokens={args.prefill_length} prefill_ms={1000 * prefill_seconds:.3f} "
        f"one_cached_decode_ms={1000 * one_decode_seconds:.3f}"
    )

    # 让两条生成路径都经过一次预热，减少首次 kernel / allocator 干扰。
    generate_naive(model, prompt, 2, 0.0, 1.0)
    generate_cached(model, prompt, 2, 0.0, 1.0)
    rows = []
    for new_tokens in lengths:
        seed_everything(args.seed)
        synchronize(device)
        start = time.perf_counter()
        naive_output = generate_naive(
            model, prompt, new_tokens, args.temperature, args.top_p
        )
        synchronize(device)
        naive_seconds = time.perf_counter() - start

        seed_everything(args.seed)
        synchronize(device)
        start = time.perf_counter()
        cached_output, cache = generate_cached(
            model, prompt, new_tokens, args.temperature, args.top_p
        )
        synchronize(device)
        cached_seconds = time.perf_counter() - start

        if args.temperature == 0.0 and not torch.equal(naive_output, cached_output):
            raise AssertionError("贪心生成下 naive 与 cached 输出不一致")

        prompt_len = prompt.size(1)
        naive_positions = new_tokens * prompt_len + new_tokens * (new_tokens - 1) // 2
        cached_positions = prompt_len + new_tokens - 1
        actual_cache_bytes = kv_cache_bytes(cache)
        cache_len = prompt_len + new_tokens - 1
        expected_cache_bytes = (
            2
            * model.config.n_layer
            * cache_len
            * model.config.n_kv_head
            * (model.config.n_embd // model.config.n_head)
            * cache[0][0].element_size()
        )
        if actual_cache_bytes != expected_cache_bytes:
            raise AssertionError(
                f"KV cache 不符：实测 {actual_cache_bytes}，公式 {expected_cache_bytes}"
            )

        speedup = naive_seconds / cached_seconds
        row = {
            "new_tokens": new_tokens,
            "naive_seconds": naive_seconds,
            "cached_seconds": cached_seconds,
            "speedup": speedup,
            "naive_input_positions": naive_positions,
            "cached_input_positions": cached_positions,
            "kv_cache_bytes": actual_cache_bytes,
        }
        rows.append(row)
        print(
            f"new_tokens={new_tokens:4d} naive={naive_seconds:.4f}s "
            f"cached={cached_seconds:.4f}s speedup={speedup:.2f}x "
            f"input_positions={naive_positions:,}/{cached_positions:,} "
            f"kv_cache={actual_cache_bytes / 1e6:.3f}MB"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"benchmark_csv={args.csv}")


def run_generate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, chars, _ = load_checkpoint(args.checkpoint, device)
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}
    prompt = encode(args.prompt, stoi)[None, :].to(device)
    output, _ = generate_cached(
        model, prompt, args.max_new_tokens, args.temperature, args.top_p
    )
    print(decode(output[0], itos))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练字符级迷你 GPT")
    train.add_argument("--data", type=Path, default=Path("data/l19/tinyshakespeare.txt"))
    train.add_argument("--output", type=Path, default=Path("outputs/l19/model.pt"))
    train.add_argument("--loss-csv", type=Path, default=Path("outputs/l19/loss.csv"))
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--train-seq-len", type=int, default=128)
    train.add_argument("--eval-interval", type=int, default=100)
    train.add_argument("--eval-iters", type=int, default=20)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--max-seq-len", type=int, default=2048)
    train.add_argument("--n-layer", type=int, default=4)
    train.add_argument("--n-head", type=int, default=4)
    train.add_argument("--n-kv-head", type=int, default=4)
    train.add_argument("--n-embd", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--prompt", default="First Citizen:\n")
    train.add_argument("--sample-tokens", type=int, default=120)
    train.add_argument("--temperature", type=float, default=0.8)
    train.add_argument("--top-p", type=float, default=0.9)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=run_train)

    benchmark = subparsers.add_parser("benchmark", help="对比 naive 与 KV cache 生成")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    benchmark.add_argument("--prompt", default="First Citizen:\n")
    benchmark.add_argument("--lengths", type=int, nargs="+", default=[100, 500, 1000])
    benchmark.add_argument("--prefill-length", type=int, default=512)
    benchmark.add_argument("--temperature", type=float, default=0.0)
    benchmark.add_argument("--top-p", type=float, default=1.0)
    benchmark.add_argument("--csv", type=Path, default=Path("outputs/l19/benchmark.csv"))
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.set_defaults(func=run_benchmark)

    generate = subparsers.add_parser("generate", help="用已训练 checkpoint 生成文本")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    generate.add_argument("--prompt", default="First Citizen:\n")
    generate.add_argument("--max-new-tokens", type=int, default=300)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=run_generate)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_names = (
        "steps",
        "batch_size",
        "train_seq_len",
        "eval_interval",
        "eval_iters",
        "max_seq_len",
        "n_layer",
        "n_head",
        "n_kv_head",
        "n_embd",
        "sample_tokens",
        "prefill_length",
        "max_new_tokens",
    )
    for name in positive_names:
        minimum = 0 if name == "steps" else 1
        if hasattr(args, name) and getattr(args, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} 必须为正数")
    if hasattr(args, "train_seq_len") and args.train_seq_len > args.max_seq_len:
        parser.error("--train-seq-len 不能超过 --max-seq-len")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    args.func(args)


if __name__ == "__main__":
    main()
