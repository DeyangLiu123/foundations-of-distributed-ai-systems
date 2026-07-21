"""L09: 在 CPU、Apple Silicon 或 CUDA 上训练一个 MNIST MLP。

依赖：Python 3.9-3.12，torch==2.5.1，torchvision==0.20.1。
运行示例：
    python "M1 深度学习基础/实践代码/L09_train_mnist.py" --device cpu
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class MnistMlp(nn.Module):
    """784 -> 256 -> 128 -> 10 的分类 MLP。"""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "fashionmnist"), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/l09"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.save_every < 1:
        parser.error("epochs、batch-size 和 save-every 必须为正数")
    if args.workers < 0:
        parser.error("workers 不能为负数")
    return args


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
        raise RuntimeError(f"请求的设备 {requested!r} 当前不可用；可改用 --device cpu")
    return torch.device(requested)


def maybe_limit(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("limit 必须为正数")
    return Subset(dataset, range(min(limit, len(dataset))))


def make_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    dataset_cls = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    transform = transforms.ToTensor()
    train_set = dataset_cls(args.data_dir, train=True, download=True, transform=transform)
    test_set = dataset_cls(args.data_dir, train=False, download=True, transform=transform)
    train_set = maybe_limit(train_set, args.limit_train)
    test_set = maybe_limit(test_set, args.limit_test)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(test_set, shuffle=False, **common),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not math.isfinite(loss.item()):
            raise RuntimeError(
                "loss 已变为 NaN 或 Inf。停止运行；请记录学习率、epoch 和终端输出。"
            )
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    elapsed = time.perf_counter() - start
    return total_loss / total_examples, elapsed


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, correct / total_examples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]) + 1


def process_memory_mib() -> Optional[float]:
    """psutil 是可选依赖；没安装时仍能完成训练。"""
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1024**2


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_loader, test_loader = make_loaders(args, device)
    model = MnistMlp().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    expected_parameters = (784 * 256 + 256) + (256 * 128 + 128) + (128 * 10 + 10)
    actual_parameters = count_parameters(model)
    if actual_parameters != expected_parameters:
        raise AssertionError(f"参数量不符：得到 {actual_parameters}，预期 {expected_parameters}")

    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)
        print(f"已从 {args.resume} 恢复，将从 epoch {start_epoch + 1} 开始。")

    print(f"device={device} dataset={args.dataset} parameters={actual_parameters:,}")
    print(f"train_examples={len(train_loader.dataset):,} batch_size={args.batch_size}")
    memory_before = process_memory_mib()
    if memory_before is not None:
        print(f"process_rss_before={memory_before:.1f} MiB")

    checkpoint_path = args.output_dir / "checkpoint.pt"
    for epoch in range(start_epoch, args.epochs):
        train_loss, seconds = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        samples_per_second = len(train_loader.dataset) / seconds
        print(
            f"epoch={epoch + 1:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"test_accuracy={accuracy:.2%} time={seconds:.1f}s "
            f"samples_per_second={samples_per_second:.0f}"
        )
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(checkpoint_path, model, optimizer, epoch, args)
            print(f"checkpoint={checkpoint_path}")

    memory_after = process_memory_mib()
    if memory_after is not None:
        print(f"process_rss_after={memory_after:.1f} MiB")
    else:
        print("未安装可选依赖 psutil；可用活动监视器或任务管理器观察进程内存。")


if __name__ == "__main__":
    main()
