#!/usr/bin/env python3
"""Validate the structural contract for published course notes.

This checker intentionally does not judge whether a technical statement is
true. It catches repository-local invariants that are cheap to verify before a
human reviewer checks sources and explanation quality.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
COURSE_PATH_RE = re.compile(
    r"^M(?P<module>[0-9]) [^/]+/L(?P<lesson>[0-9]{2}) (?P<name>.+)\.md$"
)
COURSE_LINK_RE = re.compile(r"\[\[(L[0-9]{2} [^\]|#]+)(?:[\]|#])")
PROGRESS_RE = re.compile(
    r"^- \[(?P<state>[ xX])\] (?P<lesson>L[0-9]{2}) "
    r"(?P<name>.+?)(?:（[0-9]{4}-[0-9]{2}-[0-9]{2}）)?\s*$"
)
DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
REQUIRED_FIELDS = ("lesson", "module", "title", "status", "date", "terms", "prereqs", "tags")

THEORY_HEADINGS = (
    "## 论文里的这段话",
    "## 回到开头那段话",
    "## 术语卡片",
    "## 自测",
    "## 延伸阅读",
)
PRACTICE_HEADINGS = (
    "## 本次实践你要亲眼看到什么",
    "## 〇、环境准备",
    "## 一、逐步任务",
    "## 二、观察与解释",
    "## 三、挑战题",
    "## 回到本次实践",
    "## 术语卡片",
    "## 自测",
    "## 延伸阅读",
)
SENSITIVE_PATTERNS = (
    ("OpenAI-style API key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----")),
    ("absolute home path", re.compile(r"/(?:Users|home)/[^\s/]+")),
)
TEXT_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".base", ".txt"}


def parse_frontmatter(text: str) -> Dict[str, object]:
    """Parse the small YAML subset used by this vault without PyYAML."""
    match = FRONTMATTER_RE.match(text)
    if match is None:
        return {}

    values: Dict[str, object] = {}
    current_list: List[str] | None = None
    for raw_line in match.group("body").splitlines():
        if raw_line.startswith("  - "):
            if current_list is not None:
                current_list.append(raw_line[4:].strip().strip('"'))
            continue

        current_list = None
        key_match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):(?:\s*(.*))?$", raw_line)
        if key_match is None:
            continue
        key, raw_value = key_match.groups()
        raw_value = raw_value or ""
        if not raw_value:
            current_list = []
            values[key] = current_list
        else:
            values[key] = raw_value.strip().strip('"')
    return values


def course_targets_from_overview(root: Path) -> Set[str]:
    overview = root / "00 课程总览.md"
    if not overview.exists():
        return set()
    return set(COURSE_LINK_RE.findall(overview.read_text(encoding="utf-8")))


def progress_items(root: Path) -> Dict[str, bool]:
    board = root / "01 进度看板.md"
    if not board.exists():
        return {}
    result: Dict[str, bool] = {}
    for line in board.read_text(encoding="utf-8").splitlines():
        match = PROGRESS_RE.match(line)
        if match is None:
            continue
        target = f"{match.group('lesson')} {match.group('name')}"
        result[target] = match.group("state").lower() == "x"
    return result


def course_files(root: Path) -> Iterable[Tuple[Path, re.Match[str]]]:
    for path in sorted(root.glob("M*/L*.md")):
        relative = path.relative_to(root).as_posix()
        match = COURSE_PATH_RE.match(relative)
        if match is not None:
            yield path, match


def validate_note(
    root: Path,
    path: Path,
    path_match: re.Match[str],
    expected_targets: Set[str],
    board_items: Dict[str, bool],
) -> List[str]:
    errors: List[str] = []
    relative = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8")
    metadata = parse_frontmatter(text)
    target = f"L{path_match.group('lesson')} {path_match.group('name')}"

    if target not in expected_targets:
        errors.append(f"{relative}: filename {target!r} is not listed in 00 课程总览.md")
    for field in REQUIRED_FIELDS:
        if field not in metadata or metadata[field] in ("", []):
            errors.append(f"{relative}: missing frontmatter field {field!r}")

    expected_lesson = f"L{path_match.group('lesson')}"
    expected_module = f"M{path_match.group('module')}"
    if metadata.get("lesson") != expected_lesson:
        errors.append(f"{relative}: frontmatter lesson must be {expected_lesson}")
    if metadata.get("module") != expected_module:
        errors.append(f"{relative}: frontmatter module must be {expected_module}")
    if isinstance(metadata.get("date"), str) and not DATE_RE.match(str(metadata["date"])):
        errors.append(f"{relative}: date must use YYYY-MM-DD")
    if not isinstance(metadata.get("terms"), list) or not metadata.get("terms"):
        errors.append(f"{relative}: terms must be a non-empty YAML list")

    if metadata.get("status") == "已完成":
        headings = PRACTICE_HEADINGS if metadata.get("type") == "practice" else THEORY_HEADINGS
        for heading in headings:
            if heading not in text:
                errors.append(f"{relative}: completed note is missing heading {heading!r}")
        if "> [!example]" not in text:
            errors.append(f"{relative}: completed note is missing a 算一算 callout")
        if "> [!note]-" not in text:
            errors.append(f"{relative}: completed note is missing collapsed self-test answers")
        if board_items.get(target) is not True:
            errors.append(f"{relative}: completed note is not checked in 01 进度看板.md")

    for linked_target in COURSE_LINK_RE.findall(text):
        if linked_target not in expected_targets:
            errors.append(f"{relative}: wikilink target {linked_target!r} is not in 00 课程总览.md")
    return errors


def validate_progress(
    root: Path,
    expected_targets: Set[str],
    board_items: Dict[str, bool],
    notes_by_target: Dict[str, Dict[str, object]],
) -> List[str]:
    errors: List[str] = []
    for target, checked in board_items.items():
        if target not in expected_targets:
            errors.append(f"01 进度看板.md: {target!r} is not in 00 课程总览.md")
        if checked and target not in notes_by_target:
            errors.append(f"01 进度看板.md: checked course {target!r} has no note file")
    for target, metadata in notes_by_target.items():
        if metadata.get("status") == "已完成" and board_items.get(target) is not True:
            errors.append(f"{target}: status is 已完成 but progress board is not checked")
    return errors


def validate_sensitive_content(root: Path) -> List[str]:
    errors: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                relative = path.relative_to(root).as_posix()
                errors.append(f"{relative}: possible {label} found")
    return errors


def validate_repository(root: Path) -> List[str]:
    errors: List[str] = []
    required_files = ("00 课程总览.md", "01 进度看板.md", "02 术语总表.md")
    for name in required_files:
        if not (root / name).exists():
            errors.append(f"missing required repository file: {name}")
    if errors:
        return errors

    expected_targets = course_targets_from_overview(root)
    if not expected_targets:
        return ["00 课程总览.md: no Lxx course wikilinks found"]
    board_items = progress_items(root)
    notes_by_target: Dict[str, Dict[str, object]] = {}
    seen_lessons: Set[str] = set()

    for path, path_match in course_files(root):
        target = f"L{path_match.group('lesson')} {path_match.group('name')}"
        lesson = path_match.group("lesson")
        if lesson in seen_lessons:
            errors.append(f"duplicate course file for L{lesson}")
        seen_lessons.add(lesson)
        text = path.read_text(encoding="utf-8")
        notes_by_target[target] = parse_frontmatter(text)
        errors.extend(validate_note(root, path, path_match, expected_targets, board_items))

    errors.extend(validate_progress(root, expected_targets, board_items, notes_by_target))
    errors.extend(validate_sensitive_content(root))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root to validate",
    )
    args = parser.parse_args()
    errors = validate_repository(args.root.resolve())
    if errors:
        print("Course validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Course validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
