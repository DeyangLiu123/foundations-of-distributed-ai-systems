from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_course.py"
SPEC = importlib.util.spec_from_file_location("validate_course", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


VALID_NOTE = """---
lesson: L01
module: M0
title: Test
status: 已完成
date: 2026-07-21
terms:
  - test term
prereqs:
  - \"[[L01 Test]]\"
tags:
  - course/M0
---

# L01 Test

## 论文里的这段话

## 回到开头那段话

> [!example] 算一算
> $1 + 1 = 2$。

## 术语卡片

## 自测

> [!note]- 参考答案
> 正确。

## 延伸阅读
"""


def write_fixture(root: Path, note: str = VALID_NOTE, progress: str | None = None) -> None:
    (root / "00 课程总览.md").write_text(
        "# 总览\n\n| # | 课程 |\n|---|---|\n| L01 | [[L01 Test]] |\n",
        encoding="utf-8",
    )
    (root / "01 进度看板.md").write_text(
        progress or "# 进度\n\n- [x] L01 Test（2026-07-21）\n", encoding="utf-8"
    )
    (root / "02 术语总表.md").write_text("# 术语\n", encoding="utf-8")
    course_dir = root / "M0 Intro"
    course_dir.mkdir()
    (course_dir / "L01 Test.md").write_text(note, encoding="utf-8")


class ValidateCourseTests(unittest.TestCase):
    def validate(self, note: str = VALID_NOTE, progress: str | None = None) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_fixture(root, note=note, progress=progress)
            return validator.validate_repository(root)

    def test_valid_repository_passes(self) -> None:
        self.assertEqual(self.validate(), [])

    def test_missing_frontmatter_is_reported(self) -> None:
        errors = self.validate(note="# L01 Test\n")
        self.assertTrue(any("missing frontmatter field 'lesson'" in error for error in errors))

    def test_wrong_lesson_number_is_reported(self) -> None:
        errors = self.validate(note=VALID_NOTE.replace("lesson: L01", "lesson: L02"))
        self.assertTrue(any("frontmatter lesson must be L01" in error for error in errors))

    def test_missing_heading_is_reported(self) -> None:
        errors = self.validate(note=VALID_NOTE.replace("## 延伸阅读\n", ""))
        self.assertTrue(any("missing heading '## 延伸阅读'" in error for error in errors))

    def test_unchecked_completed_note_is_reported(self) -> None:
        errors = self.validate(progress="# 进度\n\n- [ ] L01 Test\n")
        self.assertTrue(any("not checked" in error for error in errors))

    def test_filename_not_in_overview_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_fixture(root)
            course = root / "M0 Intro" / "L01 Test.md"
            course.rename(root / "M0 Intro" / "L01 Different title.md")
            errors = validator.validate_repository(root)
        self.assertTrue(any("is not listed in 00 课程总览.md" in error for error in errors))

    def test_secret_pattern_is_reported(self) -> None:
        fake_key = "sk-" + "abcdefghijklmnop"
        errors = self.validate(note=VALID_NOTE + f"\nsecret = {fake_key}\n")
        self.assertTrue(any("possible OpenAI-style API key" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
