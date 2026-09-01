# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval.eval_harness import _load_tasks_config

_OLMO_EVALS = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "configs"
    / "olmo_evals.yaml"
)
_PAPER_TASKS = [
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
    "gsm8k",
    "mbpp",
    "humaneval",
    "hendrycks_math",
    "drop",
]


class TestLoadTasksConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, text: str) -> str:
        path = self.tmp / name
        path.write_text(text)
        return str(path)

    def test_olmo_evals_has_the_ten_paper_tasks(self):
        tasks = _load_tasks_config(str(_OLMO_EVALS))
        self.assertEqual([t["task"] for t in tasks], _PAPER_TASKS)
        drop = tasks[-1]
        self.assertEqual(drop["num_fewshot"], 3)
        self.assertEqual(drop["generation_kwargs"]["max_gen_toks"], 100)

    def test_olmo_evals_rebases_under_dataset_root(self):
        root = self.tmp / "eval_data"
        for name in {
            "hellaswag",
            "winogrande",
            "ai2_arc",
            "mmlu",
            "gsm8k",
            "mbpp",
            "humaneval",
            "hendrycks_math",
            "drop",
        }:
            (root / name).mkdir(parents=True)
        tasks = _load_tasks_config(str(_OLMO_EVALS), str(root))
        by_task = {t["task"]: t["dataset_path"] for t in tasks}
        self.assertEqual(by_task["hellaswag"], str(root / "hellaswag"))
        self.assertEqual(by_task["arc_easy"], str(root / "ai2_arc"))
        self.assertEqual(by_task["arc_challenge"], str(root / "ai2_arc"))
        # Per-task overrides other than dataset_path are untouched.
        self.assertEqual(tasks[-1]["num_fewshot"], 3)

    def test_rebase_uses_basename_and_skips_entries_without_path(self):
        root = self.tmp / "root"
        (root / "gsm8k").mkdir(parents=True)
        path = self._write(
            "tasks.yaml",
            "tasks:\n"
            "  - hellaswag\n"
            "  - task: winogrande\n"
            "  - task: gsm8k\n"
            "    dataset_path: /cluster/eval_data/gsm8k\n",
        )
        tasks = _load_tasks_config(path, str(root))
        self.assertEqual(tasks[0], "hellaswag")
        self.assertEqual(tasks[1], {"task": "winogrande"})
        self.assertEqual(
            tasks[2], {"task": "gsm8k", "dataset_path": str(root / "gsm8k")}
        )

    def test_without_dataset_root_entries_are_verbatim(self):
        path = self._write(
            "tasks.json",
            json.dumps({"tasks": [{"task": "gsm8k", "dataset_path": "gsm8k"}]}),
        )
        self.assertEqual(
            _load_tasks_config(path), [{"task": "gsm8k", "dataset_path": "gsm8k"}]
        )

    def test_top_level_list_is_accepted(self):
        path = self._write("tasks.yaml", "- hellaswag\n- arc_easy\n")
        self.assertEqual(_load_tasks_config(path), ["hellaswag", "arc_easy"])

    def test_missing_dataset_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            _load_tasks_config(str(_OLMO_EVALS), str(self.tmp / "absent"))

    def test_missing_rebased_directory_raises(self):
        root = self.tmp / "root"
        root.mkdir()
        path = self._write(
            "tasks.yaml", "tasks:\n  - task: gsm8k\n    dataset_path: gsm8k\n"
        )
        with self.assertRaises(FileNotFoundError):
            _load_tasks_config(path, str(root))

    def test_non_string_dataset_path_raises_with_dataset_root(self):
        root = self.tmp / "root"
        root.mkdir()
        path = self._write(
            "tasks.yaml", "tasks:\n  - task: gsm8k\n    dataset_path: 3\n"
        )
        with self.assertRaises(ValueError):
            _load_tasks_config(path, str(root))

    def test_invalid_specs_raise(self):
        with self.assertRaises(FileNotFoundError):
            _load_tasks_config(str(self.tmp / "absent.yaml"))
        with self.assertRaises(ValueError):
            _load_tasks_config(self._write("tasks.txt", "hellaswag\n"))
        with self.assertRaises(ValueError):
            _load_tasks_config(self._write("empty.yaml", "tasks: []\n"))
        with self.assertRaises(ValueError):
            _load_tasks_config(self._write("bad.yaml", "foo: bar\n"))


if __name__ == "__main__":
    unittest.main()
