import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bloodninja_kaggle_prepare", ROOT / "kaggle/prepare.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KagglePreparationTests(unittest.TestCase):
    def test_private_pinned_submission_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "kaggle").mkdir()
            (base / "kaggle/runner.py").write_text("import json\nCONFIG=json.loads('__BLOODNINJA_CONFIG__')\n")
            out = base / "submission"
            meta = MODULE.prepare(base, "Researcher_3", "a" * 40, "plancraft-test", "single", 1, out)
            self.assertEqual(meta["id"], "researcher_3/bloodninja-control")
            self.assertTrue(meta["enable_gpu"])
            self.assertTrue(meta["enable_internet"])
            self.assertTrue(meta["is_private"])
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(meta["kernel_type"], "script")
            code = (out / "runner.py").read_text()
            self.assertNotIn("__BLOODNINJA_CONFIG__", code)
            self.assertIn('"max_instances": 1', code)
            self.assertIn("a" * 40, code)
            self.assertEqual(json.loads((out / "config-public.json").read_text())["arm"], "single")
            self.assertFalse((out / "kaggle.json").exists())

    def test_invalid_source_and_configuration_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root / "kaggle").mkdir()
            (root / "kaggle/runner.py").write_text("'__BLOODNINJA_CONFIG__'")
            for bad_user, bad_sha, bad_dataset, bad_count in [
                ("../oops", "b"*40, "plancraft-test", 1),
                ("researcher", "main", "plancraft-test", 1),
                ("researcher", "b"*40, "unsupported", 1),
                ("researcher", "b"*40, "finance-agent", 0),
            ]:
                with self.subTest(username=bad_user, sha=bad_sha, dataset=bad_dataset, n=bad_count):
                    with self.assertRaises(ValueError):
                        MODULE.prepare(root,bad_user,bad_sha,bad_dataset,"single",bad_count,root/"out")

    def test_real_kaggle_runner_syntax(self):
        import ast
        code=(ROOT/"kaggle/runner.py").read_text()
        ast.parse(code)
        self.assertIn("'__BLOODNINJA_CONFIG__'", code)
        self.assertNotIn("KAGGLE_API_TOKEN", code)
        self.assertIn("Qwen/Qwen3-8B-AWQ", code)


if __name__ == "__main__":
    unittest.main()
