"""No-provider validation that Kaggle paired exports never leak task details."""
from pathlib import Path
import csv
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("bloodninja_kaggle_collect",
                                             ROOT/"bloodninja/kaggle_collect.py")
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScoreExportTests(unittest.TestCase):
    def record(self, root, arch, score, field="success"):
        run=root/"plancraft-test"/arch/"Qwen"/"test"/"run"
        (run/"instance_runs"/"0000").mkdir(parents=True,exist_ok=True)
        (run/"run_config.yaml").write_text(json.dumps({"agent":{"name":arch}}))
        (run/"instance_runs"/"0000"/"instance_save.yaml").write_text(json.dumps({
            "inp":{"inventory":{"stone":"1"},"target":"diamond"},
            "metrics":{field:score,"num_steps":8},
            "output":{"answer":"SECRET_RESPONSE_NOT_TO_BE_EXPORTED"}}))

    def test_paired_plan_craft_scores_are_stable_and_sanitized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"exp"
            self.record(root,"single-agent",False)
            self.record(root,"single-worker-queued",True)
            output=Path(tmp)/"scores.csv"
            fake_yaml=SimpleNamespace(safe_load=json.loads)
            with patch.dict(sys.modules,{"yaml":fake_yaml}):
                records=MODULE.export_scores(root,output)
            self.assertEqual(len(records),2)
            self.assertEqual(records[0]["task_id"],records[1]["task_id"])
            self.assertNotEqual(records[0]["arm"],records[1]["arm"])
            data=output.read_text()
            self.assertNotIn("diamond",data)
            self.assertNotIn("stone",data)
            self.assertNotIn("SECRET_RESPONSE_NOT_TO_BE_EXPORTED",data)
            self.assertIn("task_id,arm,score",data)

    def test_missing_score_is_not_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"exp"
            self.record(root,"single-agent",False,field="unknown_score")
            with patch.dict(sys.modules,{"yaml":SimpleNamespace(safe_load=json.loads)}):
                with self.assertRaisesRegex(ValueError,"Missing/invalid benchmark binary score"):
                    MODULE.export_scores(root,Path(tmp)/"scores.csv")


if __name__=="__main__":
    unittest.main()
