import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloodninja.export_metrics import export_metrics
from analyze import summarize, read

try:
    import yaml
except ImportError:
    yaml = None

@unittest.skipIf(yaml is None, 'PyYAML not installed; upstream compatibility job installs it')
class ScoreExportTests(unittest.TestCase):
    def test_only_task_scores_exported_with_pairable_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'exp_outputs'
            sensitive = 'PRIVATE ANSWER AND PROMPT SHOULD NOT ESCAPE'
            for arm, score in [('single-agent', False),
                               ('multi-agent-centralized', True),
                               ('single-worker-queued', True)]:
                d = root / 'finance_agent' / arm / 'model' / 'dated-run'
                file = d / 'instance_runs' / '0000' / 'instance_save.yaml'
                file.parent.mkdir(parents=True)
                (d / 'run_config.yaml').write_text(yaml.safe_dump({'agent': {'name': arm}}))
                file.write_text(yaml.safe_dump({
                    'inp': {'task_id': 'finance-Q001', 'question': sensitive},
                    'output': {'answer': sensitive},
                    'metrics': {'is_correct': score, 'rubric_score': float(score),
                                'grading_method': 'rubric', 'extracted_answer': sensitive},
                    'expected_output': sensitive}))
            out = Path(tmp) / 'metrics.csv'
            rows = export_metrics(root, out)
            self.assertEqual(len(rows), 3)
            self.assertNotIn(sensitive, out.read_text())
            self.assertEqual(set(row['arm'] for row in rows), {'single','multi','queue'})
            self.assertEqual(summarize(read(out), reps=200)[0], 1)

    def test_no_silent_fabrication_when_score_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'finance_agent' / 'single-agent'
            p = d / 'instance_runs' / '0000' / 'instance_save.yaml'
            p.parent.mkdir(parents=True)
            (d / 'run_config.yaml').write_text(yaml.safe_dump({'agent': {'name': 'single-agent'}}))
            p.write_text(yaml.safe_dump({'inp': {'task_id': 'X'}, 'metrics': {'rubric_score': 1}}))
            with self.assertRaisesRegex(ValueError, 'binary accuracy'):
                export_metrics(Path(tmp), Path(tmp)/'out.csv')

if __name__ == '__main__':
    unittest.main()
