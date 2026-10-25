import unittest
from types import SimpleNamespace
from bloodninja.grading import strict_grade_one_criterion

class Template:
    def compile(self, **kwargs):
        return kwargs

class Judge:
    def __init__(self, events):
        self.events = list(events)
        self.calls = 0
    def invoke(self, messages):
        event = self.events[self.calls]
        self.calls += 1
        if isinstance(event, BaseException):
            raise event
        return SimpleNamespace(text=lambda: event)

class StrictGraderTests(unittest.TestCase):
    @staticmethod
    def dataset(judge):
        return SimpleNamespace(
            eval_prompts={'grader':Template()}, eval_llm=judge,
            _yes=lambda response: 'correct: yes' in response.lower())

    def test_recoverable_provider_error_is_not_incorrect(self):
        judge = Judge([RuntimeError('520'), 'correct: yes'])
        self.assertTrue(strict_grade_one_criterion(
            self.dataset(judge), 'q','response',
            {'operator':'correctness','criteria':'fact'}, sleep=lambda _:None))
        self.assertEqual(judge.calls,2)

    def test_unrecoverable_provider_error_invalidates(self):
        judge = Judge([RuntimeError('520')]*3)
        with self.assertRaisesRegex(RuntimeError,'INVALID, not incorrect'):
            strict_grade_one_criterion(
                self.dataset(judge), 'q','response',
                {'operator':'correctness','criteria':'fact'}, sleep=lambda _:None)
        self.assertEqual(judge.calls,3)

    def test_upstream_contradiction_inversion_preserved(self):
        judge=Judge(['correct: no'])
        self.assertTrue(strict_grade_one_criterion(
            self.dataset(judge), 'q','response',
            {'operator':'contradiction','criteria':'fact'},sleep=lambda _:None))

    def test_missing_criterion_preserves_upstream_behavior(self):
        judge=Judge([])
        self.assertFalse(strict_grade_one_criterion(
            self.dataset(judge),'q','response',{'criteria':''},sleep=lambda _:None))
        self.assertEqual(judge.calls,0)

if __name__=='__main__':
    unittest.main()
