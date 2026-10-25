"""Fail closed on transient Finance-Agent rubric-grader errors.

The pinned upstream implementation catches every exception raised by the
rubric judge and returns False. That silently conflates provider outages
with incorrect answers. This patch keeps its rubric and judgment semantics,
but retries transient failures before raising rather than fabricating misses.
Applies equally to S, M, and Q in the free-tier comparison runner.
"""
from __future__ import annotations

import time


def strict_grade_one_criterion(dataset, question, response, criterion, *,
                               attempts=3, sleep=time.sleep):
    operator = (criterion.get("operator") or "").strip().lower()
    criteria_text = (criterion.get("criteria") or "").strip()
    if not criteria_text:
        return False  # Existing upstream behavior for empty criteria.
    assert dataset.eval_prompts is not None
    assert dataset.eval_llm is not None
    messages = dataset.eval_prompts["grader"].compile(
        question=question, response=response, correct_answer=criteria_text
    )
    for attempt in range(attempts):
        try:
            judge_text = dataset.eval_llm.invoke(messages).text()
            passed = dataset._yes(judge_text)
            return not passed if operator == "contradiction" else passed
        except Exception as exc:
            if attempt + 1 >= attempts:
                raise RuntimeError(
                    "Finance-Agent grader unavailable after "
                    f"{attempts} attempts; result is INVALID, not incorrect"
                ) from exc
            sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def install_strict_grading():
    """Patch only the pinned, already-imported Finance-Agent dataset class."""
    from agent_scaling.datasets.finance_agent import FinanceAgentDataset
    def strict(self, question, response, criterion):
        return strict_grade_one_criterion(self, question, response, criterion)
    FinanceAgentDataset._grade_one_criterion = strict
