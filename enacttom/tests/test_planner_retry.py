from types import SimpleNamespace

import pytest

from enacttom.planner import BenchmarkLLMInfrastructureError, EmtomPlanner
from habitat_llm.llm.base_llm import LLMRequestError


def _make_planner() -> EmtomPlanner:
    planner = object.__new__(EmtomPlanner)
    planner.planner_config = SimpleNamespace()
    return planner


def test_generate_with_retry_retries_rate_limit(monkeypatch):
    planner = _make_planner()
    sleep_calls = []
    monkeypatch.setattr("enacttom.planner.time.sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr("enacttom.planner.random.random", lambda: 0.5)

    class RetryThenSucceedLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            del prompt, kwargs
            self.calls += 1
            if self.calls == 1:
                raise LLMRequestError(
                    "rate limited",
                    status_code=429,
                    headers={"retry-after": "7"},
                    retryable=True,
                )
            return "ok"

    llm = RetryThenSucceedLLM()
    assert planner._generate_with_retry(llm, "prompt", stop="Assigned!") == "ok"
    assert llm.calls == 2
    assert sleep_calls == [7.0]


def test_generate_with_retry_stops_after_cap(monkeypatch):
    planner = _make_planner()
    sleep_calls = []
    monkeypatch.setattr("enacttom.planner.time.sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr("enacttom.planner.random.random", lambda: 0.5)

    class AlwaysRateLimitedLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            del prompt, kwargs
            self.calls += 1
            raise LLMRequestError(
                "rate limited",
                status_code=429,
                retryable=True,
            )

    llm = AlwaysRateLimitedLLM()
    with pytest.raises(BenchmarkLLMInfrastructureError, match="failed after"):
        planner._generate_with_retry(llm, "prompt")
    assert llm.calls == planner.LLM_RETRY_ATTEMPTS
    assert len(sleep_calls) == planner.LLM_RETRY_ATTEMPTS - 1


def test_generate_with_retry_does_not_retry_nontransient_error(monkeypatch):
    planner = _make_planner()
    sleep_calls = []
    monkeypatch.setattr("enacttom.planner.time.sleep", lambda seconds: sleep_calls.append(seconds))

    class PermanentFailureLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, **kwargs):
            del prompt, kwargs
            self.calls += 1
            raise ValueError("No OPENAI_API_KEY provided")

    llm = PermanentFailureLLM()
    with pytest.raises(BenchmarkLLMInfrastructureError, match="Non-retryable benchmark LLM failure"):
        planner._generate_with_retry(llm, "prompt")
    assert llm.calls == 1
    assert sleep_calls == []
