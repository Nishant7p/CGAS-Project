"""Tests for the API runner's retry and quota policy (no network): generation/run_api.py."""

import datetime as dt
import json

import pytest

from generation.run_api import (
    MAX_WAIT_PER_CALL_S,
    Quota,
    StopRun,
    call_with_retries,
    is_daily_quota,
    quota_day,
    quota_violation,
    seconds_until_daily_reset,
)

GEMINI_DAILY = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaDimensions": {"location": "global", "model": "gemini-3.5-flash-lite"},
                        "quotaValue": "500",
                    }
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"},
        ],
    }
}
GEMINI_MINUTE = json.loads(
    json.dumps(GEMINI_DAILY).replace("PerDayPerProjectPerModel", "PerMinutePerProjectPerModel").replace('"500"', '"10"')
)


class ApiError(Exception):
    """Shaped like google.genai.errors.APIError: an int .code, a .status string and the parsed JSON .details."""

    def __init__(self, code, details=None, text=""):
        super().__init__(f"{code} {text or json.dumps(details)}")
        self.code, self.status, self.details = code, "ERR", details


P = {"key": "v1|KOR|plain|00", "system": "s", "user": "u"}


def scripted(*outcomes):
    """A fake API call that raises or returns the given outcomes in order."""
    it = iter(outcomes)

    def call(system, user, seed):
        o = next(it)
        if isinstance(o, Exception):
            raise o
        return o

    return call


def test_quota_violation_reads_dict_details_and_repr_strings():
    assert quota_violation(ApiError(429, GEMINI_DAILY)) == (
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        500,
        37.0,
    )
    as_repr = Exception("429 RESOURCE_EXHAUSTED. " + repr(GEMINI_DAILY))  # str(exc) uses Python quoting
    assert quota_violation(as_repr) == ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", 500, 37.0)
    assert quota_violation(Exception("503 overloaded")) == (None, None, None)


def test_is_daily_quota():
    assert is_daily_quota("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "")
    assert not is_daily_quota("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quota exceeded")
    assert is_daily_quota(None, "Rate limit reached on tokens per day (TPD): Limit 200000")  # Groq wording


def test_daily_reset_is_midnight_pacific_and_dst_aware():
    utc = dt.timezone.utc
    assert seconds_until_daily_reset("gemini", dt.datetime(2026, 11, 22, 13, 0, tzinfo=utc)) == 19 * 3600  # 05:00 PST
    # 2026-03-08 is the US DST start: from 01:00 PST that day to midnight PDT is 22 real hours, not 23.
    assert seconds_until_daily_reset("gemini", dt.datetime(2026, 3, 8, 9, 0, tzinfo=utc)) == 22 * 3600
    assert seconds_until_daily_reset("groq") == 900.0
    assert quota_day("gemini", dt.datetime(2026, 11, 23, 5, 0, tzinfo=utc)) == "2026-11-22"  # still the 22nd in LA
    assert quota_day("groq", dt.datetime(2026, 11, 23, 5, 0, tzinfo=utc)) == "2026-11-23"


def test_transient_errors_back_off_then_succeed(tmp_path):
    slept = []
    call = scripted(ApiError(503, text="overloaded"), ApiError(429, GEMINI_MINUTE), {"text": "ok"})
    res, started, latency, n_retries = call_with_retries(
        call, P, 1, "gemini", Quota({}, tmp_path / "q.json", "gemini"), wait_daily=False, sleep=slept.append
    )
    assert res == {"text": "ok"} and n_retries == 2 and latency >= 0 and started.endswith("+00:00")
    assert slept == [10, 38]  # exponential backoff, then the server's retryDelay (37 s) + 1 s


def test_daily_quota_stops_cleanly_and_records_the_server_limit(tmp_path):
    q = Quota({}, tmp_path / "q.json", "gemini")
    with pytest.raises(StopRun) as e:
        call_with_retries(scripted(ApiError(429, GEMINI_DAILY)), P, 1, "gemini", q, wait_daily=False, sleep=None)
    assert e.value.exit_code == 0
    saved = json.loads((tmp_path / "q.json").read_text())
    assert saved["server_daily_quota"]["limit"] == 500
    assert saved["server_daily_quota"]["quota_id"].startswith("GenerateRequestsPerDay")


def test_daily_quota_with_wait_sleeps_until_reset_then_continues(tmp_path):
    slept = []
    call = scripted(ApiError(429, GEMINI_DAILY), {"text": "ok"})
    res, *_ = call_with_retries(
        call, P, 1, "gemini", Quota({}, tmp_path / "q.json", "gemini"), wait_daily=True, sleep=slept.append
    )
    assert res == {"text": "ok"}
    assert len(slept) == 1 and 120 < slept[0] <= 24 * 3600 + 120


@pytest.mark.parametrize("code,exit_code", [(401, 1), (403, 1), (400, 1)])
def test_auth_and_bad_requests_stop_immediately(tmp_path, code, exit_code):
    q = Quota({}, tmp_path / "q.json", "gemini")
    with pytest.raises(StopRun) as e:
        call_with_retries(scripted(ApiError(code, text="nope")), P, 1, "gemini", q, wait_daily=True, sleep=None)
    assert e.value.exit_code == exit_code


def test_persistent_failures_give_up_after_the_wait_budget(tmp_path):
    slept = []
    overloaded = ApiError(503, text="This model is currently experiencing high demand.")
    call = scripted(*[overloaded] * 1000)
    with pytest.raises(StopRun) as e:
        call_with_retries(call, P, 1, "gemini", Quota({}, tmp_path / "q.json"), wait_daily=True, sleep=slept.append)
    assert e.value.exit_code == 1 and slept[:6] == [10, 20, 40, 80, 160, 300] and max(slept) == 300
    assert MAX_WAIT_PER_CALL_S - 300 < sum(slept) <= MAX_WAIT_PER_CALL_S  # hours of patience, then a clean stop


def test_quota_rolls_over_to_a_new_day_but_keeps_the_server_limit(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"date": "2000-01-01", "requests": 9, "tokens": 9, "server_daily_quota": {"limit": 7}}))
    q = Quota({"rpd": 10}, path, "gemini")
    assert q.state["requests"] == 0 and q.state["server_daily_quota"] == {"limit": 7}
    assert q.daily_left()
    q.record(100)
    assert json.loads(path.read_text())["tokens"] == 100
