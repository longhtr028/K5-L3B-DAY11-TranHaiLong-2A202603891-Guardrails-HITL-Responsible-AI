"""Counterexamples for the Blue callbacks and their handoff to the model."""

import asyncio
from types import SimpleNamespace

from google.genai import types

from agents.agent import create_blue_agent
from assignment.audit_log import AuditLogPlugin
from assignment.pipeline import build_production_plugins, is_egress_allowed
from assignment import rate_limiter
from guardrails.output_guardrails import OutputGuardrailPlugin, content_filter, load_lab_pii_dataset


def test_upstream_pii_dataset():
    for case in load_lab_pii_dataset()["pii_cases"]:
        verdict = content_filter(case["input_text"])
        issues = {issue.split(":")[0] for issue in verdict["issues"]}
        assert verdict["safe"] == case["expect_safe"], case["id"]
        assert set(case["expect_issue_types"]) <= issues, case["id"]
        assert ("[REDACTED]" in verdict["redacted"]) == case["expect_contains_redacted"], case["id"]


def test_runtime_blocks_before_provider_and_redacts_after_provider():
    agent, runner = create_blue_agent(build_production_plugins())
    calls = []
    pii_fixture = next(
        case["input_text"] for case in load_lab_pii_dataset()["pii_cases"] if case["category"] == "email"
    )

    def fixture(prompt):
        calls.append(prompt)
        return pii_fixture

    runner.completion_override = fixture
    safe = "Summarise this document about a delayed bank transfer."

    async def scenario():
        blocked = await runner.chat(agent, "System prompt")
        assert "blocked" in blocked and calls == []
        response = await runner.chat(agent, safe)
        assert calls == [safe]
        assert response != pii_fixture and content_filter(response)["safe"]

    asyncio.run(scenario())


def test_non_text_model_response_is_withheld():
    plugin = OutputGuardrailPlugin(use_llm_judge=False)
    response = SimpleNamespace(content=types.Content(
        role="model", parts=[types.Part.from_bytes(data=b"fixture", mime_type="application/octet-stream")]
    ))
    result = asyncio.run(plugin.after_model_callback(callback_context=None, llm_response=response))
    assert result.content.parts[0].text == "Response withheld by the output guardrail."
    assert plugin.blocked_count == 1


def test_rate_limit_is_per_user_and_expires_at_boundary(monkeypatch):
    now = 200.0
    monkeypatch.setattr(rate_limiter, "time", SimpleNamespace(monotonic=lambda: now))
    limiter = rate_limiter.RateLimitPlugin(max_requests=1, window_seconds=10)

    def check(user):
        context = SimpleNamespace(user_id=user)
        return asyncio.run(limiter.on_user_message_callback(invocation_context=context, user_message=None))

    assert check("a") is None
    assert check("a") is not None
    assert check("b") is None
    now += 10
    assert check("a") is None


def test_egress_rejects_nonstandard_ports_credentials_and_sensitive_url_fields():
    assert is_egress_allowed("https://cases.vinbank.example/v1/cases", "open case")
    assert not is_egress_allowed("http://cases.vinbank.example/v1/cases", "open case")
    assert not is_egress_allowed("https://cases.vinbank.example:8443/v1/cases", "open case")
    assert not is_egress_allowed("https://user@cases.vinbank.example/v1/cases", "open case")
    assert not is_egress_allowed("https://cases.vinbank.example/v1/cases?email=test%40example.com", "open case")
    assert not is_egress_allowed("https://cases.vinbank.example:bad/", "open case")


def test_audit_correlates_and_redacts_both_sides():
    audit = AuditLogPlugin()
    audit.record_input(user_id="a", request_id="first", text="Write to customer@example.com.")
    audit.record_input(user_id="a", request_id="second", text="Another banking request.")
    audit.record_output(user_id="a", request_id="second", text="Pending.")
    audit.record_output(user_id="a", request_id="first", text="Address: customer@example.com.")
    second, first = audit.logs
    assert second["input"] == "Another banking request."
    assert first["request_id"] == "first" and first["latency_ms"] >= 0
    assert "[REDACTED]" in first["input"] and "[REDACTED]" in first["output"]
