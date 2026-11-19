"""Privacy-preserving request and decision logs for the Blue pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Observe requests without retaining raw secret or PII matches."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[float, dict]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        key = (user_id, request_id)
        if key in self._open:
            raise ValueError("duplicate in-flight request")
        self._open[key] = (monotonic(), {
            "input": content_filter(text)["redacted"],
            "started_at": utc_now_iso(),
        })

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        start, request = self._open.pop((user_id, request_id), (None, {}))
        self.logs.append({
            "user_id": user_id, "request_id": request_id, **request,
            "completed_at": utc_now_iso(),
            "output": content_filter(text)["redacted"],
            "blocked": blocked, "layer": layer,
            "latency_ms": round((monotonic() - start) * 1000, 3) if start is not None else None,
        })

    def export_json(self, filepath: str | None = None):
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
