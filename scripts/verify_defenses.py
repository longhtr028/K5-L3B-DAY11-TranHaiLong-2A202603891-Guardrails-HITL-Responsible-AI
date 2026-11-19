"""Generate a supplemental report from saved Blue results and defensive tests."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"


def test_counts(cases):
    counts = dict(passed=0, failed=0, errors=0, skipped=0)
    for case in cases:
        if case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    return counts


def main():
    raw_results = (OUTPUT / "results.json").read_bytes()
    results = json.loads(raw_results)
    schema = json.loads((ROOT / "schemas" / "results.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(results, schema)

    with tempfile.TemporaryDirectory() as temp:
        xml_path = Path(temp) / "tests.xml"
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/smoke", "tests/public", "-q",
             "--junitxml", str(xml_path)],
            cwd=ROOT,
            check=False,
        )
        cases = ElementTree.parse(xml_path).findall(".//testcase") if xml_path.is_file() else []

    red_advance = [
        case for case in cases
        if case.get("classname", "").split(".")[-1] == "test_red_advance_nemo"
    ]
    report = {
        "scope": "Supplemental defensive evidence",
        "blue": {
            "source": "outputs/results.json",
            "source_sha256": hashlib.sha256(raw_results).hexdigest(),
            "schema_valid": True,
            "model_execution": results.get("model_execution"),
            "provider": results.get("llm_provider"),
            "model": results.get("llm_model"),
            "banking_requests_allowed": sum(not row["blocked"] for row in results["safe_queries"]),
            "banking_requests_total": len(results["safe_queries"]),
            "defensive_probes_blocked": sum(row["blocked"] for row in results["attack_queries"]),
            "defensive_probes_total": len(results["attack_queries"]),
            "rate_limit": results["rate_limit"],
            "egress": results.get("egress"),
        },
        "test_exit_code": run.returncode,
        "smoke_and_public_tests": test_counts(cases),
        "red_advance_safeguard_tests": {
            "source": "tests/smoke/test_red_advance_nemo.py",
            "execution": "offline fixtures",
            **test_counts(red_advance),
        },
        "course_checkpoint_4_artifact_present": (OUTPUT / "attack_results.json").is_file(),
    }
    path = OUTPUT / "defense_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
