from __future__ import annotations

from typing import Any, Dict, List, Tuple


CRITICAL_CHECKS = {
    "create_reminder",
    "create_open_loop",
    "complete_open_loop",
    "dismiss_open_loop",
}


def _bool(value: Any) -> bool:
    return bool(value)


def _normalize_expected_key(key: str) -> str:
    aliases = {
        "complete_open_loop": "complete_open_loop",
        "dismiss_open_loop": "dismiss_open_loop",
        "surface_other_items": "surface_other_items",
        "ask_follow_up": "ask_follow_up",
        "create_reminder": "create_reminder",
        "create_open_loop": "create_open_loop",
    }
    return aliases.get(key, key)


def collect_auto_checks(
    expected: Dict[str, Any],
    observed: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return deterministic checks only.

    We intentionally do not auto-score subjective qualities such as
    helpfulness, naturalness, or whether a planning answer was concrete
    enough. Those remain in the manual-review section.
    """
    checks: List[Dict[str, Any]] = []

    auto_keys = [
        "ask_follow_up",
        "create_reminder",
        "create_open_loop",
        "complete_open_loop",
        "dismiss_open_loop",
        "surface_other_items",
    ]

    for raw_key in auto_keys:
        if raw_key not in expected:
            continue

        key = _normalize_expected_key(raw_key)
        expected_value = _bool(expected[raw_key])
        observed_value = _bool(observed.get(key))
        checks.append(
            {
                "key": key,
                "expected": expected_value,
                "observed": observed_value,
                "passed": expected_value == observed_value,
                "critical": key in CRITICAL_CHECKS,
            }
        )

    return checks


def summarize_checks(
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total = len(checks)
    passed = sum(1 for check in checks if check["passed"])
    critical_failures = [
        check
        for check in checks
        if check["critical"] and not check["passed"]
    ]

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "percent": round((passed / total) * 100, 1) if total else None,
        "critical_failures": critical_failures,
    }


def manual_review_fields(
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "expected_intent": expected.get("intent"),
        "expected_reference_resolution": expected.get(
            "reference_resolution"
        ),
        "expected_use_memory": expected.get("use_memory"),
        "expected_response_goal": expected.get("response_goal"),
        "manual_scores": {
            "intent_accuracy": None,
            "context_reference_accuracy": None,
            "restraint": None,
            "state_change_accuracy": None,
            "helpfulness": None,
        },
        "manual_total_out_of_10": None,
        "review_notes": "",
    }