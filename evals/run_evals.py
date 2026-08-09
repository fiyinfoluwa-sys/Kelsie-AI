from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple


# Allow: python3 evals/run_evals.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import main as kelsie  # noqa: E402
from evals.scoring import (  # noqa: E402
    collect_auto_checks,
    manual_review_fields,
    summarize_checks,
)


DEFAULT_PROFILE: Dict[str, Any] = {
    "name": "Eval User",
    "timezone": "America/Toronto",
    "proactivity": "balanced",
    "proactivity_level": "balanced",
    "quiet_hours_enabled": False,
    "quiet_hours_start": "23:00",
    "quiet_hours_end": "08:00",
    "memory_enabled": True,
    "adaptive_tone": True,
    "accessibility_simplified_language": False,
}


RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit_exceeded",
    "error code: 429",
    "status code: 429",
    "tokens per day",
)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error
    return cases


def normalize_profile(case: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(DEFAULT_PROFILE)
    raw = case.get("profile")
    if isinstance(raw, dict):
        profile.update(raw)

    # Some eval cases specify quiet_hours_active as an intended condition
    # rather than literal times. Translate that into a quiet-hours window
    # containing the current local time.
    if bool(profile.pop("quiet_hours_active", False)):
        local_now = kelsie.get_profile_local_datetime(profile)
        start_hour = (local_now.hour - 1) % 24
        end_hour = (local_now.hour + 1) % 24
        profile["quiet_hours_enabled"] = True
        profile["quiet_hours_start"] = f"{start_hour:02d}:00"
        profile["quiet_hours_end"] = f"{end_hour:02d}:59"

    proactivity = str(
        profile.get("proactivity")
        or profile.get("proactivity_level")
        or "balanced"
    )
    profile["proactivity"] = proactivity
    profile["proactivity_level"] = proactivity
    return profile


def make_memory_record(case: Dict[str, Any]) -> Dict[str, Any]:
    raw_items = case.get("memory")
    memory: Dict[str, List[Dict[str, Any]]] = {
        "facts": [],
        "relationships": [],
        "situations": [],
        "preferences": [],
        "patterns": [],
    }

    if not isinstance(raw_items, list):
        return {"memory": memory}

    category_map = {
        "person": "relationships",
        "relationship": "relationships",
        "goal": "situations",
        "project": "situations",
        "deadline": "situations",
        "situation": "situations",
        "preference": "preferences",
        "pattern": "patterns",
        "fact": "facts",
    }

    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue

        raw_type = str(item.get("type") or "fact").strip().lower()
        category = category_map.get(raw_type, "facts")
        value = str(item.get("value") or "").strip()

        if not value:
            continue

        memory[category].append(
            {
                "key": f"eval:{raw_type}:{index}",
                "value": value,
                "confidence": 0.95,
            }
        )

    return {"memory": memory}


def make_open_loop_state(case: Dict[str, Any]) -> Dict[str, Any]:
    raw_items = case.get("open_loops")
    output: List[Dict[str, Any]] = []

    if isinstance(raw_items, list):
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                continue

            output.append(
                {
                    "id": int(item.get("id") or index),
                    "action": str(item.get("action") or "").strip(),
                    "person": item.get("person"),
                    "project": item.get("project"),
                    "timing_text": item.get("timing_text"),
                    "scheduled_for": item.get("scheduled_for"),
                    "status": str(item.get("status") or "open"),
                    "confidence": 0.95,
                }
            )

    return {"open": output}


def make_reminder_state(case: Dict[str, Any]) -> Dict[str, Any]:
    raw_items = case.get("reminders")
    output: List[Dict[str, Any]] = []

    if isinstance(raw_items, list):
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                continue

            copy = dict(item)
            copy.setdefault("id", index)
            output.append(copy)

    return {"upcoming": output}


def conversation_without_latest(
    case: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    conversation = case.get("conversation")

    if not isinstance(conversation, list) or not conversation:
        return [], ""

    normalized = [
        {
            "role": str(turn.get("role") or "user"),
            "content": str(turn.get("content") or ""),
        }
        for turn in conversation
        if isinstance(turn, dict)
    ]

    latest_user_index: Optional[int] = None

    for index in range(len(normalized) - 1, -1, -1):
        if normalized[index]["role"] == "user":
            latest_user_index = index
            break

    if latest_user_index is None:
        return normalized, ""

    latest = normalized[latest_user_index]["content"]

    # Production Kelsie includes the newest user turn in the recent-message
    # window, so the eval does the same.
    return normalized, latest


def open_loop_action_observed(
    decision: Dict[str, Any],
) -> bool:
    candidate = decision.get("open_loop")

    if not isinstance(candidate, dict):
        return False

    try:
        confidence = float(candidate.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return bool(candidate.get("action")) and (
        confidence >= kelsie.OPEN_LOOP_MIN_CONFIDENCE
    )


def completion_observed(
    decision: Dict[str, Any],
) -> bool:
    try:
        loop_id = int(decision.get("complete_open_loop_id"))
    except (TypeError, ValueError):
        return False

    return (
        loop_id > 0
        and kelsie.decision_confidence(
            decision.get("reference_confidence")
        ) >= kelsie.REFERENCE_ACTION_MIN_CONFIDENCE
    )


def dismissal_observed(
    decision: Dict[str, Any],
) -> bool:
    try:
        loop_id = int(decision.get("dismiss_open_loop_id"))
    except (TypeError, ValueError):
        return False

    return (
        loop_id > 0
        and kelsie.decision_confidence(
            decision.get("reference_confidence")
        ) >= kelsie.REFERENCE_ACTION_MIN_CONFIDENCE
    )


def is_rate_limit_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in RATE_LIMIT_MARKERS)


async def call_and_capture(awaitable) -> Tuple[Any, str]:
    """Call a backend coroutine while capturing messages it prints.

    Kelsie's production backend intentionally swallows provider errors and
    returns fallbacks. For an eval, we need to know when that happened so a
    fallback is never accidentally counted as a successful AI response.
    """
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        result = await awaitable

    return result, buffer.getvalue()


def build_observed(
    decision: Dict[str, Any],
    reminder_parse: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    reminder_confirmation = bool(
        isinstance(reminder_parse, dict)
        and reminder_parse.get("type") == "reminder_confirmation"
    )
    reminder_needs_clarification = bool(
        isinstance(reminder_parse, dict)
        and reminder_parse.get("type") == "assistant"
    )

    return {
        "ask_follow_up": bool(
            decision.get("ask_follow_up", False)
            or decision.get("needs_clarification", False)
            or reminder_needs_clarification
        ),
        "create_reminder": reminder_confirmation,
        "create_open_loop": open_loop_action_observed(decision),
        "complete_open_loop": completion_observed(decision),
        "dismiss_open_loop": dismissal_observed(decision),
        "surface_other_items": bool(
            decision.get("should_reference_other_items", False)
        ),
    }


async def run_case(
    case: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    profile = normalize_profile(case)
    recent_messages, latest_user_message = conversation_without_latest(case)
    reminder_state = make_reminder_state(case)
    open_loop_state = make_open_loop_state(case)
    memory_record = make_memory_record(case)

    selected_memory = kelsie.select_relevant_memory(
        memory_record,
        latest_user_message,
    )

    personal_context = kelsie.build_personal_context(
        profile=profile,
        reminder_state=reminder_state,
        open_loop_state=open_loop_state,
        memory_record=memory_record,
        current_summary="",
        past_summaries=[],
        latest_user_message=latest_user_message,
        recent_messages=recent_messages,
    )

    decision, decision_log = await call_and_capture(
        kelsie.interpret_chat_turn(
            profile=profile,
            recent_messages=recent_messages,
            reminder_state=reminder_state,
            open_loop_state=open_loop_state,
            latest_user_message=latest_user_message,
            personal_context=personal_context,
            previous_summary="",
        )
    )

    if is_rate_limit_text(decision_log):
        return {
            "id": case.get("id"),
            "category": case.get("category"),
            "status": "rate_limited",
            "rate_limited_layer": "decision",
            "provider_log": decision_log.strip(),
            "expected": case.get("expected", {}),
        }

    if decision is None:
        return {
            "id": case.get("id"),
            "category": case.get("category"),
            "status": "error",
            "error": "Decision layer returned no result.",
            "provider_log": decision_log.strip(),
            "expected": case.get("expected", {}),
        }

    reminder_parse: Optional[Dict[str, Any]] = None

    if bool(decision.get("reminder_requested")):
        reminder_parse, reminder_log = await call_and_capture(
            kelsie.parse_chat_reminder_request(
                profile=profile,
                recent_messages=recent_messages,
                latest_user_message=latest_user_message,
                personal_context=personal_context,
            )
        )

        if is_rate_limit_text(reminder_log):
            return {
                "id": case.get("id"),
                "category": case.get("category"),
                "status": "rate_limited",
                "rate_limited_layer": "reminder_parser",
                "provider_log": reminder_log.strip(),
                "expected": case.get("expected", {}),
                "decision": decision,
            }

    observed = build_observed(decision, reminder_parse)
    observed["selected_memory"] = selected_memory

    expected = case.get("expected", {})
    checks = collect_auto_checks(expected, observed)
    auto_summary = summarize_checks(checks)

    # Decision mode stops here. It intentionally skips the expensive visible
    # response model call.
    if mode == "decision":
        return {
            "id": case.get("id"),
            "category": case.get("category"),
            "status": "completed",
            "mode": mode,
            "conversation": case.get("conversation", []),
            "expected": expected,
            "observed": observed,
            "decision": decision,
            "reminder_parse": reminder_parse,
            "reply": None,
            "auto_checks": checks,
            "auto_summary": auto_summary,
            "manual_review": manual_review_fields(expected),
            "notes": case.get("notes", ""),
        }

    fake_state = {
        "open_loop_created": observed["create_open_loop"],
        "open_loop_already_existed": False,
        "open_loop_completed": observed["complete_open_loop"],
        "open_loop_dismissed": observed["dismiss_open_loop"],
        "pending_reminder_cancelled": bool(
            decision.get("cancel_pending_reminder", False)
        ),
        "topic_closed": bool(decision.get("close_topic", False)),
        "memory_saved": False,
        "memory_items_saved": [],
        "personalization_policy": kelsie.get_personalization_policy(profile),
    }

    if reminder_parse is not None:
        reply = str(reminder_parse.get("message") or "")
        response_log = ""
    else:
        reply, response_log = await call_and_capture(
            kelsie.generate_grounded_chat_reply(
                profile=profile,
                recent_messages=recent_messages,
                latest_user_message=latest_user_message,
                decision=decision,
                state_result=fake_state,
                reminder_state=reminder_state,
                open_loop_state=open_loop_state,
                personal_context=personal_context,
            )
        )

        if is_rate_limit_text(response_log):
            return {
                "id": case.get("id"),
                "category": case.get("category"),
                "status": "rate_limited",
                "rate_limited_layer": "response",
                "provider_log": response_log.strip(),
                "expected": expected,
                "observed": observed,
                "decision": decision,
                "auto_checks": checks,
                "auto_summary": auto_summary,
                "reply": None,
            }

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "status": "completed",
        "mode": mode,
        "conversation": case.get("conversation", []),
        "expected": expected,
        "observed": observed,
        "decision": decision,
        "reminder_parse": reminder_parse,
        "reply": reply,
        "auto_checks": checks,
        "auto_summary": auto_summary,
        "manual_review": manual_review_fields(expected),
        "notes": case.get("notes", ""),
    }


def print_case_result(result: Dict[str, Any]) -> None:
    case_id = result.get("id", "?")
    category = result.get("category", "?")
    status = result.get("status")

    if status == "rate_limited":
        layer = result.get("rate_limited_layer", "provider")
        print(
            f"[RATE LIMITED] {case_id} — {category} "
            f"({layer} layer)"
        )
        print("       Case excluded from baseline scoring.")
        return

    if status == "error":
        print(
            f"[ERROR] {case_id} — {category}: "
            f"{result.get('error', 'Unknown error')}"
        )
        return

    summary = result["auto_summary"]
    total = summary["total"]
    passed = summary["passed"]
    percent = summary["percent"]

    if total:
        label = "PASS" if summary["failed"] == 0 else "CHECK"
        print(
            f"[{label}] {case_id:<4} {category:<28} "
            f"{passed}/{total} auto checks"
            + (
                f" ({percent}%)"
                if percent is not None
                else ""
            )
        )
    else:
        print(
            f"[REVIEW] {case_id:<4} {category:<28} "
            "no deterministic checks"
        )

    for check in result.get("auto_checks", []):
        if not check["passed"]:
            critical = " CRITICAL" if check["critical"] else ""
            print(
                f"       -{critical} {check['key']}: "
                f"expected={check['expected']} "
                f"observed={check['observed']}"
            )

    if result.get("mode") == "full":
        print(f"       Kelsie: {result.get('reply', '')}")
    else:
        intent = result.get("decision", {}).get("intent")
        reply_mode = result.get("decision", {}).get("reply_mode")
        print(
            f"       Decision: intent={intent!r}, "
            f"reply_mode={reply_mode!r}"
        )


def summarize_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_category: Dict[str, Dict[str, int]] = {}
    total_checks = 0
    passed_checks = 0
    critical_failures = 0
    errors = 0
    rate_limited = 0
    completed = 0

    for result in results:
        status = result.get("status")

        if status == "rate_limited":
            rate_limited += 1
            continue

        if status == "error":
            errors += 1
            continue

        completed += 1

        category = str(result.get("category") or "unknown")
        category_summary = by_category.setdefault(
            category,
            {
                "checks": 0,
                "passed": 0,
                "failed": 0,
            },
        )

        auto = result.get("auto_summary", {})
        category_summary["checks"] += int(auto.get("total") or 0)
        category_summary["passed"] += int(auto.get("passed") or 0)
        category_summary["failed"] += int(auto.get("failed") or 0)

        total_checks += int(auto.get("total") or 0)
        passed_checks += int(auto.get("passed") or 0)
        critical_failures += len(
            auto.get("critical_failures") or []
        )

    return {
        "attempted_cases": len(results),
        "completed_cases": completed,
        "errors": errors,
        "rate_limited_cases": rate_limited,
        "auto_checks": total_checks,
        "auto_passed": passed_checks,
        "auto_failed": total_checks - passed_checks,
        "auto_percent": (
            round((passed_checks / total_checks) * 100, 1)
            if total_checks
            else None
        ),
        "critical_failures": critical_failures,
        "by_category": by_category,
        "manual_review_pending": len(
            [
                result
                for result in results
                if result.get("status") == "completed"
            ]
        ),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Kelsie's intelligence evaluation suite."
    )

    parser.add_argument(
        "--cases",
        default=str(
            Path(__file__).with_name(
                "kelsie_eval_cases_v1.jsonl"
            )
        ),
        help="Path to JSONL eval cases.",
    )

    parser.add_argument(
        "--case",
        action="append",
        help="Run one case ID. Repeat to run several IDs.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected cases.",
    )

    parser.add_argument(
        "--mode",
        choices=("decision", "full"),
        default="decision",
        help=(
            "decision = cheaper semantic/controller test; "
            "full = also generate Kelsie's visible reply."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional result JSON path.",
    )

    args = parser.parse_args()

    if kelsie.client is None or not kelsie.AI_MODEL:
        raise SystemExit(
            "\nKelsie's AI client is not configured.\n"
            "Make sure your project-root .env contains your provider key "
            "and model, then run this command from the Kelsie-AI root.\n"
        )

    cases = load_cases(Path(args.cases))

    if args.case:
        selected_ids = {value.strip() for value in args.case}
        cases = [
            case
            for case in cases
            if str(case.get("id")) in selected_ids
        ]

    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    if not cases:
        raise SystemExit("No evaluation cases matched your selection.")

    print()
    print("KELSIE INTELLIGENCE EVAL")
    print(f"Mode: {args.mode}")
    print(f"Model: {kelsie.AI_MODEL}")
    print(f"Provider: {kelsie.AI_PROVIDER}")
    print(f"Cases: {len(cases)}")
    print(
        "Safe mode: the runner does not call get_kelsie_response(), "
        "so it does not mutate your live reminders/open loops."
    )

    if args.mode == "decision":
        print(
            "Decision mode skips the visible-response model call "
            "to reduce token usage."
        )
    else:
        print(
            "Full mode also generates Kelsie's visible reply and "
            "uses more tokens."
        )

    print()

    results: List[Dict[str, Any]] = []
    stopped_for_rate_limit = False

    for index, case in enumerate(cases, start=1):
        print(
            f"Running {index}/{len(cases)}: "
            f"{case.get('id')}...",
            end="\r",
            flush=True,
        )

        result = await run_case(case, args.mode)
        results.append(result)

        print(" " * 80, end="\r")
        print_case_result(result)

        if result.get("status") == "rate_limited":
            stopped_for_rate_limit = True
            print()
            print(
                "Stopping here so rate-limit fallbacks are not "
                "mistaken for real Kelsie results."
            )
            break

    summary = summarize_results(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path(__file__).resolve().parent
            / "results"
            / f"baseline_{args.mode}_{timestamp}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "model": kelsie.AI_MODEL,
        "provider": kelsie.AI_PROVIDER,
        "stopped_for_rate_limit": stopped_for_rate_limit,
        "summary": summary,
        "results": results,
    }

    output_path.write_text(
        json.dumps(
            output_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("BASELINE SUMMARY")
    print(f"Attempted: {summary['attempted_cases']}")
    print(f"Completed: {summary['completed_cases']}")
    print(f"Rate-limited: {summary['rate_limited_cases']}")
    print(f"Errors: {summary['errors']}")

    print(
        "Deterministic checks: "
        f"{summary['auto_passed']}/{summary['auto_checks']}"
        + (
            f" ({summary['auto_percent']}%)"
            if summary["auto_percent"] is not None
            else ""
        )
    )

    print(
        f"Critical state failures: "
        f"{summary['critical_failures']}"
    )

    if stopped_for_rate_limit:
        print()
        print(
            "This is a PARTIAL run, not a complete baseline. "
            "Rate-limited cases were excluded from scoring."
        )

    if args.mode == "decision":
        print()
        print(
            "Decision mode evaluates controller/state behavior only. "
            "Use --mode full on selected cases to review the actual "
            "visible answer."
        )

    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())