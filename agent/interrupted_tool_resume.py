"""Load-time stub for tool calls left unanswered when a session died mid-tool (#99869)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("hermes_state")

INTERRUPTED_TOOL_CALL_RESULT = (
    "[No result recorded — the session was interrupted while this tool call "
    "was in flight (process crash, kill, or timeout). Its side effects may have "
    "been applied fully, partially, or not at all; verify the current state "
    "before relying on it.]"
)


def _answer_interrupted_tool_calls(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Give the transcript's trailing unanswered tool calls an explicit result.

    The agent loop persists an assistant ``tool_calls`` turn BEFORE the tools
    run. When the process then dies mid-tool, the durable transcript ends with
    that turn and no result rows — and ``repair_message_sequence`` Pass 2 would
    prune the unanswered calls, so the resumed model never learns a write was
    in flight (#99869).

    Only the TRAILING run is touched. Synthetic results are marked durable so a
    later flush does not write them as real rows; a repeat resume regenerates them.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY
    from agent.message_sanitization import coalesce_tool_call_id

    answered: set[str] = set()
    idx = len(messages) - 1
    while idx >= 0 and messages[idx].get("role") == "tool":
        answered.add(str(messages[idx].get("tool_call_id") or ""))
        idx -= 1
    if idx < 0 or messages[idx].get("role") != "assistant":
        return messages
    synthetic = []
    skipped_idless = 0
    for tc in messages[idx].get("tool_calls") or []:
        call_id = coalesce_tool_call_id(tc)
        if not call_id:
            skipped_idless += 1
            continue
        if call_id in answered:
            continue
        fn = tc.get("function") or {}
        synthetic.append({
            "role": "tool",
            "content": INTERRUPTED_TOOL_CALL_RESULT,
            "tool_call_id": call_id,
            "tool_name": fn.get("name", "") if isinstance(fn, dict) else "",
            _DB_PERSISTED_MARKER_KEY: True,
        })
    if synthetic:
        logger.info(
            "Answered %d interrupted tool call(s) while restoring session (#99869)",
            len(synthetic),
        )
    if skipped_idless:
        logger.debug(
            "Skipped %d trailing tool call(s) with no pairing id while restoring session",
            skipped_idless,
        )
    return messages + synthetic
