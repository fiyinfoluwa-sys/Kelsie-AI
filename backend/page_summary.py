from __future__ import annotations

from typing import Any, Callable, Dict, Optional

try:
    from .reading_routes import (
        register_page_summary_routes as register_reading_routes,
    )
    from .context_engine import register_context_routes
except ImportError:
    from reading_routes import (  # type: ignore[no-redef]
        register_page_summary_routes as register_reading_routes,
    )
    from context_engine import register_context_routes  # type: ignore[no-redef]


def register_page_summary_routes(
    app: Any,
    client: Any,
    ai_model: Optional[str],
    ai_provider: Optional[str],
    extract_json_object: Callable[
        [str],
        Optional[Dict[str, Any]],
    ],
) -> None:
    """
    Compatibility bootstrap used by backend.main.

    Reading routes stay in reading_routes.py exactly as they worked before.
    Context interpretation/matching is registered beside them without changing
    backend.main, database.py, reminders, open loops, memory, or chat.
    """

    register_reading_routes(
        app=app,
        client=client,
        ai_model=ai_model,
        ai_provider=ai_provider,
        extract_json_object=extract_json_object,
    )

    register_context_routes(
        app=app,
        client=client,
        ai_model=ai_model,
        ai_provider=ai_provider,
        extract_json_object=extract_json_object,
    )