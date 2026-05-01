import logging
import time
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler

from app.config import settings


def setup_logging() -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog so asyncpg/httpx/etc. land in same sink
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler()],
        level=getattr(logging, settings.log_level.upper()),
    )


class ToolCallLogger(BaseCallbackHandler):
    """LangChain callback that logs every MCP tool call with duration and errors."""

    def __init__(self, component: str) -> None:
        self._log = structlog.get_logger().bind(component=component)
        self._start_times: dict[str, float] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        self._start_times[str(run_id)] = time.monotonic()
        self._log.info("tool_call_start", tool=tool_name, input=input_str)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        duration = time.monotonic() - self._start_times.pop(str(run_id), time.monotonic())
        output_str = output.content if hasattr(output, "content") else str(output)
        self._log.info("tool_call_end", duration_ms=round(duration * 1000), output=output_str[:200])

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        duration = time.monotonic() - self._start_times.pop(str(run_id), time.monotonic())
        self._log.error("tool_call_error", duration_ms=round(duration * 1000), error=str(error))
