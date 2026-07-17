"""User/operator feedback persistence for offline learning loops with crash protection."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class FeedbackLabel(str, Enum):
    """Supported feedback categories."""

    CORRECT = "correct"
    TOO_RISKY = "too_risky"
    TOO_CONSERVATIVE = "too_conservative"
    WRONG_DIRECTION = "wrong_direction"
    UNCLEAR_INSTRUCTION = "unclear_instruction"
    OTHER = "other"


@dataclass(frozen=True)
class FeedbackRecord:
    """Feedback entry associated with a scene timestamp."""

    scene_timestamp: str
    label: FeedbackLabel
    reason: str
    created_at: str


class FeedbackStore:
    """Append-only JSONL store for runtime navigation feedback. Safe from disk write errors."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.warning("Could not create feedback directory: %s", exc)

    def add(self, scene_timestamp: str, label: FeedbackLabel, reason: str) -> FeedbackRecord | None:
        """Create and persist a feedback record. Fails gracefully on I/O issues."""
        record = FeedbackRecord(
            scene_timestamp=scene_timestamp,
            label=label,
            reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                payload = asdict(record)
                payload["label"] = record.label.value
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            LOGGER.info("Feedback successfully logged: %s - %s", label.value, reason)
            return record
        except (OSError, IOError) as exc:
            LOGGER.warning("Failed to write operator feedback log to %s: %s", self._log_path, exc)
            return None
        except Exception as exc:
            LOGGER.warning("Unexpected error during feedback logging: %s", exc)
            return None
