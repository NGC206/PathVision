"""Conversation Memory to prevent repetitive guidance and handle periodic reassurance."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning.situation_manager import InteractionMode, SituationType

LOGGER = logging.getLogger(__name__)


class ConversationMemory:
    """Remembers spoken prompts and throttles voice guidance to avoid spamming the user."""

    def __init__(self, reassurance_interval: float = 12.0) -> None:
        self.last_spoken_sentence = ""
        self.last_situation: SituationType | None = None
        self.last_navigation_command = ""
        self.last_warning = ""
        self.last_speech_time = 0.0
        self.reassurance_interval = reassurance_interval
        LOGGER.info("ConversationMemory initialized | reassurance_interval=%.1fs", reassurance_interval)

    def should_speak(
        self,
        text: str,
        situation: SituationType,
        command: str,
        mode: InteractionMode,
        time_now: float,
    ) -> bool:
        """Evaluate if the instruction should be spoken or suppressed."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return False

        # Mode-specific routing
        is_alert = (mode.value == "alert")
        is_user_request = (mode.value in ("description", "orientation"))

        if is_user_request:
            LOGGER.debug("User-initiated request. Allowing speech.")
            return True

        if is_alert:
            # Emergency/warning preemption
            # If it's the exact same warning, throttle it to 8.0 seconds to prevent annoying loops
            if cleaned_text == self.last_spoken_sentence or cleaned_text == self.last_warning:
                if time_now - self.last_speech_time < 8.0:
                    return False
            self.last_warning = cleaned_text
            return True

        # Guidance or Orientation mode
        # Suppress identical repeating sentences
        if cleaned_text == self.last_spoken_sentence:
            # Even if it is identical, if the reassurance interval has elapsed, we can speak it as reassurance
            if time_now - self.last_speech_time >= self.reassurance_interval:
                LOGGER.debug("Reassurance interval elapsed. Allowing duplicate speech: '%s'", cleaned_text)
                return True
            return False

        # If the situation or command has changed, we want to speak the new instruction
        if situation != self.last_situation or command != self.last_navigation_command:
            LOGGER.debug("Situation/Command shifted. Allowing speech.")
            return True

        # If nothing changed, but the reassurance interval has elapsed, allow it
        if time_now - self.last_speech_time >= self.reassurance_interval:
            LOGGER.debug("Periodic timeout elapsed. Allowing reassurance.")
            return True

        # Otherwise, prefer silence
        return False

    def update(
        self,
        text: str,
        situation: SituationType,
        command: str,
        time_now: float,
    ) -> None:
        """Update memory state after successful speech playback."""
        self.last_spoken_sentence = text.strip()
        self.last_situation = situation
        self.last_navigation_command = command
        self.last_speech_time = time_now
        LOGGER.debug("ConversationMemory updated: '%s'", self.last_spoken_sentence)
