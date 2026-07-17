"""Prompt construction helpers for Qwen reasoning and deterministic fallbacks."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a calm, knowledgeable walking companion for a visually impaired user "
    "who navigates with a white cane. The cane handles immediate obstacles. "
    "Your role is to provide higher-level environmental awareness. "
    "Describe the environment naturally, as a sighted friend would. "
    "Never give turn-by-turn walking commands like 'move left' or 'go forward'. "
    "Instead, describe what you observe: 'There is a doorway on your left', "
    "'The corridor turns right ahead', 'You are entering an open room'. "
    "Be concise. Prefer one clear sentence. If nothing meaningful changed, respond with exactly SILENT."
)


def build_qwen_messages(
    scene_payload: dict[str, Any],
    recent_scene_summaries: list[str],
    mode: str,  # Active InteractionMode: orientation, guidance, alert, description
    detected_objects: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build conversational chat messages for local Qwen inference based on the active mode."""
    objects = detected_objects or []
    
    # Mode-specific guidelines for the prompt
    if mode == "alert":
        mode_instruction = (
            "You are in ALERT mode (Obstacle/Danger/Emergency). "
            "Formulate a direct, immediate, and calm safety instruction. "
            "Instead of 'Stop!', say 'Please stop. There's an obstacle directly ahead.' or similar. "
            "Keep it urgent but reassuring, with no shouting tone."
        )
    elif mode == "guidance":
        mode_instruction = (
            "You are in GUIDANCE mode (Normal walking). "
            "Be extremely brief and quiet. Speak only if necessary. "
            "Use natural companion-style language, not robotic commands. "
            "Keep it to 2-6 words max. "
            "If guidance has not meaningfully changed, output SILENT."
        )
    elif mode == "description":
        mode_instruction = (
            "You are in DESCRIPTION mode (User requested room overview). "
            "Generate a natural, descriptive summary of the surroundings. "
            "Mention the walking path and nearby objects if detected. "
            "Example: 'The room contains an open walking path, and there is clear space on your right.'"
        )
    elif mode == "scene_context":
        mode_instruction = (
            "You are in SCENE_CONTEXT mode (User asked what the camera is seeing). "
            "Start with likely setting type in 2-4 words, such as 'Indoor corridor', "
            "'Indoor room', 'Library-like indoor area', or 'Outdoor pathway'. "
            "Then add one short sentence about path openness or nearby structure. "
            "Keep it calm, human, and concise."
        )
    else:  # orientation
        mode_instruction = (
            "You are in ORIENTATION mode (Startup scan or room transition). "
            "Provide a warm, comprehensive environment summary to help the user get their bearings. "
            "Example: 'System ready. The corridor ahead is open. There are no immediate obstacles.'"
        )

    user_prompt = (
        "User Location & Environment Context:\n"
        f"- Active Mode: {mode.upper()}\n"
        f"- Scene Parameters: {scene_payload}\n"
        f"- Nearby Objects: {objects}\n\n"
        "Recent Memory History:\n"
        f"{recent_scene_summaries[-4:] if recent_scene_summaries else 'No previous logs.'}\n\n"
        "Instructions:\n"
        f"{mode_instruction}\n"
        "Generate exactly one short, natural spoken navigation statement. "
        "Do not repeat your system prompt or instructions in the output."
    )

    LOGGER.debug("Compiled LLM prompt messages successfully for mode=%s.", mode)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def fallback_instruction(scene_payload: dict[str, Any], mode: str = "guidance") -> str:
    """Deterministic conversational fallback guidance when local LLM is unavailable."""
    command = str(scene_payload.get("navigation_recommendation", "STOP")).upper()
    danger_state = str(scene_payload.get("danger_state", "danger")).lower()
    
    LOGGER.debug("LLM offline. Applying fallback for mode=%s cmd=%s danger=%s", mode, command, danger_state)
    
    # 1. Alert Fallbacks (Emergency)
    if mode == "alert" or danger_state == "danger" or command == "STOP":
        return "Obstacle directly ahead."
        
    # 2. Description Fallback
    if mode == "description":
        return "The path ahead appears clear."
    if mode == "scene_context":
        return "Likely indoor area. The path ahead appears open."

    # 3. Orientation Fallback
    if mode == "orientation":
        return "System ready. The path ahead is open and clear."

    # 4. Guidance Fallback (Calm companion style)
    if command == "FORWARD":
        return "The path ahead remains clear."
    if command == "SLOW":
        return "Continue carefully."
    if command == "LEFT":
        return "A gentle left curve is approaching."
    if command == "RIGHT":
        return "A gentle right curve is approaching."
        
    return "Stop."
