"""Validation for untrusted chat requests and bounded conversation context."""

import math


MAX_PROMPT_CHARS = 10_000
MAX_MESSAGE_CHARS = 20_000
MAX_CONVERSATION_MESSAGES = 40
MAX_CONVERSATION_CHARS = 40_000


def validate_chat_payload(prompt, conversation):
    """Validate and normalize prompt/history values received from a client."""
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        raise ValueError("Invalid request.")
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS or _has_unsupported_controls(prompt):
        raise ValueError("Invalid request.")

    if conversation is None:
        conversation = []
    if not isinstance(conversation, list) or len(conversation) > MAX_CONVERSATION_MESSAGES:
        raise ValueError("Invalid request.")

    normalized_conversation = []
    total_chars = 0
    for item in conversation:
        if not isinstance(item, dict):
            raise ValueError("Invalid request.")
        role = item.get("role")
        content = item.get("content")
        if (
            not isinstance(role, str)
            or role not in {"user", "assistant"}
            or not isinstance(content, str)
            or len(content) > MAX_MESSAGE_CHARS
            or _has_unsupported_controls(content)
        ):
            raise ValueError("Invalid request.")
        total_chars += len(content)
        if total_chars > MAX_CONVERSATION_CHARS:
            raise ValueError("Invalid request.")

        normalized_item = {"role": role, "content": content}
        duration = item.get("duration_s")
        if duration is not None:
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
            ):
                raise ValueError("Invalid request.")
            if isinstance(duration, float) and not math.isfinite(duration):
                raise ValueError("Invalid request.")
            if not 0 <= duration <= 3600:
                raise ValueError("Invalid request.")
            normalized_item["duration_s"] = float(duration)
        created_at = item.get("created_at")
        if created_at is not None:
            if not isinstance(created_at, str) or len(created_at) > 64:
                raise ValueError("Invalid request.")
            normalized_item["created_at"] = created_at
        normalized_conversation.append(normalized_item)

    return prompt, normalized_conversation


def trim_conversation(conversation, max_chars=MAX_CONVERSATION_CHARS):
    """Keep the newest history entries within the request validation limits."""
    conversation = list(conversation)
    total_chars = sum(len(item["content"]) for item in conversation)
    while len(conversation) > MAX_CONVERSATION_MESSAGES or total_chars > max_chars:
        total_chars -= len(conversation[0]["content"])
        conversation.pop(0)
    return conversation


def _has_unsupported_controls(value):
    return any(ord(character) < 32 and character not in "\t\n\r" for character in value)
