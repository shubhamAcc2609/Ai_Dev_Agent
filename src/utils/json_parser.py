import json
import re
from typing import Any, Dict


def extract_json_object(response_text: str) -> Dict[str, Any]:
    """
    Extracts the first valid JSON object from an LLM response.

    Local reasoning models may wrap JSON in markdown, prose, or <think> tags.
    This parser scans for the first decodeable JSON object instead of assuming
    the first and last braces in the response belong to the same object.
    """
    text = _strip_reasoning_blocks(response_text or "").strip()
    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No valid JSON object found in response: {response_text}")


def _strip_reasoning_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
