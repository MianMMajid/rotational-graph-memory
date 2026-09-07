#!/usr/bin/env python3
"""
semantic_grounding_loop.py
A Python-only semantic grounding sandbox with an LLM-in-the-loop translator.
"""

from __future__ import annotations

import os
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI, OpenAIError


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_RETRIES = 5
DATA_DIR = Path("data")
TRAIN_DATA_PATH = DATA_DIR / "train.jsonl"
VALID_DATA_PATH = DATA_DIR / "valid.jsonl"
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto").strip().lower()
LOCAL_MLX_MODEL = os.getenv("LOCAL_MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit")
LOCAL_MLX_MAX_TOKENS = os.getenv("LOCAL_MLX_MAX_TOKENS", "64")

SYSTEM_PROMPT = (
    "You are a Semantic Parsing Engine. You receive an English command. Your ONLY job is "
    "to translate it into a valid Python function call using the available API: "
    "`move_object(target_color, target_shape, new_position)`. Output ONLY the raw Python "
    "string (e.g., `move_object('red', 'cube', 2)`). No markdown, no explanations. "
    "Important: only `cube` and `sphere` are valid shapes. Interpret `block` as `cube`."
)


class Sandbox:
    """1D sandbox with hard physical constraints."""

    def __init__(self) -> None:
        self.positions: list[Optional[Dict[str, str]]] = [None] * 5
        self.positions[0] = {"color": "red", "shape": "cube"}
        self.positions[4] = {"color": "blue", "shape": "sphere"}

    def move_object(self, target_color: str, target_shape: str, new_pos: int) -> None:
        if not 0 <= new_pos <= 4:
            raise ValueError("SandboxError: Position out of bounds.")

        source_pos = self._find_object_position(target_color, target_shape)
        if source_pos is None:
            raise ValueError("SandboxError: Target object does not exist.")

        if self.positions[new_pos] is not None:
            raise ValueError("SandboxError: Collision detected. Position is occupied.")

        self.positions[new_pos] = self.positions[source_pos]
        self.positions[source_pos] = None

    def _find_object_position(self, target_color: str, target_shape: str) -> Optional[int]:
        for idx, obj in enumerate(self.positions):
            if obj is not None and obj["color"] == target_color and obj["shape"] == target_shape:
                return idx
        return None

    def snapshot(self) -> str:
        rendered = []
        for i, obj in enumerate(self.positions):
            if obj is None:
                rendered.append(f"{i}: empty")
            else:
                rendered.append(f"{i}: {obj['color']} {obj['shape']}")
        return "\n".join(rendered)


def _append_training_record(command: str, action_call: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample = {"text": f"User: {command}\\nAssistant: {action_call}"}
    line = json.dumps(sample, ensure_ascii=False)
    with TRAIN_DATA_PATH.open("a", encoding="utf-8") as train_file:
        train_file.write(line + "\n")
    with VALID_DATA_PATH.open("a", encoding="utf-8") as valid_file:
        valid_file.write(line + "\n")


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def _extract_action(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text

    # If the model wraps with markdown fences, keep the last fenced block.
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        text = blocks[-1]

    text = text.replace("```python", "").replace("```", "").strip()

    # Keep only the first non-empty line to avoid multi-line drift.
    first_line = text.splitlines()[0].strip()
    return first_line


def _translate_with_local_mlx(prompt: str) -> str:
    cmd = [
        "python3",
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        LOCAL_MLX_MODEL,
        "--system-prompt",
        SYSTEM_PROMPT,
        "--prompt",
        prompt,
        "--max-tokens",
        LOCAL_MLX_MAX_TOKENS,
        "--temp",
        "0",
        "--verbose",
        "F",
        "--ignore-chat-template",
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Local MLX translation failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return _extract_action(completed.stdout or "")


def translate_to_action(english_command: str, error_feedback: str = None) -> str:
    user_prompt = (
        "Translate this command into a raw Python API call: "
        f"\"{english_command}\""
    )

    if error_feedback:
        user_prompt += (
            f"\nYour previous attempt failed with this runtime error: {error_feedback}. "
            "Adjust your translation and output the corrected function call. "
            "If this was a collision, keep the same target and move it to a different valid empty position."
        )

    if LLM_BACKEND in {"mlx", "mlx-lm", "local", "local-mlx"}:
        return _translate_with_local_mlx(user_prompt)

    if LLM_BACKEND == "auto":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return _extract_action(response.choices[0].message.content or "")

        return _translate_with_local_mlx(user_prompt)

    # Explicit OpenAI mode
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return _extract_action(response.choices[0].message.content or "")


def execute_loop(sandbox: Sandbox, english_command: str) -> Optional[str]:
    error_feedback = None

    for attempt in range(1, MAX_RETRIES + 1):
        _log(f"Command: {english_command} | attempt {attempt}/{MAX_RETRIES}")

        try:
            contextual_command = (
                f"{english_command}\nCurrent sandbox state:\n{sandbox.snapshot()}"
            )
            action_code = translate_to_action(contextual_command, error_feedback)
        except (OpenAIError, RuntimeError) as exc:
            _log(f"LLM translation failed: {exc}")
            return None

        _log(f"LLM output: {action_code}")

        try:
            # Restricted eval with only the move_object API exposed.
            eval(action_code, {"__builtins__": {}}, {"move_object": sandbox.move_object})
            _log("Execution succeeded. Updated sandbox state:")
            print(sandbox.snapshot())
            return action_code

        except Exception as exc:  # pragma: no cover
            _log(f"Execution failed: {exc}")
            print(sandbox.snapshot())
            error_feedback = str(exc)

    _log(f"Maximum retries exceeded for command: {english_command}")
    return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox()

    test_commands = [
        "Move the red block to position 2.",
        "Place the blue sphere where the red cube is.",
    ]

    _log("Initial sandbox state:")
    print(sandbox.snapshot())

    for command in test_commands:
        generated_action = execute_loop(sandbox, command)
        if generated_action is None:
            return

        _append_training_record(command, generated_action)

    _log("All test commands completed.")


if __name__ == "__main__":
    main()
