#!/usr/bin/env python3
"""
neuro_symbolic_loop.py

Autonomous LLM <-> Lean 4 feedback loop for proof synthesis.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict
from shutil import which
from datetime import datetime

from openai import OpenAI, OpenAIError


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt5.3codex spark high")
MAX_RETRIES = 5
TEMP_LEAN_FILE = Path("temp_proof.lean")
OUTPUT_DIR = Path("verified_patterns")

SYSTEM_PROMPT = (
    "You are a strict autoformalization engine. You translate natural language math and "
    "logic into formal Lean 4 theorem code. You output ONLY valid, compilable Lean 4 code. "
    "No markdown, no explanations. Use the `Mathlib` library if necessary. Do not use `sorry` "
    "in your proofs."
)


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def _extract_lean_code(raw: str) -> str:
    """
    Clean LLM output to plain Lean code if fenced markdown appears.
    """
    if not raw:
        return ""

    code_blocks = re.findall(r"```(?:lean)?\s*\n(.*?)```", raw, flags=re.DOTALL)
    if code_blocks:
        raw = code_blocks[-1]  # final fenced block is most likely the full answer

    # remove any accidental surrounding fences
    raw = raw.replace("```lean", "").replace("```", "")
    return raw.strip()


def generate_lean_proof(problem_statement: str, error_feedback: str = None) -> str:
    """
    Ask the LLM to generate (or fix) Lean 4 code for the given math statement.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    user_prompt = (
        "Translate this natural-language math logic statement into Lean 4 code.\n\n"
        f"Problem: {problem_statement}"
    )
    user_prompt += (
        "\n\nEnvironment constraint: this is a standalone Lean file run with core Lean only (no "
        "Mathlib environment is guaranteed). Do not use `import Mathlib`.\n"
        "If needed, define simple local helper predicates (for example, define `Even` explicitly)."
    )
    if error_feedback:
        user_prompt += (
            "\n\n"
            "Your previous attempt failed with this compiler error. Fix the logical flaw and "
            "output the corrected Lean 4 code.\n"
            f"{error_feedback}"
        )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content or ""
    return _extract_lean_code(raw)


def verify_with_lean(lean_code: str) -> Dict[str, object]:
    """
    Write Lean code to a temp file, run `lean temp_proof.lean`,
    and return compile outcome + outputs.
    """
    TEMP_LEAN_FILE.write_text(lean_code, encoding="utf-8")

    try:
        completed = subprocess.run(
            ["lean", str(TEMP_LEAN_FILE)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if TEMP_LEAN_FILE.exists():
            TEMP_LEAN_FILE.unlink()

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    trace = (stdout + ("\n" if stdout and stderr else "") + stderr).strip()
    status = "Success" if completed.returncode == 0 else "Failed"

    return {
        "status": status,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "error_trace": trace,
    }


def main() -> None:
    if which("lean") is None:
        raise RuntimeError(
            "Lean CLI not found. Ensure elan is installed and `lean` is on PATH."
        )

    test_problem = (
        "Prove that for any two integers p and q, if p is even and q is even, "
        "then their sum (p + q) is even."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    error_feedback = None
    last_code = ""

    for attempt in range(1, MAX_RETRIES + 1):
        _log(f"Attempt {attempt}/{MAX_RETRIES} — generating Lean code")
        try:
            lean_code = generate_lean_proof(test_problem, error_feedback)
        except (OpenAIError, RuntimeError) as exc:
            _log(f"LLM request failed: {exc}")
            return

        if not lean_code.strip():
            _log("LLM returned empty output; retrying.")
            error_feedback = "Model returned empty output. Please output full Lean 4 code."
            continue

        last_code = lean_code
        _log("Generated Lean code:\n" + lean_code)

        result = verify_with_lean(lean_code)

        if result["status"] == "Success":
            success_file = (
                OUTPUT_DIR
                / f"verified_even_sum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.lean"
            )
            success_file.write_text(lean_code + "\n", encoding="utf-8")

            print("\n" + "=" * 100)
            print("🎉 MATH-PROOF VERIFICATION PASSED 🎉")
            print("Lean accepted the generated theorem. This run is a verified success.")
            print(f"Saved proof to: {success_file}")
            print("=" * 100 + "\n")
            print(lean_code)
            return

        _log(
            f"Verification failed (code {result['returncode']}). "
            "Sending compiler trace back to LLM."
        )
        print("\n--- Lean error trace ---")
        print(result["error_trace"])
        print("--- end trace ---\n")

        error_feedback = result["error_trace"]

    _log("Reached max retries without success.")
    failed_path = OUTPUT_DIR / "failed_last_attempt.lean"
    if last_code:
        failed_path.write_text(last_code + "\n", encoding="utf-8")
        _log(f"Saved last attempted proof to {failed_path}")


if __name__ == "__main__":
    main()
