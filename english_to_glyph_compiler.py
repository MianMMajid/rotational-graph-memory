#!/usr/bin/env python3
"""Compile free-form English prompts into a strict topological 3-character sequence.

The output sequence uses the hidden Topological Character Language:
  X/Y -> object target
  M/D -> action (move/delete)
  A/B/C/D/E -> destination slot (0-4)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ParsedIntent:
    target_label: str
    target_code: str
    action_label: str
    action_code: str
    dest_label: str
    dest_code: str


class PromptCompiler:
    """Lightweight semantic lexer/parser for topological commands."""

    def __init__(self) -> None:
        self.target_map: Dict[str, str] = {
            "red cube": "X",
            "red block": "X",
            "blue sphere": "Y",
            "blue ball": "Y",
            "blue orb": "Y",
        }
        self.target_label = {
            "X": "Red Cube",
            "Y": "Blue Sphere",
        }

        self.action_map: Dict[str, str] = {
            "move": "M",
            "drop": "M",
            "place": "M",
            "put": "M",
            "shift": "M",
            "relocate": "M",
            "delete": "D",
            "remove": "D",
            "destroy": "D",
            "erase": "D",
            "clear": "D",
        }
        self.action_label = {
            "M": "Move",
            "D": "Delete",
        }

        self.destination_map: Dict[str, str] = {
            "A": "0",
            "B": "1",
            "C": "2",
            "D": "3",
            "E": "4",
        }
        self.position_map: Dict[str, str] = {
            "0": "A",
            "1": "B",
            "2": "C",
            "3": "D",
            "4": "E",
            "zero": "A",
            "one": "B",
            "two": "C",
            "three": "D",
            "four": "E",
            "1st": "B",
            "first": "B",
            "2nd": "C",
            "second": "C",
            "3rd": "D",
            "third": "D",
            "4th": "E",
            "fourth": "E",
        }
        self.ignore_words = {
            "please",
            "can",
            "could",
            "would",
            "you",
            "kindly",
            "hey",
            "hi",
            "hello",
            "the",
            "a",
            "an",
            "to",
            "of",
            "my",
            "me",
            "is",
            "are",
            "can you",
        }

    def parse_prompt(self, english_string: str) -> ParsedIntent:
        if not isinstance(english_string, str):
            raise ValueError("Input must be a text prompt.")

        normalized = self._normalize(english_string)
        target_code, target_label = self._extract_target(normalized)
        action_code, action_label = self._extract_action(normalized)
        dest_code, dest_num = self._extract_destination(normalized)

        return ParsedIntent(
            target_label=target_label,
            target_code=target_code,
            action_label=action_label,
            action_code=action_code,
            dest_label=f"Pos {dest_num}",
            dest_code=dest_code,
        )

    def _normalize(self, text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        tokens = [tok for tok in lowered.split() if tok not in self.ignore_words]
        return " ".join(tokens)

    def _extract_target(self, normalized: str) -> Tuple[str, str]:
        # Longest-first phrase match gives "red cube" priority over partial overlaps.
        hits = []
        for phrase, code in self.target_map.items():
            pattern = re.compile(rf"\b{re.escape(phrase)}\b")
            m = pattern.search(normalized)
            if m:
                hits.append((m.start(), code, phrase))

        if not hits:
            raise ValueError(
                "Could not determine target. Try explicit phrasing like 'red cube' or 'blue sphere'."
            )

        _, code, phrase = min(hits, key=lambda item: item[0])
        return code, self.target_label[code]

    def _extract_action(self, normalized: str) -> Tuple[str, str]:
        hits = []
        for phrase, code in self.action_map.items():
            pattern = re.compile(rf"\b{re.escape(phrase)}\b")
            for match in pattern.finditer(normalized):
                hits.append((match.start(), code, phrase))

        if not hits:
            raise ValueError(
                "Could not determine action. Use move/drop/place or delete/remove/destroy terms."
            )

        _, code, _ = min(hits, key=lambda item: item[0])
        return code, self.action_label[code]

    def _extract_destination(self, normalized: str) -> Tuple[str, str]:
        # 1) explicit destination keywords
        patterns = [
            r"\b(?:position|slot|index|cell|idx)\s*(?:number\s*)?(?P<p>[a-z0-9]+)\b",
            r"\b(?:to|at|into|in)\s*(?:position|slot|index|cell|idx)?\s*(?P<p>[a-z0-9]+)\b",
            r"\bpos\s*(?P<p>[a-z0-9]+)\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, normalized):
                raw = match.group("p")
                if not raw:
                    continue
                token = raw.strip().strip("#")
                if token in self.position_map:
                    code = self.position_map[token]
                    return code, self.destination_map[code]
                if token in {"0", "1", "2", "3", "4", "zero", "one", "two", "three", "four"}:
                    code = self.position_map[token]
                    return code, self.destination_map[code]

        # 2) short fallback: detect any trailing isolated index token
        tokens = normalized.split()
        for i, token in enumerate(tokens):
            if token in {"position", "pos", "slot", "index", "cell"} and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt in self.position_map:
                    code = self.position_map[nxt]
                    return code, self.destination_map[code]

        raise ValueError(
            "Could not determine destination slot. Use phrases like 'position 2' or 'slot 4'."
        )

    def generate_glyph_array(self, parsed: ParsedIntent) -> str:
        sequence = (
            parsed.target_code
            + parsed.action_code
            + parsed.dest_code
        )
        if len(set(sequence)) != 3:
            raise SyntaxError("Ambiguous English Prompt")
        return sequence


def main() -> None:
    compiler = PromptCompiler()
    print("Topological Character Compiler ready. Type 'exit' to quit.\n")

    while True:
        raw = input("USER PROMPT > ").strip()
        if not raw:
            continue

        lowered = raw.lower()
        if lowered in {"exit", "quit", "q"}:
            print("Shutting down compiler.")
            break

        try:
            parsed = compiler.parse_prompt(raw)
        except Exception as exc:
            print(f"[Lexer] Failed to parse input: {exc}")
            continue

        print(
            f"[Lexer] Extracted: Target={parsed.target_label}, "
            f"Action={parsed.action_label}, Dest={parsed.dest_label}"
        )
        print("[Compiler] Assembling Glyph Array...")

        try:
            sequence = compiler.generate_glyph_array(parsed)
        except SyntaxError as exc:
            print(f"[Compiler] {exc}")
            continue

        array_tokens = ", ".join(sequence)
        print(f"[Handoff] Sending array [{array_tokens}] to AI Agent...")
        print(f"[Output] Compiled sequence: {sequence}\n")


if __name__ == "__main__":
    main()
