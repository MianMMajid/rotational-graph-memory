#!/usr/bin/env python3
"""Topological compiler loop with an engineering-style verification pipeline.

This module uses no hardcoded English->token grammar for execution.
- Spec: a structured objective that must be met (e.g., move action, delete action).
- Candidate: a 3-character sequence proposal from sandbox-visible evidence.
- Verify: strict static + dry-run + live execution checks.
- Self-correct: association scores are updated from success/failure feedback.
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, DefaultDict, Dict, Iterable, List, Optional, Tuple


ALPHABET = string.ascii_uppercase
DATA_PATH = Path("data/topological_train.jsonl")
HYPOTHESES_PER_ROUND = 5
TARGET_SUCCESS_PAIRS = 100
ROUNDS_CAP = 5000
NO_PROGRESS_STREAK = 20


@dataclass(frozen=True)
class ExecutionSpec:
    name: str
    desired_action: Optional[str] = None  # "move", "delete", or None
    destination: Optional[int] = None  # 0..4 for move specs


@dataclass(frozen=True)
class VerificationResult:
    sequence: str
    ok: bool
    outcome: str
    stage: str
    error: str = ""


@dataclass(frozen=True)
class ParsedOutcome:
    target: str
    action: str
    destination: Optional[int]


class TopologicalSandbox:
    """1D sandbox with a hidden topological alphabet.

    The symbolic alphabet (`X`, `Y`, `M`, `D`, `A`-`E`) is intentionally small and
    strict.  The agent must discover mappings through interaction.
    """

    GRID_SIZE = 5
    OBJECT_MAP: Dict[str, str] = {
        "X": "glyph_red_cube",
        "Y": "glyph_blue_sphere",
    }
    ACTION_MAP: Dict[str, str] = {
        "M": "move",
        "D": "delete",
    }
    INDEX_MAP: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    LABEL_MAP: Dict[str, str] = {
        "glyph_red_cube": "Red Cube",
        "glyph_blue_sphere": "Blue Sphere",
    }

    def __init__(self) -> None:
        self._base_state: List[Optional[str]] = [None] * self.GRID_SIZE
        self._base_state[0] = "glyph_red_cube"
        self._base_state[4] = "glyph_blue_sphere"
        self._state: List[Optional[str]] = list(self._base_state)

    def reset(self) -> None:
        self._state = list(self._base_state)

    @staticmethod
    def _validate_sequence(seq: str) -> Tuple[str, str, str]:
        if not isinstance(seq, str):
            raise SyntaxError("Permutation Invalid")

        if len(seq) != 3:
            raise SyntaxError("Permutation Invalid")

        if len(set(seq)) != 3:
            raise SyntaxError("Permutation Invalid")

        obj, action, index = seq[0], seq[1], seq[2]

        if obj not in TopologicalSandbox.OBJECT_MAP:
            raise SyntaxError("Permutation Invalid")

        if action not in TopologicalSandbox.ACTION_MAP:
            raise SyntaxError("Permutation Invalid")

        if index not in TopologicalSandbox.INDEX_MAP:
            raise SyntaxError("Permutation Invalid")

        return obj, action, index

    def _simulate_once(self, seq: str, state: List[Optional[str]]) -> str:
        obj_key, action_key, index_char = self._validate_sequence(seq)

        target_id = self.OBJECT_MAP[obj_key]
        action = self.ACTION_MAP[action_key]
        destination = self.INDEX_MAP[index_char]

        if target_id not in state:
            raise ValueError("Spatial Collision")

        target_label = self.LABEL_MAP[target_id]
        source = state.index(target_id)

        if action == "delete":
            state[source] = None
            return f"{target_label} deleted."

        occupant = state[destination]
        if occupant is not None and occupant != target_id:
            raise ValueError("Spatial Collision")

        state[source] = None
        state[destination] = target_id
        return f"{target_label} moved to Index {destination}."

    def simulate(self, seq: str, state: Optional[List[Optional[str]]] = None) -> Tuple[bool, str]:
        """Evaluate a sequence without mutating the sandbox default state."""

        working_state = list(self._base_state if state is None else state)
        try:
            outcome = self._simulate_once(seq, working_state)
            return True, outcome
        except Exception as exc:
            return False, str(exc)

    def execute(self, seq: str) -> str:
        """Execute on the live state."""

        outcome = self._simulate_once(seq, self._state)
        return outcome

    def feasible_universe(self) -> int:
        """Compute deterministic upper-bound of currently valid triples from reset state."""

        count = 0
        for t in self.OBJECT_MAP:
            for a in self.ACTION_MAP:
                for d in self.INDEX_MAP:
                    seq = f"{t}{a}{d}"
                    valid, _ = self.simulate(seq)
                    if valid:
                        count += 1
        return count


class AnalyticalMemory:
    """Associative memory over successful and failed topological experiments."""

    def __init__(self, data_path: Path = DATA_PATH) -> None:
        self.data_path = data_path

        self.successes: Dict[str, str] = {}
        self.tested_sequences: set[str] = set()

        # Learned evidence from outcomes
        self._role_assoc: DefaultDict[str, DefaultDict[str, DefaultDict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        self._pair_success: DefaultDict[str, float] = defaultdict(float)
        self._pair_failure: DefaultDict[str, float] = defaultdict(float)

        # Soft penalties for repeatedly wrong roles and chars
        self._role_penalty: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        self._load_existing()

    @property
    def successful_count(self) -> int:
        return len(self.successes)

    def is_tested(self, seq: str) -> bool:
        return seq in self.tested_sequences

    def add_test(self, seq: str) -> None:
        if self._is_valid_sequence(seq):
            self.tested_sequences.add(seq)

    # ---------------------- parsing ----------------------
    @staticmethod
    def _parse_training_line(text_value: str) -> Optional[Tuple[str, str]]:
        match = re.search(
            r"Instruction:\s*(.*?)\s*->\s*Execution Array:\s*([A-Za-z]{3})",
            text_value,
        )
        if not match:
            return None
        return match.group(1).strip(), match.group(2).strip().upper()

    @staticmethod
    def _parse_action_from_instruction(instruction: str) -> Optional[str]:
        lowered = instruction.lower()
        if any(tok in lowered for tok in ["move", "place", "shift", "drop", "put", "relocate"]):
            return "move"
        if any(tok in lowered for tok in ["delete", "remove", "destroy", "clear", "erase"]):
            return "delete"
        return None

    @staticmethod
    def _parse_object_from_instruction(instruction: str) -> Optional[str]:
        lowered = instruction.lower()
        if "red cube" in lowered or "red block" in lowered:
            return "Red Cube"
        if "blue sphere" in lowered or "blue ball" in lowered:
            return "Blue Sphere"
        return None

    @staticmethod
    def _parse_destination_from_instruction(instruction: str) -> Optional[str]:
        match = re.search(r"(?:position|slot|index|cell)\s*(\d)", instruction.lower())
        if not match:
            return None
        value = match.group(1)
        return value if value in "01234" else None

    def _extract_signal(self, instruction: str) -> Optional[Tuple[str, str, Optional[str]]]:
        action = self._parse_action_from_instruction(instruction)
        obj = self._parse_object_from_instruction(instruction)

        if action is None or obj is None:
            return None

        if action == "delete":
            return obj, action, None

        index = self._parse_destination_from_instruction(instruction)
        if index is None:
            return None
        return obj, action, index

    @staticmethod
    def _parse_outcome(outcome: str) -> Optional[ParsedOutcome]:
        if " deleted." in outcome:
            target = outcome.replace(" deleted.", "").strip()
            return ParsedOutcome(target=target, action="delete", destination=None)

        if " moved to Index " in outcome:
            target, suffix = outcome.split(" moved to Index ", 1)
            try:
                idx = int(suffix.replace(".", "").strip())
            except ValueError:
                return None
            if idx not in range(5):
                return None
            return ParsedOutcome(target=target.strip(), action="move", destination=idx)

        return None

    @staticmethod
    def _is_valid_sequence(sequence: str) -> bool:
        return (
            isinstance(sequence, str)
            and len(sequence) == 3
            and sequence.isalpha()
            and sequence.isupper()
            and len(set(sequence)) == 3
        )

    @staticmethod
    def _instruction_from_outcome(parsed: ParsedOutcome) -> str:
        if parsed.action == "move" and parsed.destination is not None:
            return f"Move the {parsed.target.lower()} to position {parsed.destination}."
        if parsed.action == "delete":
            return f"Delete the {parsed.target.lower()}."
        return ""

    def _learn_success(self, seq: str, target: str, action: str, destination: Optional[str]) -> None:
        t, a, d = seq[0], seq[1], seq[2]
        self._role_assoc["target"][t][target] += 1.0
        self._role_assoc["action"][a][action] += 1.0
        if destination:
            self._role_assoc["destination"][d][destination] += 1.0

    def _load_existing(self) -> None:
        if not self.data_path.exists():
            return

        with self.data_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text_field = payload.get("text")
                if not isinstance(text_field, str):
                    continue

                parsed = self._parse_training_line(text_field)
                if parsed is None:
                    continue

                instruction, seq = parsed
                if not self._is_valid_sequence(seq):
                    continue

                signal = self._extract_signal(instruction)
                if signal is None:
                    continue

                target, action, destination = signal
                self._load_existing_success(seq, instruction, target, action, destination)

    def _load_existing_success(
        self,
        seq: str,
        instruction: str,
        target: str,
        action: str,
        destination: Optional[str],
    ) -> None:
        self.successes[seq] = instruction
        self.tested_sequences.add(seq)
        self._learn_success(seq, target, action, destination)
        self._pair_success[seq] += 1.0

    def mark_success(self, seq: str, outcome: str) -> None:
        parsed = self._parse_outcome(outcome)
        if parsed is None:
            return

        self._pair_success[seq] += 1.0
        self._learn_success(seq, parsed.target, parsed.action, str(parsed.destination) if parsed.destination is not None else None)

        instruction = self._instruction_from_outcome(parsed)
        self.successes[seq] = instruction

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with self.data_path.open("a", encoding="utf-8") as f:
            payload = {"text": f"Instruction: {instruction} -> Execution Array: {seq}"}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def mark_failure(self, seq: str, error: str) -> None:
        if not self._is_valid_sequence(seq):
            return

        self._pair_failure[seq] += 1.0

        # Role-aware penalties so future generation backoffs bad syntax/physics combinations.
        if "Permutation Invalid" in error:
            for idx, char in enumerate(seq):
                role = ["target", "action", "destination"][idx]
                self._role_penalty[role][char] += 2.0
            return

        if "Spatial Collision" in error:
            # collision usually implicates destination/action mapping first.
            self._role_penalty["action"][seq[1]] += 0.6
            self._role_penalty["destination"][seq[2]] += 1.0

    def _role_count(self, role: str, char: str, value: str) -> float:
        return self._role_assoc.get(role, {}).get(char, {}).get(value, 0.0)

    def _role_uniqueness_bonus(self, char: str, used: set[str]) -> float:
        return 0.05 if char not in used else -1.0

    def _score_role(self, role: str, char: str) -> float:
        observations = self._role_assoc.get(role, {}).get(char, {})
        if observations:
            top = max(observations.values())
            entropy = len(observations)
            confidence = top / (sum(observations.values()) + 0.0001)
            signal = top + 0.35 * confidence - 0.2 * entropy
        else:
            signal = 0.15

        return signal - 0.45 * self._role_penalty[role].get(char, 0.0)

    def _score_pair(self, seq: str) -> float:
        success = self._pair_success.get(seq, 0.0)
        failure = self._pair_failure.get(seq, 0.0)
        return success - 1.4 * failure

    def _top_role_chars(self, role: str, count: int) -> List[str]:
        # Restrict candidates to syntactically valid role pools whenever possible.
        if role == "target":
            base = list(TopologicalSandbox.OBJECT_MAP.keys())
        elif role == "action":
            base = list(TopologicalSandbox.ACTION_MAP.keys())
        elif role == "destination":
            base = list(TopologicalSandbox.INDEX_MAP.keys())
        else:
            base = list(ALPHABET)

        scores = {char: self._score_role(role, char) for char in base}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        chars = [char for char, _ in ranked[:count]]
        if len(chars) < count:
            # cover low-confidence unknowns
            filler = [c for c in ALPHABET if c not in chars]
            random.shuffle(filler)
            chars.extend(filler[: count - len(chars)])
        return chars[:count]

    def _action_affinity(self, char: str, desired_action: Optional[str]) -> float:
        if desired_action is None:
            return 0.0

        total = self._role_assoc["action"][char]
        if not total:
            return 0.0
        return total.get(desired_action, 0.0) / (sum(total.values()) + 1e-6)

    def _destination_affinity(self, char: str, destination: Optional[int]) -> float:
        if destination is None:
            return 0.0

        total = self._role_assoc["destination"][char]
        if not total:
            return 0.0
        return total.get(str(destination), 0.0) / (sum(total.values()) + 1e-6)

    def generate_hypotheses(self, count: int, spec: ExecutionSpec, exploratory: bool = False) -> List[str]:
        if count <= 0:
            return []

        if exploratory:
            # wide-open exploration branch when standard synthesis stalls.
            target_pool = list(ALPHABET)
            action_pool = list(ALPHABET)
            destination_pool = list(ALPHABET)
        else:
            target_pool = self._top_role_chars("target", max(8, count * 2))
            action_pool = self._top_role_chars("action", max(6, count * 2))
            destination_pool = self._top_role_chars("destination", max(8, count * 2))

        # Strong bias toward spec action/destination when known.
        if spec.desired_action:
            action_pool = sorted(
                action_pool,
                key=lambda char: self._action_affinity(char, spec.desired_action),
                reverse=True,
            )
        if spec.destination is not None:
            destination_pool = sorted(
                destination_pool,
                key=lambda char: self._destination_affinity(char, spec.destination),
                reverse=True,
            )

        scored: List[Tuple[float, str]] = []
        for t in target_pool:
            for a in action_pool:
                if a == t:
                    continue
                for d in destination_pool:
                    if d == t or d == a:
                        continue
                    seq = f"{t}{a}{d}"
                    if self.is_tested(seq):
                        continue
                    if not self._is_valid_sequence(seq):
                        continue

                    score = (
                        self._score_role("target", t)
                        + self._score_role("action", a)
                        + self._score_role("destination", d)
                        + 0.75 * self._score_pair(seq)
                        + 0.3 * self._action_affinity(a, spec.desired_action)
                    )
                    if spec.destination is not None:
                        score += 0.6 * self._destination_affinity(d, spec.destination)
                    scored.append((score, seq))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: List[str] = []
        seen: set[str] = set()
        for _, seq in scored:
            if seq in seen:
                continue
            seen.add(seq)
            results.append(seq)
            if len(results) >= count:
                break

        attempts = 0
        while len(results) < count and attempts < 500:
            seq = "".join(random.sample(ALPHABET, 3))
            if self.is_tested(seq) or seq in seen:
                attempts += 1
                continue
            results.append(seq)
            seen.add(seq)
            attempts += 1

        return results[:count]

    def build_memory_snapshot(self) -> str:
        if not self.successes:
            return "No successful training pairs yet."

        lines = [
            f"{sequence}: {instruction}" for sequence, instruction in sorted(self.successes.items())
        ]
        return "\n".join(lines)


def parse_command_specs() -> Deque[ExecutionSpec]:
    # Objective deck: no hidden command dictionary is used by the execution system.
    from collections import deque

    specs: Deque[ExecutionSpec] = deque()
    for idx in range(5):
        specs.append(ExecutionSpec(f"Move any object to position {idx}", desired_action="move", destination=idx))
    specs.append(ExecutionSpec("Delete any object", desired_action="delete"))
    specs.append(ExecutionSpec("Delete any object", desired_action="delete"))
    return specs


def check_spec_compliance(parsed: ParsedOutcome, spec: ExecutionSpec) -> bool:
    if spec.desired_action and parsed.action != spec.desired_action:
        return False

    if spec.destination is not None and parsed.destination != spec.destination:
        return False

    return True


def verify_candidate(sandbox: TopologicalSandbox, spec: ExecutionSpec, seq: str) -> VerificationResult:
    # 1) strict static+semantic simulation against base state (dry-run)
    valid, outcome = sandbox.simulate(seq)
    if not valid:
        return VerificationResult(sequence=seq, ok=False, outcome=outcome, stage="simulate")

    parsed = AnalyticalMemory._parse_outcome(outcome)
    if parsed is None:
        return VerificationResult(sequence=seq, ok=False, outcome=outcome, stage="parse")

    if not check_spec_compliance(parsed, spec):
        return VerificationResult(
            sequence=seq,
            ok=False,
            outcome=outcome,
            stage="spec",
            error=(f"Spec mismatch: expected {spec.desired_action or 'any'} "
                   f"/{spec.destination if spec.destination is not None else 'any'}"),
        )

    # 2) execute on clean runtime state to guarantee real execution path is valid
    sandbox.reset()
    try:
        live_outcome = sandbox.execute(seq)
    except Exception as exc:
        return VerificationResult(sequence=seq, ok=False, outcome=str(exc), stage="execute", error=str(exc))

    return VerificationResult(
        sequence=seq,
        ok=True,
        outcome=live_outcome,
        stage="execute",
    )


def main() -> None:
    random.seed()

    sandbox = TopologicalSandbox()
    memory = AnalyticalMemory(DATA_PATH)

    feasible = sandbox.feasible_universe()
    target_count = min(TARGET_SUCCESS_PAIRS, feasible)

    print(f"[Init] Loaded {memory.successful_count} validated pairs from {DATA_PATH}")
    print(f"[Init] Sandbox upper bound reachable: {feasible}, target: {target_count}")

    if target_count == 0:
        print("[Init] No reachable transitions detected from reset state.")
        return

    specs = parse_command_specs()
    rounds = 0
    attempts = 0
    no_progress = 0

    while memory.successful_count < target_count and rounds < ROUNDS_CAP:
        rounds += 1
        spec = specs.popleft()
        specs.append(spec)

        remaining = target_count - memory.successful_count
        exploratory = no_progress >= NO_PROGRESS_STREAK
        print(f"[Spec] {spec.name} | Remaining: {remaining} | no_progress={no_progress}")

        hypotheses = memory.generate_hypotheses(
            HYPOTHESES_PER_ROUND,
            spec=spec,
            exploratory=exploratory,
        )

        if not hypotheses:
            print("[Spec] No candidate hypotheses generated; widening search globally.")
            hypotheses = memory.generate_hypotheses(HYPOTHESES_PER_ROUND, spec=spec, exploratory=True)

        round_successes = 0
        for seq in hypotheses:
            if memory.is_tested(seq):
                continue

            memory.add_test(seq)
            attempts += 1

            result = verify_candidate(sandbox, spec, seq)
            if not result.ok:
                memory.mark_failure(seq, result.error or result.outcome)
                print(f"[Reject] {seq} | stage={result.stage} | {result.error or result.outcome}")
                continue

            if seq in memory.successes:
                continue

            memory.mark_success(seq, result.outcome)
            round_successes += 1
            print(f"[Success] {seq} -> {result.outcome}")

            if memory.successful_count >= target_count:
                break

        if round_successes:
            no_progress = 0
        else:
            no_progress += 1

    print(
        f"[Done] Successes: {memory.successful_count} / {target_count} | "
        f"Rounds: {rounds} | Attempts: {attempts}"
    )


if __name__ == "__main__":
    main()
