#!/usr/bin/env python3
"""Glyph-grounding sandbox with strict spatial physics and automatic data collection.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw

MODEL_NAME = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
MAX_RETRIES = 5
GRID_SIZE = 5
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 180
GLYPH_SIZE = 110
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
TRAIN_PATH = DATA_DIR / "train.jsonl"

# Runtime cache for loaded VLM objects.
_VLM_STATE: Dict[str, Any] = {"model": None, "processor": None}


class GlyphSandbox:
    """A 1D spatial sandbox with cached PIL glyph primitives."""

    def __init__(self) -> None:
        self.grid_size = GRID_SIZE
        self.canvas_size = (CANVAS_WIDTH, CANVAS_HEIGHT)
        self.glyph_size = GLYPH_SIZE

        # Pointer array tracks which glyph occupies each slot.
        self.grid: list[Optional[str]] = [None] * self.grid_size
        self.grid[0] = "glyph_A"
        self.grid[4] = "glyph_B"

        # Pre-rendered glyph cache (in-memory only).
        self.glyphs: Dict[str, Image.Image] = self._build_glyph_cache()

    def _build_glyph_cache(self) -> Dict[str, Image.Image]:
        square = self._make_square_glyph((220, 52, 63, 255))
        circle = self._make_circle_glyph((52, 120, 246, 255))
        tri = self._make_triangle_glyph((56, 189, 94, 255))
        return {"glyph_A": square, "glyph_B": circle, "glyph_C": tri}

    def _make_square_glyph(self, color: Tuple[int, int, int, int]) -> Image.Image:
        glyph = Image.new("RGBA", (self.glyph_size, self.glyph_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glyph)
        draw.rectangle((0, 0, self.glyph_size - 1, self.glyph_size - 1), fill=color)
        return glyph

    def _make_circle_glyph(self, color: Tuple[int, int, int, int]) -> Image.Image:
        glyph = Image.new("RGBA", (self.glyph_size, self.glyph_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glyph)
        draw.ellipse((0, 0, self.glyph_size - 1, self.glyph_size - 1), fill=color)
        return glyph

    def _make_triangle_glyph(self, color: Tuple[int, int, int, int]) -> Image.Image:
        glyph = Image.new("RGBA", (self.glyph_size, self.glyph_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glyph)
        points = [
            (self.glyph_size // 2, 6),
            (self.glyph_size - 6, self.glyph_size - 6),
            (6, self.glyph_size - 6),
        ]
        draw.polygon(points, fill=color)
        return glyph

    def render_canvas(self, filename: str = "grid_canvas.png") -> str:
        """Compose current glyph pointers into a single 5-slice canvas and save it."""
        base = Image.new("RGBA", self.canvas_size, (250, 250, 250, 255))

        slot_w = self.canvas_size[0] // self.grid_size
        slot_h = self.canvas_size[1]
        glyph_h = self.glyphs["glyph_A"].height
        y = (slot_h - glyph_h) // 2

        for idx, glyph_id in enumerate(self.grid):
            if glyph_id is None:
                continue
            if glyph_id not in self.glyphs:
                continue

            slot_x0 = idx * slot_w
            # Relative slice math (position derived from slot boundaries).
            slot_x1 = min((idx + 1) * slot_w, self.canvas_size[0])
            glyph_w = self.glyphs[glyph_id].width
            x = slot_x0 + (slot_x1 - slot_x0 - glyph_w) // 2
            base.alpha_composite(self.glyphs[glyph_id], (x, y))

        base.save(filename)
        return filename

    def move_glyph(self, target_glyph_id: str, destination_index: int) -> None:
        if not isinstance(destination_index, int):
            destination_index = int(destination_index)

        if target_glyph_id not in self.glyphs:
            raise ValueError("Composition Error: Pointer Target Unresolved")

        if target_glyph_id not in self.grid:
            raise ValueError("Composition Error: Pointer Target Unresolved")

        if destination_index < 0 or destination_index >= self.grid_size:
            raise ValueError("Composition Error: Spatial Collision Detected")

        occupant = self.grid[destination_index]
        if occupant is not None and occupant != target_glyph_id:
            raise ValueError("Composition Error: Spatial Collision Detected")

        source = self.grid.index(target_glyph_id)
        self.grid[source] = None
        self.grid[destination_index] = target_glyph_id

    def execute_action_string(self, python_code_str: str) -> str:
        """Execute exactly one call in the form move_glyph('<glyph_id>', <index>)."""
        expr = clean_action_string(python_code_str)
        if not expr:
            raise ValueError(f"Invalid action syntax: {python_code_str}")

        tree = ast.parse(expr, mode="exec")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            raise ValueError("Invalid action expression: expected single function call")

        call = tree.body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Invalid action expression: expected move_glyph call")

        if call.func.id != "move_glyph":
            raise ValueError("Invalid action expression: expected move_glyph")

        if len(call.args) != 2:
            raise ValueError("Invalid move_glyph arity: expected 2 arguments")

        arg0 = call.args[0]
        arg1 = call.args[1]
        if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
            raise ValueError("Invalid move_glyph target glyph id")

        if isinstance(arg1, ast.UnaryOp) and isinstance(arg1.op, ast.USub) and isinstance(arg1.operand, ast.Constant):
            destination = -int(arg1.operand.value)
        elif isinstance(arg1, ast.Constant) and isinstance(arg1.value, int):
            destination = int(arg1.value)
        else:
            raise ValueError("Invalid move_glyph destination index")

        self.move_glyph(arg0.value, destination)
        return expr


def clean_action_string(text: str) -> str:
    """Sanitize LLM output to enforce a single raw move_glyph call string."""
    if not text:
        return ""

    s = text.strip()
    s = s.replace("```python", "").replace("```", "").strip()

    # Pull first valid call if wrappers/comments are present.
    match = re.search(r"move_glyph\s*\(\s*['\"]([^'\"]+)['\"],\s*([-+]?\d+)\s*\)", s)
    if not match:
        return ""

    glyph_id = match.group(1)
    index = int(match.group(2))
    return f"move_glyph('{glyph_id}', {index})"


def _parse_error_feedback_to_prompt(error_feedback: Optional[str]) -> str:
    if not error_feedback:
        return ""
    return (
        "Your previous attempt failed with this runtime error: "
        f"{error_feedback}. Adjust your translation and output the corrected function call."
    )


def _load_vlm_model():
    if _VLM_STATE["model"] is not None and _VLM_STATE["processor"] is not None:
        return _VLM_STATE["model"], _VLM_STATE["processor"]

    try:
        from mlx_vlm import load
    except Exception as exc:  # pragma: no cover - module may be absent
        raise RuntimeError(
            "mlx_vlm is not importable in this environment.\n"
            "Install it with: pip install mlx-vlm"
        ) from exc

    model, processor = load(MODEL_NAME)
    _VLM_STATE["model"] = model
    _VLM_STATE["processor"] = processor
    return model, processor


def _ensure_image_format(image_path: str) -> Any:
    try:
        from mlx_vlm.utils import load_image

        return load_image(image_path)
    except Exception:
        # Generic Pillow fallback path for processors that can consume PIL objects.
        return Image.open(image_path).convert("RGB")


def query_vlm(image_path: str, english_command: str, error_log: Optional[str] = None) -> str:
    """Query local mlx-vlm and force a single `move_glyph('glyph_X', index)` result."""
    system_prompt = (
        "You are a Functional Token Mapper. Look at the grid canvas. "
        "Translate the English instruction into a single Python call: move_glyph('glyph_X', index). "
        "Output ONLY the raw function string."
    )
    user_prompt = english_command
    penalty = _parse_error_feedback_to_prompt(error_log)
    if penalty:
        user_prompt = f"{english_command} {penalty}"

    prefill = "move_glyph("
    try:
        model, processor = _load_vlm_model()
    except RuntimeError as exc:
        # Keep loop alive for environments without mlx-vlm by raising a clear deterministic message.
        raise RuntimeError(str(exc))

    try:
        image_obj = _ensure_image_format(image_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_obj},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        if hasattr(processor, "apply_chat_template"):
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompt = f"{prompt}{prefill}"
        else:
            prompt = f"{system_prompt}\nUser: {user_prompt}\nAssistant: {prefill}"

        inputs = processor(prompt, images=[image_obj], return_tensors="np")
        tokens = model.generate(
            **inputs,
            max_tokens=64,
            temperature=0.0,
        )
        raw = processor.decode(tokens[0], skip_special_tokens=True)
    except Exception as exc:
        raise RuntimeError(f"VLM execution failed: {exc}") from exc

    # Aggressive cleanup: enforce raw move call prefix.
    cleaned = clean_action_string(raw)
    if not cleaned:
        raw_line = raw.splitlines()[0] if isinstance(raw, str) and raw else ""
        if raw_line.startswith("move_glyph("):
            cleaned = raw_line.strip()
    if not cleaned:
        raise RuntimeError(f"Invalid VLM output: {raw}")

    return cleaned


def append_training_example(command: str, action: str, canvas_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(canvas_path)},
                    {"type": "text", "text": command},
                ],
            },
            {
                "role": "assistant",
                "content": action,
            },
        ]
    }

    with TRAIN_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    commands = [
        "Move the red block to position 2.",
        "Place the blue sphere where the red cube is.",
    ]

    sandbox = GlyphSandbox()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for i, command in enumerate(commands, start=1):
        print(f"\n=== Executing command {i}/{len(commands)}: {command}")
        retries = 0
        error_feedback = None
        action = None

        while retries < MAX_RETRIES:
            attempt = retries + 1
            print(f"Attempt {attempt}/{MAX_RETRIES}")

            # Always render a current canvas before asking VLM.
            latest_canvas = Path("grid_canvas.png")
            sandbox.render_canvas(str(latest_canvas))

            try:
                action = query_vlm(str(latest_canvas), command, error_log=error_feedback)
                print(f"VLM Action: {action}")
            except Exception as exc:
                error_feedback = f"{exc}"
                retries += 1
                print(f"VLM failure: {error_feedback}")
                continue

            try:
                sandbox.execute_action_string(action)
            except Exception as exc:
                error_feedback = f"{exc}"
                retries += 1
                print(f"Execution failure: {error_feedback}")
                continue

            # Success path: persist verified state and training example.
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            saved_canvas = IMAGES_DIR / f"canvas_{ts}.png"
            sandbox.render_canvas(str(saved_canvas))

            append_training_example(command, action, saved_canvas)
            print("Success: state updated and training sample saved.")
            print(f"State: {sandbox.grid}")
            print(f"Canvas snapshot: {saved_canvas}")
            break

            
        if retries >= MAX_RETRIES:
            print(
                f"Failure: could not resolve command after {MAX_RETRIES} retries.\n"
                f"Last error: {error_feedback}"
            )
            continue


if __name__ == "__main__":
    main()
