#!/usr/bin/env python3
"""Automated curriculum data generator for the GlyphSandbox VLM task."""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

# 5-slot 1D sandbox settings
GRID_SIZE = 5
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 180
GLYPH_SIZE = 110
NUM_EXAMPLES = 1000

# English names + glyph labels
GLYPH_PALETTE: Dict[str, Dict[str, str]] = {
    "glyph_A": {"name": "red block", "color": "red", "rgb": (220, 52, 63, 255), "shape": "square"},
    "glyph_B": {"name": "blue sphere", "color": "blue", "rgb": (52, 120, 246, 255), "shape": "circle"},
    "glyph_C": {"name": "green pyramid", "color": "green", "rgb": (56, 189, 94, 255), "shape": "triangle"},
}

glyph_ids = list(GLYPH_PALETTE.keys())


def build_glyph_cache() -> Dict[str, Image.Image]:
    glyphs: Dict[str, Image.Image] = {}
    for glyph_id, meta in GLYPH_PALETTE.items():
        shape = meta["shape"]
        color = meta["rgb"]

        img = Image.new("RGBA", (GLYPH_SIZE, GLYPH_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if shape == "square":
            draw.rectangle((0, 0, GLYPH_SIZE - 1, GLYPH_SIZE - 1), fill=color)
        elif shape == "circle":
            draw.ellipse((0, 0, GLYPH_SIZE - 1, GLYPH_SIZE - 1), fill=color)
        elif shape == "triangle":
            points = [
                (GLYPH_SIZE // 2, 6),
                (GLYPH_SIZE - 6, GLYPH_SIZE - 6),
                (6, GLYPH_SIZE - 6),
            ]
            draw.polygon(points, fill=color)
        else:
            raise ValueError(f"Unknown shape for {glyph_id}")

        glyphs[glyph_id] = img

    return glyphs


def generate_random_grid() -> List[str | None]:
    """Place 1 to 3 unique glyphs into a 5-slot grid."""
    grid: List[str | None] = [None] * GRID_SIZE

    k = random.randint(1, min(3, len(glyph_ids), GRID_SIZE))
    active_glyphs = random.sample(glyph_ids, k=k)
    active_positions = random.sample(range(GRID_SIZE), k=k)

    for glyph_id, pos in zip(active_glyphs, active_positions):
        grid[pos] = glyph_id

    return grid


def render_grid_to_image(grid_array: List[str | None], filename: str, cache: Dict[str, Image.Image]) -> str:
    base = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (245, 245, 245, 255))
    slot_w = CANVAS_WIDTH // GRID_SIZE
    slot_h = CANVAS_HEIGHT
    glyph_h = cache["glyph_A"].height
    y = (slot_h - glyph_h) // 2

    for idx, glyph_id in enumerate(grid_array):
        if glyph_id is None:
            continue
        if glyph_id not in cache:
            continue

        slot_x0 = idx * slot_w
        slot_x1 = min((idx + 1) * slot_w, CANVAS_WIDTH)
        glyph_w = cache[glyph_id].width
        x = slot_x0 + (slot_x1 - slot_x0 - glyph_w) // 2
        base.alpha_composite(cache[glyph_id], (x, y))

    base.save(filename)
    return filename


def _pick_random_occupied(grid_array: List[str | None]) -> Tuple[int, str]:
    occupied_indices = [i for i, v in enumerate(grid_array) if v is not None]
    if not occupied_indices:
        raise ValueError("Grid must contain at least one glyph.")

    idx = random.choice(occupied_indices)
    return idx, grid_array[idx]


def generate_curriculum_pair(grid_array: List[str | None]) -> Tuple[str, str]:
    """Return (english_command, python_call)."""
    phase = random.choice([1, 2, 3])

    source_idx, glyph_id = _pick_random_occupied(grid_array)
    glyph_name = GLYPH_PALETTE[glyph_id]["name"]

    if phase == 1:
        return (
            f"Select the {glyph_name}.",
            f"select_glyph('{glyph_id}')",
        )

    if phase == 2:
        return (
            f"Delete the {glyph_name}.",
            f"delete_glyph('{glyph_id}')",
        )

    empty_indices = [i for i, v in enumerate(grid_array) if v is None]
    empty_index = random.choice(empty_indices)
    return (
        f"Move the {glyph_name} to position {empty_index}.",
        f"move_glyph('{glyph_id}', {empty_index})",
    )


def main() -> None:
    out_dir = Path("data")
    images_dir = out_dir / "images"
    train_file = out_dir / "train.jsonl"

    images_dir.mkdir(parents=True, exist_ok=True)

    glyph_cache = build_glyph_cache()

    with train_file.open("w", encoding="utf-8") as f:
        for _ in range(NUM_EXAMPLES):
            grid = generate_random_grid()
            english, code = generate_curriculum_pair(grid)

            uid = uuid.uuid4().hex
            img_path = images_dir / f"state_{uid}.png"
            render_grid_to_image(grid, str(img_path), glyph_cache)

            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_path)},
                            {"type": "text", "text": english},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": code,
                    },
                ]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Generated {NUM_EXAMPLES} examples at {train_file}")


if __name__ == "__main__":
    main()
