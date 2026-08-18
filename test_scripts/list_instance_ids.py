#!/usr/bin/env python3
"""List the instance IDs and pixel counts in semantic-instance images."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
from PIL import Image


DEFAULT_INPUT = Path("~/input/custom/dataset_easy/semantic_instance")
SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff"}


def natural_key(path: Path) -> list[int | str]:
    """Sort semantic_instance_2 before semantic_instance_10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def count_ids(path: Path) -> dict[int, int]:
    with Image.open(path) as image:
        if len(image.getbands()) != 1:
            raise ValueError(
                f"{path} is {image.mode!r}; expected a single-channel label image"
            )
        ids, counts = np.unique(np.asarray(image), return_counts=True)
        max_id=ids.max()
        return dict(zip(ids.astype(int).tolist(), counts.astype(int).tolist())), max_id


def write_occurrences(paths: list[Path], output: TextIO) -> None:
    writer = csv.writer(output)
    writer.writerow(("image", "id", "pixel_count", "image_fraction"))
    max_ids=[]

    for path in paths:
        counts,max_id = count_ids(path)
        pixel_total = sum(counts.values())
        for instance_id in sorted(counts):
            pixel_count = counts[instance_id]
            max_ids.append(max_id)
            writer.writerow(
                (
                    path.name,
                    instance_id,
                    pixel_count,
                    f"{pixel_count / pixel_total:.8f}",
                )
            )
    print(max(max_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List every instance ID found in each semantic-instance image, "
            "including its pixel count."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"mask directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write CSV to this file instead of standard output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    paths = sorted(
        (
            path
            for path in input_dir.glob("semantic_instance_*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=natural_key,
    ) if input_dir.is_dir() else []

    if not paths:
        print(f"No supported images found in {input_dir}", file=sys.stderr)
        return 1

    if args.output:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output:
            write_occurrences(paths, output)
        print(f"Wrote ID occurrences for {len(paths)} images to {output_path}")
    else:
        write_occurrences(paths, sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
