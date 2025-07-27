#!/usr/bin/env python
"""CLI to generate CWT image dataset from hawk-data tests."""
import argparse
import os
from typing import Dict, List

import numpy as np

try:
    from hawk import FST  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    FST = None  # type: ignore

from src.utils.cwt import generate_cwt_image, save_images

TESTS_PER_CLASS: Dict[str, List[str]] = {
    "HS": [
        "HS_WN/01",
        "HS_WN/02",
        "HS_WN/03",
        "HS_WN/04",
        "HS_WN/05",
    ],
    "DS_CTE": [
        "DS_WN/01",
        "DS_WN/02",
        "DS_WN/03",
        "DS_WN/19",
        "DS_WN/20",
        "DS_WN/21",
        "DS_WN/64",
        "DS_WN/65",
        "DS_WN/66",
    ],
    "DS_RLE": [
        "DS_WN/04",
        "DS_WN/05",
        "DS_WN/06",
        "DS_WN/22",
        "DS_WN/23",
        "DS_WN/24",
        "DS_WN/67",
        "DS_WN/68",
        "DS_WN/69",
    ],
    "DS_TLE": [
        "DS_WN/07",
        "DS_WN/08",
        "DS_WN/09",
        "DS_WN/25",
        "DS_WN/26",
        "DS_WN/27",
        "DS_WN/70",
        "DS_WN/71",
        "DS_WN/72",
    ],
}


def extract_signal(group) -> np.ndarray:
    """Return stacked mean strain for the 10 FBG sensors."""
    signals = []
    for i in range(1, 11):
        sensor = f"SW_FB{i}"
        strain = group[sensor]["strain"][:]
        signal_mean = strain.mean(axis=1)
        signals.append(signal_mean)
    return np.array(signals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CWT images from hawk-data")
    parser.add_argument("data_dir", help="Path to hawk_data directory")
    parser.add_argument("output_dir", help="Where to store generated images")
    parser.add_argument("--width", type=int, default=224, help="Target scalogram width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if FST is None:
        raise RuntimeError("hawk-data package is required to generate the dataset")

    data = FST(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for clazz, tests in TESTS_PER_CLASS.items():
        for test in tests:
            print(f"Processing {test} ({clazz})")
            group = data[test]
            signals = extract_signal(group)
            images = generate_cwt_image(signals, target_width=args.width)
            save_images(images, args.output_dir, test, clazz)
            print(f"  -> saved {len(images)} images")


if __name__ == "__main__":
    main()
