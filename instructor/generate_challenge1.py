"""Instructor-only: generate the hidden-image exercise and check recovery.

Run from the repository root:
    .venv/bin/python instructor/generate_challenge1.py
Add --preview /tmp/challenge1_recovered.png to inspect the reference recovery.
Do not distribute this script or the original image with the student exercise.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOW, HIGH = 112, 143


def embed(rgb, seed=42):
    """Encode RGB signal in a narrow band; replace 75% with out-of-band noise."""
    rng = np.random.default_rng(seed)
    signal = np.rint(LOW + rgb.astype(np.float32) * (HIGH - LOW) / 255).astype(np.uint8)
    allowed = np.concatenate((np.arange(LOW), np.arange(HIGH + 1, 256))).astype(np.uint8)
    noise = rng.choice(allowed, size=rgb.shape)
    # Shared support preserves complete RGB samples; noise colors are independent.
    keep = rng.random(rgb.shape[:2]) < 0.25
    return np.where(keep[:, :, None], signal, noise).astype(np.uint8)


def recover(rgb):
    """Reference: histogram band selection, normalized smoothing, contrast stretch."""
    valid = np.all((rgb >= LOW) & (rgb <= HIGH), axis=2).astype(np.float32)
    weight = cv2.GaussianBlur(valid, (0, 0), 2.0)
    numerator = cv2.GaussianBlur(rgb.astype(np.float32) * valid[:, :, None], (0, 0), 2.0)
    smoothed = numerator / np.maximum(weight[:, :, None], 1e-6)
    return np.clip((smoothed - LOW) * 255 / (HIGH - LOW), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/challenge1.jpeg')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/challenge1_noisy.png')
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    bgr = cv2.imread(str(args.source))
    if bgr is None:
        parser.error(f'Cannot read {args.source}')
    if args.output.suffix.lower() != '.png':
        parser.error('Use PNG to preserve the encoded intensity values')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    noisy = embed(rgb, args.seed)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Cannot save {args.output}')
    if args.preview:
        if not cv2.imwrite(str(args.preview), cv2.cvtColor(recover(noisy), cv2.COLOR_RGB2BGR)):
            raise OSError(f'Cannot save {args.preview}')
    print(f'Saved {args.output}: {noisy.shape[1]} x {noisy.shape[0]}, RGB, uint8')


if __name__ == '__main__':
    main()
