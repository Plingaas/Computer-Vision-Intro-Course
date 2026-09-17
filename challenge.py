"""Challenge 1: uncover an image using RGB histograms and noise removal.

Input: data/challenge1_noisy.png. Work only from this image!
1. Inspect the three channel histograms. Find a band with unusual frequency.
2. Select pixels belonging to that band in all three channels.
3. Reconstruct the missing pixels using a neighborhood filter.
4. Stretch the recovered intensity range to 0..255 and reveal the image.

Questions: Why does ordinary blurring fail? Why must rejected pixels be
excluded from the averaging weights? What happens if the image is saved as JPEG?
"""

from pathlib import Path

import cv2
import numpy as np


def show_image(name, rgb):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 640)
    cv2.imshow(name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def show_histograms(rgb):
    """Plot each RGB channel without requiring matplotlib."""
    canvas = np.full((600, 768, 3), 245, np.uint8)
    for channel, (name, color) in enumerate(zip(
            ('Red', 'Green', 'Blue'), ((0, 0, 220), (0, 150, 0), (220, 0, 0)))):
        counts = np.bincount(rgb[:, :, channel].ravel(), minlength=256)
        bottom = channel * 200 + 175
        heights = np.rint(counts / max(1, counts.max()) * 140).astype(int)
        for value, height in enumerate(heights):
            cv2.line(canvas, (value * 3, bottom), (value * 3, bottom - height), color, 2)
        cv2.putText(canvas, f'{name}: intensity 0 to 255; peak {counts.max()} pixels',
                    (10, channel * 200 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.namedWindow('RGB histograms', cv2.WINDOW_NORMAL)
    cv2.imshow('RGB histograms', canvas)


def remove_noise(rgb, low, high):
    # Keep only pixels whose three channels lie in the signal band.
    valid = np.all((rgb >= low) & (rgb <= high), axis=2).astype(np.float32)
    weights = cv2.GaussianBlur(valid, (0, 0), 2.0)
    masked = rgb.astype(np.float32) * valid[:, :, None]
    smoothed = cv2.GaussianBlur(masked, (0, 0), 2.0)
    # Normalize by valid neighbors so discarded noise does not darken the image.
    return smoothed / np.maximum(weights[:, :, None], 1e-6)


def stretch_contrast(image, low, high):
    if high <= low:
        raise ValueError('high must be greater than low')
    expanded = (image.astype(np.float32) - low) * 255 / (high - low)
    return np.clip(expanded, 0, 255).astype(np.uint8)


def main():
    path = Path(__file__).parent / 'data/challenge1_noisy.png'
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        show_image('Challenge 1 - noisy image', rgb)
        show_histograms(rgb)
        # The histogram's concentrated signal band spans intensities 112..143.
        low, high = 112, 143
        denoised = remove_noise(rgb, low, high)
        recovered = stretch_contrast(denoised, low, high)
        show_image('Challenge 1 - recovered image', recovered)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
