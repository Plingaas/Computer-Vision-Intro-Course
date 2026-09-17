"""Generate deterministic, original practice images; existing files are preserved.

Run from repository root: python scripts/generate_task_data.py
Only NumPy and OpenCV are required. No downloads are used.
"""
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'


def save_missing(name, image):
    path = DATA / name
    if path.exists():
        print(f'Keeping {path.name}')
    elif not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')
    else:
        print(f'Created {path.name}')


def make_shapes():
    image = np.full((480, 640, 3), 230, np.uint8)
    cv2.rectangle(image, (45, 60), (250, 220), (35, 65, 100), -1)
    cv2.circle(image, (420, 145), 80, (70, 130, 40), -1)
    triangle = np.array([[310, 405], [450, 270], [570, 405]], np.int32)
    cv2.fillPoly(image, [triangle], (130, 40, 60))
    for x in range(45, 250, 20):
        cv2.line(image, (x, 280), (x, 390), (70, 70, 70), 2)
    cv2.putText(image, 'Computer Vision', (40, 450), cv2.FONT_HERSHEY_SIMPLEX,
                1, (20, 20, 20), 2, cv2.LINE_AA)
    return image


def make_poster():
    rng = np.random.default_rng(7)
    poster = np.full((420, 560, 3), 240, np.uint8)
    cv2.rectangle(poster, (12, 12), (547, 407), (25, 25, 25), 4)
    cv2.putText(poster, 'VISION LAB', (30, 68), cv2.FONT_HERSHEY_DUPLEX,
                1.8, (30, 40, 70), 3, cv2.LINE_AA)
    cv2.putText(poster, 'Find this poster!', (35, 385), cv2.FONT_HERSHEY_SIMPLEX,
                1, (50, 30, 20), 2, cv2.LINE_AA)
    # Irregular patterns give ORB distinctive neighborhoods, unlike a repeated grid.
    for _ in range(150):
        x, y = rng.integers([35, 95], [525, 340])
        color = tuple(int(v) for v in rng.integers(0, 200, 3))
        radius = int(rng.integers(3, 13))
        cv2.circle(poster, (int(x), int(y)), radius, color, -1)
    for index, (x, y) in enumerate([(45, 135), (215, 205), (395, 300)]):
        cv2.putText(poster, str(index + 1), (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    1.5, (0, 0, 0), 3, cv2.LINE_AA)
    return poster


def make_scene(poster):
    scene = np.full((700, 960, 3), (165, 180, 190), np.uint8)
    cv2.circle(scene, (820, 125), 65, (60, 110, 190), -1)
    cv2.rectangle(scene, (45, 480), (190, 650), (70, 100, 50), -1)
    cv2.putText(scene, 'SCENE', (650, 630), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (30, 30, 30), 3, cv2.LINE_AA)
    h, w = poster.shape[:2]
    source = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    target = np.float32([[180, 130], [730, 75], [790, 510], [230, 570]])
    transform = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(poster, transform, (960, 700))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), transform, (960, 700))
    scene[mask > 0] = warped[mask > 0]
    return scene


def main():
    DATA.mkdir(exist_ok=True)
    shapes = make_shapes()
    rng = np.random.default_rng(3)
    noisy = np.clip(shapes.astype(np.float32) + rng.normal(0, 25, shapes.shape), 0, 255).astype(np.uint8)
    save_missing('task3.png', noisy)
    save_missing('task4.png', shapes)
    save_missing('task5.png', shapes)
    poster = make_poster()
    h, w = poster.shape[:2]
    source = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    target = np.float32([[35, 25], [w-35, 12], [w-15, h-35], [20, h-10]])
    transform = cv2.getPerspectiveTransform(source, target)
    second = cv2.warpPerspective(poster, transform, (w, h), borderValue=(180, 180, 180))
    save_missing('task6_a.png', poster)
    save_missing('task6_b.png', second)
    save_missing('task7_object.png', poster)
    save_missing('task7_scene.png', make_scene(poster))


if __name__ == '__main__':
    main()
