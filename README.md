# Introduction to Computer Vision

Seven short OpenCV exercises for first-year engineering students. Run commands from
this repository's root directory. Images are local; the exercises need no network.

## Setup and running

```bash
python -m pip install -r requirements.txt
python task2.py
```

Use a Python environment with a desktop display for the OpenCV windows. Press a key
in an image window to close the results. Windows in tasks 2–7 are resizable.

Task 1 is preserved as supplied. Tasks 2–7 are **student templates**: fill in each
`TODO` before running the whole pipeline. Until then, `NotImplementedError` marks
the next step to implement. Image loading, window handling, and some coordinate
handling are supplied so the exercises focus on computer vision.

Complete teacher implementations use the same function structure:

```bash
python solutions/task1.py
python solutions/task2.py
# ... through solutions/task7.py
```

## Exercises

| Task | Time | What you build and learn |
|---|---|---|
| 1 | 10–15 min | Open an image and convert BGR to grayscale; images as pixel arrays. Existing student file is unchanged. |
| 2 | 15–20 min | Threshold `data/task2.jpg`, identify its bright AprilTag region, and extract a crop. Learn binary masks and row/column slicing. |
| 3 | 10–15 min | Compare Gaussian kernels `(3, 3)` and `(11, 11)` on a noisy image. Noise reduction costs detail. |
| 4 | 10–20 min | Grayscale → light blur → Canny edges. Experiment with both edge thresholds. |
| 5 | 10–20 min | Detect Shi–Tomasi corners and draw them. Consider why corners are more distinctive than flat regions or straight edges. |
| 6 | 20–30 min | Detect ORB keypoints and descriptors, match with Hamming distance, and display the best 30 correspondences. |
| 7 | 30–45 min | Match a planar poster to a scene, estimate a homography with RANSAC, inspect inliers, and project the poster's outline. |

```text
1 Pixels / image representation
  → 2 Thresholding / masks
  → 3 Filtering / convolution intuition
  → 4 Edges
  → 5 Corners / keypoints
  → 6 Descriptors / matching
  → 7 Robust geometry with RANSAC
```

Task 2 follows the existing request to **extract the AprilTag**, rather than the
prompt document's generic threshold-only task. A supplied contour helper locates
the largest bright region in this particular photo. Students implement thresholding
and cropping; this is not an AprilTag ID decoder or a general tag detector. The tag
is only about 18 × 17 pixels, so enlarging the crop cannot restore missing detail.

In task 7, matches alone are not proof of a correct localization. Compare all
candidate matches with those accepted by RANSAC. At least four correspondences
are needed; even then, poor or degenerate matches can fail to determine a model.

### Connection to learned methods

Shi–Tomasi and ORB keypoint detection address the kind of localization problem
SuperPoint learns. ORB descriptors serve the same role as learned SuperPoint
descriptors. BFMatcher addresses the correspondence problem that SuperGlue learns.
RANSAC and geometric checks can still be used after learned features and matches.

## Images

- Tasks 1 and 2 use the existing `data/task1.png` and `data/task2.jpg`.
- Tasks 3–5 use original synthetic shapes; task 3 adds reproducible Gaussian noise.
- Tasks 6–7 use an original textured poster and known perspective transformations.
- The generated PNGs are included. To create any missing ones locally:

```bash
python scripts/generate_task_data.py
```

The generator preserves existing files and never downloads images. It does not
replace tasks 1–2 images. Existing challenge exercises are separate and unchanged.
