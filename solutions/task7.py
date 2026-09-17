"""Task 7 (30–45 min): locate a planar object using RANSAC and a homography.
A homography maps coordinates between two views of a plane.
RANSAC finds a consistent geometric model despite some incorrect matches.
"""
import cv2
import numpy as np


def open_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}. Run from the repository root.")
    return image


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1500)
    return orb.detectAndCompute(gray, None)


def match_features(desc1, desc2):
    if desc1 is None or desc2 is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    return sorted(matches, key=lambda match: match.distance)


def matched_points(kp1, kp2, matches):
    # queryIdx refers to the object; trainIdx refers to the scene.
    # Each keypoint's .pt is its (x, y) position.
    object_points = [kp1[match.queryIdx].pt for match in matches]
    scene_points = [kp2[match.trainIdx].pt for match in matches]
    # Supplied shape conversion: OpenCV expects float32 point arrays.
    return (np.float32(object_points).reshape(-1, 1, 2),
            np.float32(scene_points).reshape(-1, 1, 2))


def estimate_homography(points1, points2):
    if len(points1) < 4:
        raise ValueError("At least four correspondences are needed for a homography.")
    return cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)


def draw_detected_object(scene_image, object_shape, homography):
    height, width = object_shape[:2]
    corners = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    corners = corners.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography)
    result = scene_image.copy()
    cv2.polylines(result, [np.int32(projected)], True, (0, 255, 0), 3, cv2.LINE_AA)
    return result


def show_image(name, image):
    # Window setup is supplied so large images fit on the screen.
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    scale = min(900 / width, 600 / height, 1.0)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
    cv2.imshow(name, image)


def main():
    object_image = open_image("data/task7_object.png")
    scene_image = open_image("data/task7_scene.png")
    kp1, desc1 = detect_and_describe(object_image)
    kp2, desc2 = detect_and_describe(scene_image)
    matches = match_features(desc1, desc2)[:80]
    if len(matches) < 4:
        print("Too few matches: need at least four. Try more ORB features.")
        return
    points1, points2 = matched_points(kp1, kp2, matches)
    homography, inliers = estimate_homography(points1, points2)
    if homography is None or inliers is None or np.count_nonzero(inliers) < 4:
        print("No reliable homography found. Check the matches and images.")
        return
    located = draw_detected_object(scene_image, object_image.shape, homography)
    # Show all candidate matches, then only matches that agree with the geometry.
    candidates = cv2.drawMatches(object_image, kp1, scene_image, kp2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    consistent = cv2.drawMatches(object_image, kp1, scene_image, kp2, matches, None,
                                matchesMask=inliers.ravel().tolist(),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print(f"RANSAC kept {np.count_nonzero(inliers)} of {len(matches)} matches.")
    show_image("Task 7 - before RANSAC", candidates)
    show_image("Task 7 - RANSAC inliers", consistent)
    show_image("Task 7 - located object", located)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
