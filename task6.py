"""Task 6: ORB keypoints, descriptors, and matching.
Keypoint = where. Descriptor = what the local neighborhood looks like.
A match links similar descriptors in the two images; it can still be wrong.
"""
import cv2


def open_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}. Run from the repository root.")
    return image


def detect_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # TODO: Create ORB with cv2.ORB_create(nfeatures=1500).
    # TODO: Return keypoints and descriptors from detectAndCompute(gray, None).
    raise NotImplementedError("Detect and describe features")


def match_descriptors(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return []
    # TODO: Create cv2.BFMatcher with cv2.NORM_HAMMING and crossCheck=True.
    # TODO: Call matcher.match(), then sort by distance (smaller is better).
    # Hint: sorted(matches, key=lambda match: match.distance).
    raise NotImplementedError("Match binary descriptors")


def draw_best_matches(image1, keypoints1, image2, keypoints2, matches, num_matches=30):
    # TODO: Call cv2.drawMatches() with both images/keypoint lists,
    # matches[:num_matches], and None for the output image. Return the result.
    raise NotImplementedError("Visualize the best matches")


def show_image(name, image):
    # Window setup is supplied so large images fit on the screen.
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    scale = min(900 / width, 600 / height, 1.0)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
    cv2.imshow(name, image)


def main():
    image1 = open_image("data/task6_a.png")
    image2 = open_image("data/task6_b.png")
    keypoints1, descriptors1 = detect_features(image1)
    keypoints2, descriptors2 = detect_features(image2)
    matches = match_descriptors(descriptors1, descriptors2)
    if not matches:
        print("No matches found. Try images with more texture.")
        return
    result = draw_best_matches(image1, keypoints1, image2, keypoints2, matches)
    show_image("Task 6 - best ORB matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
