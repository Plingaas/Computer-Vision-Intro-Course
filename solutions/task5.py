"""Task 5 (10–20 min): Shi–Tomasi corners.
Why is a corner easier to locate again than a flat patch or a straight edge?
"""
import cv2


def open_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}. Run from the repository root.")
    return image


def find_corners(gray_image):
    return cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10)


def draw_corners(image, corners):
    result = image.copy()
    if corners is None:
        return result
    # OpenCV returns shape (N, 1, 2). This loop supplies integer (x, y).
    for point in corners:
        x, y = map(int, point.ravel())
        cv2.circle(result, (x, y), 4, (0, 255, 0), -1)
    return result


def show_image(name, image):
    # Window setup is supplied so large images fit on the screen.
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    scale = min(900 / width, 600 / height, 1.0)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
    cv2.imshow(name, image)


def main():
    image = open_image("data/task5.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = find_corners(gray)
    result = draw_corners(image, corners)
    show_image("Task 5 - corners", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
