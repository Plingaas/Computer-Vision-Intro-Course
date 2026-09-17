"""Task 4: intensity changes become edges.
Try changing the two Canny thresholds. Which edges disappear first?
"""
import cv2


def open_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}. Run from the repository root.")
    return image


def preprocess_image(image):
    # TODO: Convert to grayscale, then apply a (5, 5) Gaussian blur.
    # Hint: cv2.cvtColor(), cv2.COLOR_BGR2GRAY, cv2.GaussianBlur().
    raise NotImplementedError("Prepare a lightly smoothed grayscale image")


def find_edges(gray_image, low_threshold, high_threshold):
    # TODO: Return cv2.Canny() with the two supplied thresholds.
    raise NotImplementedError("Find edges")


def show_image(name, image):
    # Window setup is supplied so large images fit on the screen.
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    scale = min(900 / width, 600 / height, 1.0)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
    cv2.imshow(name, image)


def display_edges(original, blurred, edges):
    show_image("Task 4 - original", original)
    show_image("Task 4 - blurred grayscale", blurred)
    show_image("Task 4 - Canny edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = open_image("data/task4.png")
    blurred = preprocess_image(image)
    edges = find_edges(blurred, low_threshold=50, high_threshold=150)
    display_edges(image, blurred, edges)


if __name__ == "__main__":
    main()
