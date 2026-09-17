"""Task 2 (15–20 min): threshold the bright AprilTag and extract a crop.
The supplied contour helper finds the largest bright region in this photo.
We extract the tag's image here; decoding its ID is a different task.
"""
import cv2


def open_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}. Run from the repository root.")
    return image


def make_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def make_binary_mask(gray_image, threshold_value):
    _, mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return mask


def extract_tag(image, mask):
    # Supplied helper: find the largest connected bright outline.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No bright region found. Try lowering the threshold.")
    largest = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest)
    # Add a small border without going outside the image.
    x1, y1 = max(0, x - 4), max(0, y - 4)
    x2, y2 = min(image.shape[1], x + width + 4), min(image.shape[0], y + height + 4)
    return image[y1:y2, x1:x2].copy()


def show_image(name, image):
    # Window setup is supplied so large images fit on the screen.
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    scale = min(900 / width, 600 / height, 1.0)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
    cv2.imshow(name, image)


def display_results(original, mask, tag):
    show_image("Task 2 - original", original)
    show_image("Task 2 - binary mask", mask)
    # Nearest-neighbor enlargement makes the tiny tag's pixels visible.
    enlarged = cv2.resize(tag, (300, 300), interpolation=cv2.INTER_NEAREST)
    show_image("Task 2 - extracted AprilTag", enlarged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = open_image("data/task2.jpg")
    gray = make_grayscale(image)
    mask = make_binary_mask(gray, threshold_value=160)
    tag = extract_tag(image, mask)
    display_results(image, mask, tag)


if __name__ == "__main__":
    main()
