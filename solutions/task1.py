"""Teacher solution for Task 1. Run from the repository root."""
import cv2


def open_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}. Run from the repository root.")
    return image


def make_black_and_white(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def display_image(image):
    cv2.namedWindow("Task 1 - grayscale", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Task 1 - grayscale", 600, 700)
    cv2.imshow("Task 1 - grayscale", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = open_image("data/task1.png")
    gray = make_black_and_white(image)
    display_image(gray)


if __name__ == "__main__":
    main()
