"""Task 3: reduce noise with Gaussian blur.
Compare kernels (3, 3) and (11, 11). More blur removes noise AND detail.
"""
import cv2


def open_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}. Run from the repository root.")
    return image


def blur_image(image, kernel_size):
    # TODO: Apply cv2.GaussianBlur() using kernel_size and sigmaX=0.
    # Kernel dimensions must be positive odd numbers.
    raise NotImplementedError("Blur the image")


def show_image(name, image):
    # Window setup is supplied so large images fit on the screen.
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]
    scale = min(900 / width, 600 / height, 1.0)
    cv2.resizeWindow(name, int(width * scale), int(height * scale))
    cv2.imshow(name, image)


def display_comparison(original, small_blur, large_blur):
    show_image("Task 3 - noisy", original)
    show_image("Task 3 - small kernel", small_blur)
    show_image("Task 3 - large kernel", large_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = open_image("data/task3.png")
    small_blur = blur_image(image, (3, 3))
    large_blur = blur_image(image, (11, 11))
    display_comparison(image, small_blur, large_blur)


if __name__ == "__main__":
    main()
