import cv2
import numpy as np


def edge_operations(src_img, out_sobel="sobel_edges.jpg", out_canny="canny_edges.jpg",
                    th1=50, th2=50):
    """
    Apply Sobel and Canny edge detection on an image.
    """
    img = cv2.imread(src_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel edges
    sobel_mix = cv2.Sobel(smooth, cv2.CV_64F, 1, 1, ksize=1)
    sobel_final = cv2.convertScaleAbs(sobel_mix)
    cv2.imwrite(out_sobel, sobel_final)
    print(f"[INFO] Sobel result written to {out_sobel}")

    # Canny edges
    canny_final = cv2.Canny(smooth, th1, th2)
    cv2.imwrite(out_canny, canny_final)
    print(f"[INFO] Canny result written to {out_canny}")


def resize_with_pyramid(src_img, levels=2, direction="up", output="resized.jpg"):
    """
    Resize an image using pyramid up/down scaling.
    """
    img = cv2.imread(src_img)

    if direction.lower() == "up":
        for _ in range(levels):
            img = cv2.pyrUp(img)
    elif direction.lower() == "down":
        for _ in range(levels):
            img = cv2.pyrDown(img)
    else:
        raise ValueError("direction must be either 'up' or 'down'")

    cv2.imwrite(output, img)
    print(f"[INFO] Pyramid resize result saved as {output}")


def match_pattern(source_img, small_img, result_name="template_match.jpg"):
    """
    Locate a template image inside a larger one.
    """
    big = cv2.imread(source_img)
    big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    patch = cv2.imread(small_img, cv2.IMREAD_GRAYSCALE)
    w, h = patch.shape[::-1]

    response = cv2.matchTemplate(big_gray, patch, cv2.TM_CCOEFF_NORMED)
    locations = np.where(response >= 0.9)

    for pt in zip(*locations[::-1]):
        cv2.rectangle(big, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite(result_name, big)
    print(f"[INFO] Template match image saved -> {result_name}")


if __name__ == "__main__":
    # Input images
    car_img = "lambo.png"
    shape_img = "shapes-1.png"
    shape_patch = "shapes_template.jpg"

    # Run operations
    edge_operations(car_img, out_sobel="sobel_edges.jpg", out_canny="canny_edges.jpg")
    resize_with_pyramid(car_img, levels=2, direction="up", output="resized.jpg")
    match_pattern(shape_img, shape_patch, result_name="template_match.jpg")
