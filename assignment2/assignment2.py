import cv2
import numpy as np
import os


def read_img(path):
    return cv2.imread(path)


def write_img(result, source_path, tag):
    folder = os.path.dirname(source_path)
    file = os.path.basename(source_path)
    name, ext = os.path.splitext(file)
    if not ext:
        ext = ".png"
    new_file = os.path.join(folder, f"{name}_{tag}{ext}")
    cv2.imwrite(new_file, result)


def to_gray(path):
    img = read_img(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_img(gray_img, path, "gray")


def rotate_img(path, angle):
    img = read_img(path)
    if angle == 90:
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rot = cv2.rotate(img, cv2.ROTATE_180)
    write_img(rot, path, f"rot{angle}")


def scale_img(path, new_w, new_h):
    img = read_img(path)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    write_img(scaled, path, "resize")


def cut_region(path, left, right, top, bottom):
    img = read_img(path)
    h, w = img.shape[:2]
    section = img[top:h - bottom, left:w - right]
    write_img(section, path, "crop")


def blur_img(path):
    img = read_img(path)
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    write_img(blur, path, "smooth")


def duplicate(path, blank):
    img = read_img(path)
    h, w, _ = img.shape
    for r in range(h):
        for c in range(w):
            blank[r, c] = img[r, c]
    write_img(blank, path, "copy")


def shift_hue(path, container, shift_val):
    img = read_img(path)
    rows, cols, ch = img.shape
    for r in range(rows):
        for c in range(cols):
            for k in range(ch):
                container[r, c, k] = (int(img[r, c, k]) + shift_val) % 256
    write_img(container, path, "hueShift")


def add_padding(path, border=50):
    img = read_img(path)
    padded_img = cv2.copyMakeBorder(img, border, border, border, border,
                                    borderType=cv2.BORDER_REFLECT_101)
    write_img(padded_img, path, "pad")


def to_hsv(path):
    img = read_img(path)
    hsv_version = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    write_img(hsv_version, path, "hsv")


if __name__ == "__main__":
    src = r"D:\PROJECTS\ADVIK\assignment2\lena-2.png"

    add_padding(src, 100)
    cut_region(src, 80, 130, 80, 130)
    scale_img(src, 200, 200)

    base_img = read_img(src)
    h, w, c = base_img.shape
    blank_img = np.zeros((h, w, c), dtype=np.uint8)
    duplicate(src, blank_img)

    to_gray(src)
    to_hsv(src)

    shifted = np.zeros((h, w, c), dtype=np.uint8)
    shift_hue(src, shifted, 50)

    blur_img(src)
    rotate_img(src, 90)
    rotate_img(src, 180)
