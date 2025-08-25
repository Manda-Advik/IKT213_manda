import cv2

def print_image_information(image):
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size: {image.size}")
    print(f"Data type: {image.dtype}")

if __name__ == "__main__":
    img = cv2.imread("lena-1.png")
    print_image_information(img)
