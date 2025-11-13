import cv2
import numpy as np
img = cv2.imread("C:/Users/adity/Downloads/lena-2.png")
h, w, c = img.shape

#Task 1 Padding
def padding(image, w):
    # the task of adding border
    padded_image = cv2.copyMakeBorder(image,w,w,w,w,cv2.BORDER_REFLECT)
    #Saving it
    cv2.imwrite("padded_image.jpg", padded_image)
    return padded_image


output1 = padding(img, 100)

#Task 2 Cropping

def crop(image, l,r,t,b):
    # cropping
    cropped_image = image[t:b, l:r]
    # Saving
    cv2.imwrite("cropped_image.jpg", cropped_image)
    return cropped_image

l = 80
t = 80
r = w - 130
b = h - 130
output2 = crop(img,l,r,t,b)


def resize(image, w, h):
    resized_image = cv2.resize(image, (w, h))
    cv2.imwrite("resized_image.jpg", resized_image)
    return resized_image


output3 = resize(img, 200, 200)

def copy(image, emptypicturearray):

    for i in range(h):
        for j in range(w):
            for k in range(c):
                emptypicturearray[i, j, k] = image[i, j, k]

    # Save the copied image
    cv2.imwrite("copied_image.jpg", emptypicturearray)
    return emptypicturearray

# Create empty array
emptypicturearray = np.zeros((h, w, 3), dtype=np.uint8)

# Copy pixels
output4 = copy(img, emptypicturearray)

#Task 5 Grayscaling
def grayscale(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Save the grayscale image
    cv2.imwrite("grayscale_image.jpg", gray)
    return gray

output5 = grayscale(img)

def hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv_image.jpg", hsv)
    return hsv

output6 = hsv(img)

def hue_shifted(image, emptyPictureArray, hue):

    for i in range(h):
        for j in range(w):
            for k in range(c):
                new_val = image[i, j, k] + hue
                emptyPictureArray[i, j, k] = np.clip(new_val, 0, 255)

    cv2.imwrite("hue_shifted.jpg", emptyPictureArray)
    return emptyPictureArray

emptyPictureArray = np.zeros((h, w, 3), dtype=np.uint8)
output7 = hue_shifted(img, emptyPictureArray, 50)

def smoothing(image):

    smooth_image = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite("smoothed_image.jpg", smooth_image)
    return smooth_image

output8 = smoothing(img)

def rotation(image, angle):
    if angle == 90:

        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("rotated_image_90.jpg", rotated_image)
    elif angle == 180:

        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("rotated_image_180.jpg", rotated_image)
    else:

        rotated_image = image


    return rotated_image


output9a = rotation(img, 90)
output9b = rotation(img, 180)



