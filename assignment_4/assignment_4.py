import cv2
import numpy as np
from PIL import Image


def harris_corner_detection(reference_image, output_path='harris.png'):
    img = cv2.imread(reference_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    output = img.copy()
    threshold = 0.01 * dst.max()
    output[dst > threshold] = [0, 0, 255]
    cv2.imwrite(output_path, output)
    return output


def align_images(image_to_align, reference_image, max_features, good_match_percent):
    img_align = cv2.imread(image_to_align)
    img_ref = cv2.imread(reference_image)
    gray_align = cv2.cvtColor(img_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.ORB_create(nfeatures=max_features)
    kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
    kp_align, des_align = detector.detectAndCompute(gray_align, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_ref, des_align)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)
    
    num_good_matches = int(len(matches) * good_match_percent)
    good_matches = matches[:num_good_matches]
    
    matches_img = cv2.drawMatches(
        img_ref, kp_ref, 
        img_align, kp_align, 
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_align = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points_ref[i, :] = kp_ref[match.queryIdx].pt
        points_align[i, :] = kp_align[match.trainIdx].pt
    
    H, mask = cv2.findHomography(points_align, points_ref, cv2.RANSAC)
    height, width = img_ref.shape[:2]
    aligned_img = cv2.warpPerspective(img_align, H, (width, height))
    
    cv2.imwrite('aligned.png', aligned_img)
    cv2.imwrite('matches.png', matches_img)
    
    return aligned_img, matches_img

if __name__ == "__main__":
    harris_corner_detection("reference_img.png", "harris.png")
    align_images("align_this.jpg", "reference_img.png", 1500, 0.15)
