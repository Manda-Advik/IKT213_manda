import cv2
import os
import numpy as np
import zipfile
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

base_folder = "Extracted_data"

if not os.path.exists(base_folder):
    raise FileNotFoundError('Extracted_data directory not found')

subdirs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
has_dataset = any(d.lower().startswith('same') or d.lower().startswith('different') for d in subdirs)

if not has_dataset:
    raise FileNotFoundError('no same_* or different_* folders found in Extracted_data')

def load_image(filepath):
    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f"failed to load {filepath}")
    im = cv2.GaussianBlur(im, (5, 5), 0)
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return im

def compute_similarity(first, second, ratio=0.7):
    detector = cv2.SIFT_create(nfeatures=1000)
    pts1, desc1 = detector.detectAndCompute(first, None)
    pts2, desc2 = detector.detectAndCompute(second, None)
    
    if desc1 is None or desc2 is None:
        return 0
    
    idx_cfg = dict(algorithm=1, trees=5)
    search_cfg = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(idx_cfg, search_cfg)
    all_matches = matcher.knnMatch(desc1, desc2, k=2)
    
    valid_count = 0
    for pair in all_matches:
        if len(pair) == 2:
            best, second_best = pair
            if best.distance < ratio * second_best.distance:
                valid_count += 1
    
    return valid_count

directories = [d for d in sorted(os.listdir(base_folder)) if os.path.isdir(os.path.join(base_folder, d))]
actual = []
predicted = []

for d in directories:
    d_path = os.path.join(base_folder, d)
    images = sorted([os.path.join(d_path, f) for f in os.listdir(d_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))])
    
    if len(images) != 2:
        continue
    
    try:
        img_a = load_image(images[0])
        img_b = load_image(images[1])
        score = compute_similarity(img_a, img_b)
        
        truth = 1 if d.lower().startswith('same') else 0
        guess = 1 if score > 20 else 0
        
        actual.append(truth)
        predicted.append(guess)
        
        print(f"{d}: {score} matches -> {'SAME' if guess else 'DIFFERENT'}")
    except Exception as e:
        print(f"error processing {d}: {e}")

if not actual:
    print('no valid pairs processed')
else:
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    matrix = confusion_matrix(actual, predicted)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Different", "Same"])
    display.plot(cmap="Blues")
    plt.title("SIFT+FLANN Results")
    plt.show()
    
    accuracy = (actual == predicted).mean()
    print(f"accuracy: {accuracy * 100:.2f}%")