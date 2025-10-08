import cv2
import os
import numpy as np
import zipfile
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

root_dir = "Extracted_data"

if not os.path.exists(root_dir):
    raise FileNotFoundError('Extracted_data directory not found')

subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
has_dataset = any(d.lower().startswith('same') or d.lower().startswith('different') for d in subdirs)

if not has_dataset:
    raise FileNotFoundError('no same_* or different_* folders found in Extracted_data')

def load_prep(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.GaussianBlur(im, (5, 5), 0)
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return im

def good_match_count(a, b, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=1000)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(d1, d2, k=2)
    cnt = 0
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            cnt += 1
    return cnt

pairs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
y_true = []
y_pred = []
for p in pairs:
    pth = os.path.join(root_dir, p)
    imgs = sorted([os.path.join(pth, f) for f in os.listdir(pth) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp', '.jpeg'))])
    if len(imgs) != 2:
        continue
    a = load_prep(imgs[0])
    b = load_prep(imgs[1])
    matches = good_match_count(a, b)
    label = 1 if p.lower().startswith('same') else 0
    pred = 1 if matches > 20 else 0
    y_true.append(label)
    y_pred.append(pred)
    print(p + ':', matches, 'matches ->', 'SAME' if pred == 1 else 'DIFFERENT')

if not y_true:
    print('no pairs found')
    raise SystemExit

y_true = np.array(y_true)
y_pred = np.array(y_pred)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
disp.plot(cmap='Blues')
plt.title('results')
plt.show()
acc = (y_true == y_pred).mean()
print('accuracy:', f"{acc*100:.2f}%")