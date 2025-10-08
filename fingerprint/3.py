import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img_path, use_grayscale=False):
    """Preprocess image - option for grayscale or color processing."""
    if use_grayscale:
        # Grayscale processing (good for fingerprints, simple for general images)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        # For general images, just enhance contrast instead of binary threshold
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        # Color processing (better for general photos like UiA images)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        # Convert to LAB color space and enhance L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        # Convert to grayscale for feature detection (but keep color info in preprocessing)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def orb_matching(img1, img2, ratio_thresh=0.75):
    """ORB + BFMatcher approach"""
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return len(good_matches), kp1, kp2, good_matches

def sift_flann_matching(img1, img2, ratio_thresh=0.7):
    """SIFT + FLANN approach"""
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, [], [], []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return len(good_matches), kp1, kp2, good_matches

def compare_both_methods(img1_path, img2_path, match_threshold=20, visualize=True, use_grayscale=False):
    """Compare both ORB and SIFT methods on the same image pair"""
    print(f"Comparing: {img1_path} vs {img2_path}")
    print(f"Processing mode: {'Grayscale' if use_grayscale else 'Color-enhanced'}")
    print("=" * 60)
    
    # Load original color images for visualization
    img1_color = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    img1_color_rgb = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
    img2_color_rgb = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)
    
    # Process images for feature detection
    img1 = preprocess_image(img1_path, use_grayscale)
    img2 = preprocess_image(img2_path, use_grayscale)

    # ORB + BFMatcher
    orb_count, orb_kp1, orb_kp2, orb_matches = orb_matching(img1, img2)
    orb_decision = "SAME" if orb_count > match_threshold else "DIFFERENT"
    
    # SIFT + FLANN  
    sift_count, sift_kp1, sift_kp2, sift_matches = sift_flann_matching(img1, img2)
    sift_decision = "SAME" if sift_count > match_threshold else "DIFFERENT"

    # Results
    print(f"ORB + BFMatcher: {orb_count} matches -> {orb_decision}")
    print(f"SIFT + FLANN: {sift_count} matches -> {sift_decision}")
    print(f"Methods agree: {'YES' if orb_decision == sift_decision else 'NO'}")
    
    # Visualization
    if visualize and img1_color is not None and img2_color is not None:
        plt.figure(figsize=(15, 8))
        
        # ORB results
        plt.subplot(2, 1, 1)
        if orb_matches and orb_kp1 and orb_kp2:
            # Create color output image
            orb_img = cv2.drawMatches(img1_color, orb_kp1, img2_color, orb_kp2, orb_matches[:50], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            orb_img_rgb = cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB)
            plt.imshow(orb_img_rgb)
        else:
            # Fallback: show original images side by side
            combined = np.hstack([img1_color_rgb, img2_color_rgb])
            plt.imshow(combined)
        plt.title(f'ORB + BFMatcher: {orb_count} matches -> {orb_decision}')
        plt.axis('off')
        
        # SIFT results
        plt.subplot(2, 1, 2)
        if sift_matches and sift_kp1 and sift_kp2:
            # Create color output image
            sift_img = cv2.drawMatches(img1_color, sift_kp1, img2_color, sift_kp2, sift_matches[:50], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            sift_img_rgb = cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB)
            plt.imshow(sift_img_rgb)
        else:
            # Fallback: show original images side by side
            combined = np.hstack([img1_color_rgb, img2_color_rgb])
            plt.imshow(combined)
        plt.title(f'SIFT + FLANN: {sift_count} matches -> {sift_decision}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return {
        'orb_matches': orb_count,
        'sift_matches': sift_count,
        'orb_decision': orb_decision,
        'sift_decision': sift_decision,
        'agree': orb_decision == sift_decision
    }

if __name__ == "__main__":
    # Test both preprocessing approaches on UiA images
    print("Testing UiA Images with Different Preprocessing:")
    print("=" * 70)
    
    # Test 1: Color-enhanced preprocessing (better for general photos)
    print("\n1. Color-enhanced preprocessing:")
    results1 = compare_both_methods("UiA front1.png", "UiA front3.jpg", 
                                   match_threshold=20, visualize=True, use_grayscale=False)
    
    # Test 2: Grayscale preprocessing (traditional approach)
    print("\n2. Grayscale preprocessing:")
    results2 = compare_both_methods("UiA front1.png", "UiA front3.jpg", 
                                   match_threshold=20, visualize=True, use_grayscale=True)
    
    # Compare results
    print(f"\n" + "="*70)
    print("COMPARISON SUMMARY:")
    print("="*70)
    print(f"Color-enhanced: ORB={results1['orb_matches']}, SIFT={results1['sift_matches']}")
    print(f"Grayscale:      ORB={results2['orb_matches']}, SIFT={results2['sift_matches']}")
    
    print(f"\nWhich found more matches:")
    print(f"ORB: {'Color-enhanced' if results1['orb_matches'] > results2['orb_matches'] else 'Grayscale'}")
    print(f"SIFT: {'Color-enhanced' if results1['sift_matches'] > results2['sift_matches'] else 'Grayscale'}")
    
    print(f"\nRecommendation for UiA photos: {'Color-enhanced' if (results1['orb_matches'] + results1['sift_matches']) > (results2['orb_matches'] + results2['sift_matches']) else 'Grayscale'} preprocessing")