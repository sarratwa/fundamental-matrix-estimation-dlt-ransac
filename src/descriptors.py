import cv2
import numpy as np

# This function matches ORB descriptors using Hamming distance and returns a sorted list of matches

def descriptors(desc1, desc2, kps1, kps2):

    # patch extraction + SSD matcher
    # Match descriptors using brute-force matcher
    # 2. Korrespondenzanalyse (Matching)
    # https://www.geeksforgeeks.org/machine-learning/feature-matching-using-brute-force-in-opencv/

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)

    print("2. Korrespondenzanalyse (Matching), Total matches: ", len(matches))

    # Extract coordinates of matched points
    # 3. Relaxation – consistency filtering of matches
    # Compute disparity    
    # Convert matches to coordinate arrays (for distance consistency check)
    # https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
    
    # Extract coordinate arrays from keypoints
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # Compute disparity (x1 - x2)
    disparity = np.abs(pts1[:, 0] - pts2[:, 0])

    # Compute mean and standard deviation
    mean_disp = np.mean(disparity)
    std_disp  = np.std(disparity)
    
    # Keep only matches that lie within mean ± 1*std
    relaxed_matches = [
        m for i, m in enumerate(matches)
        if abs(disparity[i] - mean_disp) < std_disp
    ]
    
    # Safety fallback: If relaxation deleted all matches
    if len(relaxed_matches) < 20:
        print("WARNING: Relaxation removed too many matches → using original matches")
        return matches

    print("3. Relaxation: matches kept", len(relaxed_matches), "of", len(matches), "matches")

    # Replace matches with relaxed version
    return relaxed_matches

# manual patch descriptors
def compute_patch_descriptors(img, kps, radius=7):
    h, w = img.shape
    descs = []
    valid_kps = []

    patch_size = 2*radius + 1

    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        if x-radius < 0 or y-radius < 0 or x+radius >= w or y+radius >= h:
            continue
        
        patch = img[y-radius:y+radius+1, x-radius:x+radius+1]
        vec = patch.astype(np.float32).flatten()

        # Normalize / Normalisieren
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)

        descs.append(vec)
        valid_kps.append(kp)

    return valid_kps, np.vstack(descs)

def ssd(vec1, vec2):
    diff = vec1 - vec2
    return np.dot(diff, diff)

def match_descriptors_manual(desc1, desc2, kps1, kps2, ratio=0.8):
    matches = []

    for i, d1 in enumerate(desc1):
        # Compute SSD to all descriptors in image 2
        dists = np.array([ssd(d1, d2) for d2 in desc2])

        # Best and second-best
        best = np.argmin(dists)
        sorted_idx = np.argsort(dists)
        best_dist = dists[sorted_idx[0]]
        second_dist = dists[sorted_idx[1]]

        # Ratio test
        if best_dist < ratio*second_dist:
            match = cv2.DMatch(i, best, best_dist)
            matches.append(match)

    # RELAXATION STEP
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    disp = np.abs(pts1[:, 0] - pts2[:, 0])
    m = disp.mean()
    s = disp.std()

    relaxed = [matches[i] for i in range(len(matches)) if abs(disp[i] - float(np.mean(disp))) < s]

    return relaxed

