import cv2
import numpy as np


# This function detects corner features in two images using OpenCV goodFeaturesToTrack (Shi–Tomasi) 
# and computes ORB descriptors for them

def detect_corners_and_descriptors(img1, img2):
    
    # 1. corner setection

    # Detect strong corners using Shi–Tomasi. These points 
    # are stable feature locations for matching

    corners1 = cv2.goodFeaturesToTrack(img1, maxCorners=1500, qualityLevel=0.01, minDistance=7)
    corners2 = cv2.goodFeaturesToTrack(img2, maxCorners=1500, qualityLevel=0.01, minDistance=7)

    if corners1 is None or corners2 is None:
        raise RuntimeError("No corners found in one of the images. Try lowering qualityLevel.")

    # Convert coordinates into OpenCV KeyPoint objects
    kps1 = [cv2.KeyPoint(float(x), float(y), 31) for [[x, y]] in corners1]
    kps2 = [cv2.KeyPoint(float(x), float(y), 31) for [[x, y]] in corners2]

    # 2. ORB Descriptors
    
    # ORB creates rotation-invariant, binary descriptors.

    orb = cv2.ORB_create(nfeatures=2000)

    kps1, desc1 = orb.compute(img1, kps1)
    kps2, desc2 = orb.compute(img2, kps2)

    return kps1, desc1, kps2, desc2


# This function implements a Manual Shi–Tomasi Corner Detector 

def shi_tomasi_corners(
    img,
    max_corners=500,
    quality_level=0.01,
    min_distance=7,
    window_size=3
    ):
    
    img_f = np.float32(img)

    # 1. Compute gradients
    Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

    # 2. Structure tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # 3. Smooth with Gaussian window
    Sxx = cv2.GaussianBlur(Ixx, (window_size*2+1, window_size*2+1), 1)
    Syy = cv2.GaussianBlur(Iyy, (window_size*2+1, window_size*2+1), 1)
    Sxy = cv2.GaussianBlur(Ixy, (window_size*2+1, window_size*2+1), 1)

    # 4. Compute eigenvalues of the structure tensor
    # λ = trace/2 ± sqrt( (trace/2)^2 - det )
    trace = Sxx + Syy
    det   = Sxx * Syy - Sxy * Sxy
    sqrt_term = np.sqrt(np.maximum(0, trace*trace/4 - det))

    lambda1 = trace/2 + sqrt_term
    lambda2 = trace/2 - sqrt_term

    # 5. Shi–Tomasi corner score
    R = np.minimum(lambda1, lambda2)

    # 6. Normalize for thresholding
    R_norm = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)

    # 7. Threshold
    thresh = quality_level * R_norm.max()
    candidates = np.argwhere(R_norm > thresh)  # (y, x)

    # Sort strongest corners first
    strengths = R_norm[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-strengths)
    candidates = candidates[order]

    keypoints = []
    selected = []

    for y, x in candidates:
        # enforce min distance
        if all((x - sx)**2 + (y - sy)**2 >= min_distance**2 for sx, sy in selected):
            keypoints.append(cv2.KeyPoint(float(x), float(y), window_size*2+1))
            selected.append((x, y))
            if len(keypoints) >= max_corners:
                break

    return keypoints