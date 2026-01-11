import cv2

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
