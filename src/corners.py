# Implemented Harris corner detector?


# --------------------------------------------------------
# Detect corners using goodFeaturesToTrack
# 1. Implementierung eines Corner-Detektors
# Shi-Tomasi algorithm
# https://www.geeksforgeeks.org/python/python-detect-corner-of-an-image-using-opencv/
# --------------------------------------------------------
corners1 = cv2.goodFeaturesToTrack(img1, maxCorners=1000, qualityLevel=0.01, minDistance=7)
corners2 = cv2.goodFeaturesToTrack(img2, maxCorners=1000, qualityLevel=0.01, minDistance=7)

if corners1 is None or corners2 is None:
    raise RuntimeError("No corners found – try lowering qualityLevel or minDistance.")


# Convert corners to KeyPoint objects for ORB descriptor extraction
# https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
kps1 = [cv2.KeyPoint(float(x), float(y), 31) for [[x, y]] in corners1]
kps2 = [cv2.KeyPoint(float(x), float(y), 31) for [[x, y]] in corners2]


# --------------------------------------------------------
# Compute ORB descriptors for those keypoints -> small binary vector describing the local patch around each keypoint
# https://www.geeksforgeeks.org/python/feature-detection-and-matching-with-opencv-python/
# --------------------------------------------------------
orb = cv2.ORB_create(nfeatures=2000)
kps1, desc1 = orb.compute(img1, kps1)
kps2, desc2 = orb.compute(img2, kps2)

print("--------------------------------------------------------------------------------------\n")
print("1. Implementierung eines Corner-Detektors:\n")
print("Keypoints found:", len(kps1), len(kps2))
