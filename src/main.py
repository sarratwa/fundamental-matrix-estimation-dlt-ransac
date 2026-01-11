import matplotlib.pyplot as plt
import cv2
import numpy as np

from corners import detect_corners_and_descriptors
from descriptors import descriptors
from fmatrix import dlt_fundamental, ransac_fundamental

# Draw epipolar lines
def draw_epipolar_lines(img1, img2, F, kps1, kps2, matches):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw lines on image 1
    for r, pt in zip(lines1, pts1):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, x1 = 0, img1.shape[1]
        y0 = int((-r[2] - r[0]*x0) / r[1])
        y1 = int((-r[2] - r[0]*x1) / r[1])
        cv2.line(img1_color, (x0,y0), (x1,y1), color, 1)
        cv2.circle(img1_color, tuple(pt.astype(int)), 4, color, -1)

    # Draw lines on image 2
    for r, pt in zip(lines2, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, x1 = 0, img2.shape[1]
        y0 = int((-r[2] - r[0]*x0) / r[1])
        y1 = int((-r[2] - r[0]*x1) / r[1])
        cv2.line(img2_color, (x0,y0), (x1,y1), color, 1)
        cv2.circle(img2_color, tuple(pt.astype(int)), 4, color, -1)

    return img1_color, img2_color

# Automatic pipeline (ONLY F is manual)

def run_pipeline(img1, img2):
    print("=== Automatic features + Manual/Auto F ===")

    # 1. Automatic Corner Detection + ORB Descriptors
    kps1, desc1, kps2, desc2 = detect_corners_and_descriptors(img1, img2)
    print("Corners:", len(kps1), "/", len(kps2))

    # 2. ORB Matching + Relaxation
    matches = descriptors(desc1, desc2, kps1, kps2)
    print("Matches after relaxation:", len(matches))

    # Extract matched coordinates
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # AUTO FUNDAMENTAL MATRIX
    F_auto, mask_auto = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99
    )
    inlier_matches_auto = [m for i, m in enumerate(matches) if mask_auto[i] == 1]
    print("Auto RANSAC inliers:", len(inlier_matches_auto))

    # MANUAL FUNDAMENTAL MATRIX
    F_ransac, inliers = ransac_fundamental(pts1, pts2)
    print("Manual RANSAC inliers:", len(inliers))

    F_manual = dlt_fundamental(pts1[inliers], pts2[inliers])
    inlier_matches_manual = [matches[i] for i in inliers]

    return (kps1, kps2, F_auto, inlier_matches_auto,
            F_manual, inlier_matches_manual)


# MAIN
if __name__ == "__main__":

    img1 = cv2.imread("../data/example002L.bmp", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../data/example002R.bmp", cv2.IMREAD_GRAYSCALE)

    (kps1, kps2,
     F_auto, matches_auto,
     F_manual, matches_manual) = run_pipeline(img1, img2)

    print("\nFinal AUTO F:\n", F_auto)
    print("\nFinal MANUAL F:\n", F_manual)

    # Draw epipolar lines
    epi1_auto, epi2_auto = draw_epipolar_lines(img1, img2, F_auto, kps1, kps2, matches_auto[:20])
    epi1_manual, epi2_manual = draw_epipolar_lines(img1, img2, F_manual, kps1, kps2, matches_manual[:20])

    # SIDE-BY-SIDE PLOTTING
    plt.figure(figsize=(16,10))

    plt.subplot(221); plt.imshow(epi1_auto);   plt.title("AUTO F — Image 1"); plt.axis("off")
    plt.subplot(222); plt.imshow(epi2_auto);   plt.title("AUTO F — Image 2"); plt.axis("off")

    plt.subplot(223); plt.imshow(epi1_manual); plt.title("MANUAL F — Image 1"); plt.axis("off")
    plt.subplot(224); plt.imshow(epi2_manual); plt.title("MANUAL F — Image 2"); plt.axis("off")

    plt.show()
