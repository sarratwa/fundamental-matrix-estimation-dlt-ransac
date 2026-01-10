import matplotlib.pyplot as plt
import cv2
import numpy as np

from corners import detect_corners_and_descriptors, shi_tomasi_corners
from descriptors import descriptors, compute_patch_descriptors, match_descriptors_manual
from fmatrix import fmatrix, dlt_fundamental, ransac_fundamental

# function to visualize the epipolar lines
def draw_epipolar_lines(img1, img2, F, kps1, kps2, matches):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)

    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw lines in Image 1
    for r, pt in zip(lines1, pts1):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, x1 = 0, img1.shape[1]
        y0 = int((-r[2] - r[0]*x0) / r[1])
        y1 = int((-r[2] - r[0]*x1) / r[1])
        cv2.line(img1_color, (x0,y0), (x1,y1), color, 1)
        cv2.circle(img1_color, tuple(pt.astype(int)), 5, color, -1)

    # Draw lines in Image 2
    for r, pt in zip(lines2, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, x1 = 0, img2.shape[1]
        y0 = int((-r[2] - r[0]*x0) / r[1])
        y1 = int((-r[2] - r[0]*x1) / r[1])
        cv2.line(img2_color, (x0,y0), (x1,y1), color, 1)
        cv2.circle(img2_color, tuple(pt.astype(int)), 5, color, -1)

    return img1_color, img2_color

# manual pipeline
def run_manual_pipeline(img1, img2):

    # 1. Manual Shi-Tomasi
    kps1 = shi_tomasi_corners(img1, max_corners=800)
    kps2 = shi_tomasi_corners(img2, max_corners=800)
    print("1.Manual Corners:", len(kps1), "/", len(kps2))

    # 2. Manual patch descriptors
    kps1, desc1 = compute_patch_descriptors(img1, kps1)
    kps2, desc2 = compute_patch_descriptors(img2, kps2)

    # 3. SSD + Ratio + Relaxation
    matches = match_descriptors_manual(desc1, desc2, kps1, kps2)
    print("3.Manual matches after Relaxation:", len(matches))

    # Extract match coordinates
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # 4. Manual RANSAC
    F_ransac, inliers = ransac_fundamental(pts1, pts2)
    print("Manual RANSAC inliers:", len(inliers))

    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    # 5. Recompute F using DLT on inliers
    F_final = dlt_fundamental(pts1_in, pts2_in)
    print("Final manual F:\n", F_final)

    # Convert inliers back into matches
    final_matches = [matches[i] for i in inliers]

    return kps1, kps2, F_final, final_matches

# Automatic pipeline
def run_automatic_pipeline(img1, img2):

    # 1. Automatic corner + ORB
    kps1, desc1, kps2, desc2 = detect_corners_and_descriptors(img1, img2)
    print("Auto corners:", len(kps1), "/", len(kps2))

    # 2 + 3: Matching + Relaxation
    matches = descriptors(desc1, desc2, kps1, kps2)
    print("Auto matches after Relaxation:", len(matches))

    # 4-6 Automatic DLT + RANSAC combined
    F_final, inlier_matches = fmatrix(kps1, kps2, matches)
    print("Final automatic F:\n", F_final)

    return kps1, kps2, F_final, inlier_matches


# Main processing script for Fundamental Matrix estimation.
# Follows exactly the steps required in the Übungsblatt:
#   (1) Corner detection
#   (2) Descriptor extraction
#   (3) Descriptor matching + Relaxation
#   (4) Fundamental matrix via DLT (no RANSAC)
#   (5) RANSAC outlier removal
#   (6) Recompute F using all inliers (DLT)
#   (7) Visualization of matches + epipolar lines

if __name__ == "__main__":

    left_path  = "../data/example002L.bmp"
    right_path = "../data/example002R.bmp"

    img1 = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Error loading images")

    print("\nChoose mode:")
    print("1 = Automatic (OpenCV)")
    print("2 = Manual Implementation")
    print("3 = Both (compare)")
    
    mode = input("Your choice: ")

    # automatic pipeline
    if mode == "1":
        kps1, kps2, F, matches = run_automatic_pipeline(img1, img2)

        # Visualize automatic
        epi1, epi2 = draw_epipolar_lines(img1, img2, F, kps1, kps2, matches[:20])
        
        plt.figure(figsize=(14,7))
        plt.subplot(121)
        plt.imshow(epi1)
        plt.title("AUTO — Epipolar Lines (Image 1)")
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(epi2)
        plt.title("AUTO — Epipolar Lines (Image 2)")
        plt.axis('off')
        plt.show()


    # manual
    elif mode == "2":
        kps1, kps2, F, matches = run_manual_pipeline(img1, img2)

        epi1, epi2 = draw_epipolar_lines(img1, img2, F, kps1, kps2, matches[:20])
        
        plt.figure(figsize=(14,7))
        plt.subplot(121)
        plt.imshow(epi1)
        plt.title("MANUAL — Epipolar Lines (Image 1)")
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(epi2)
        plt.title("MANUAL — Epipolar Lines (Image 2)")
        plt.axis('off')
        plt.show()


    # both: manual and automatic
    elif mode == "3":
        print("\nRunning AUTOMATIC pipeline...")
        kps1_a, kps2_a, F_auto, matches_auto = run_automatic_pipeline(img1, img2)

        print("\nRunning MANUAL pipeline...")
        kps1_m, kps2_m, F_manual, matches_manual = run_manual_pipeline(img1, img2)

        print("\n======================")
        print(" COMPARISON RESULTS")
        print("======================")
        print("\nFundamental Matrix (Automatic):\n", F_auto)
        print("\nFundamental Matrix (Manual):\n", F_manual)
        print("\nDifference (F_manual - F_auto):\n", F_manual - F_auto)

        print("\nAutomatic inliers:", len(matches_auto))
        print("Manual inliers:", len(matches_manual))

        # Show automatic epipolar lines
        epi1_a, epi2_a = draw_epipolar_lines(img1, img2, F_auto, kps1_a, kps2_a, matches_auto[:20])
        # Show manual epipolar lines
        epi1_m, epi2_m = draw_epipolar_lines(img1, img2, F_manual, kps1_m, kps2_m, matches_manual[:20])

        plt.figure(figsize=(16,10))
        plt.subplot(221); plt.imshow(epi1_a); plt.title("Auto Epipolar Lines (Img1)"); plt.axis('off')
        plt.subplot(222); plt.imshow(epi2_a); plt.title("Auto Epipolar Lines (Img2)"); plt.axis('off')
        plt.subplot(223); plt.imshow(epi1_m); plt.title("Manual Epipolar Lines (Img1)"); plt.axis('off')
        plt.subplot(224); plt.imshow(epi2_m); plt.title("Manual Epipolar Lines (Img2)"); plt.axis('off')
        plt.show()

    else:
        print("Invalid selection.")