import cv2
import numpy as np

# Compute the Fundamental Matrix using OpenCV RANSAC

def fmatrix(kps1, kps2, matches):

    # DLT + Normalized DLT + RANSAC
    # 4. Berechnung der Fundamentalmatrix F mittels DLT- oder Normalized-DLT-Algorithmus
    # https://www.geeksforgeeks.org/python/python-opencv-epipolar-geometry/

    # Convert KeyPoint objects to coordinate arrays
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    # 4.Compute F using DLT (NO RANSAC)
    # Automatic RANSAC F estimation
    F_dlt, mask_dlt = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_8POINT  # pure algebraic DLT
    )

    print("4.Berechnung der Fundamentalmatrix F mittels DLT- oder Normalized-DLT Algorithmus: Fundamental Matrix (DLT):\n", F_dlt)


    # 5. Reduzieren Sie weitere Ausreißer durch RANSAC (automatisiert mit OpenCV)
    # https://www.geeksforgeeks.org/python/step-by-step-guide-to-using-ransac-in-opencv-using-python/

    F_ransac, mask_ransac = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99
    )

    print("5.Reduzieren Sie weitere Ausreißer durch Implementierung des RANSAC-Algorithmus :")
    print("Fundamental Matrix (RANSAC):\n", F_ransac)
    print("Inliers found:", np.sum(mask_ransac), "/", len(mask_ransac))

    # Extract inlier correspondences
    inlier_pts1 = pts1[mask_ransac.ravel() == 1]
    inlier_pts2 = pts2[mask_ransac.ravel() == 1]

    inlier_matches = [m for i, m in enumerate(matches) if mask_ransac[i] == 1]

    # 6. Recompute Fundamental Matrix using all inliers (DLT)
    # 6. Bestimmen Sie F neu mittels DLT-Algorithmus aus 4. unter Verwendung aller Inlier

    # Use only inlier correspondences to recompute F from RANSAC
    F_final, mask_final = cv2.findFundamentalMat(
        inlier_pts1,
        inlier_pts2,
        method=cv2.FM_8POINT  # DLT again
    )

    print("6.Bestimmen Sie F neu mittels DLT-Algorithmus aus 4. unter Verwendung aller Inlier :")
    print("Final Fundamental Matrix:\n", F_final)

    return F_final, inlier_matches

# Manual Normalized 8-Point Algorithm (DLT)
def normalize_points(pts):
    mean = pts.mean(axis=0)
    std = np.sqrt(((pts - mean)**2).sum(axis=1)).mean()

    scale = np.sqrt(2) / std
    T = np.array([
        [scale, 0, -scale*mean[0]],
        [0, scale, -scale*mean[1]],
        [0, 0, 1]
    ])

    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_h.T).T[:, :2]

    return pts_norm, T

def dlt_fundamental(pts1, pts2):
    # Normalize points
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    # Build matrix A
    A = []
    for (x1, y1), (x2, y2) in zip(pts1_n, pts2_n):
        A.append([x1*x2, x1*y2, x1,
                  y1*x2, y1*y2, y1,
                  x2, y2, 1])
    A = np.array(A)

    # Solve Af = 0 using SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)

    # Enforce rank-2
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize
    F = T2.T @ F @ T1

    return F / F[2,2]

# manual RANSAC
def sampson_error(F, pts1, pts2):
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])

    Fx1 = F @ pts1_h.T
    Ftx2 = F.T @ pts2_h.T

    denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
    num = (pts2_h * (F @ pts1_h.T).T).sum(axis=1)**2

    return num / denom

def ransac_fundamental(pts1, pts2, iterations=2000, threshold=1e-3):
    best_inliers = []
    best_F = None

    N = len(pts1)

    for _ in range(iterations):
        idx = np.random.choice(N, 8, replace=False)
        F = dlt_fundamental(pts1[idx], pts2[idx])

        errors = sampson_error(F, pts1, pts2)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F

    return best_F, best_inliers