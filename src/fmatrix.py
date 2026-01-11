import numpy as np

#  Normalization for DLT

def normalize_points(pts):
    mean = pts.mean(axis=0)
    std = np.sqrt(((pts - mean)**2).sum(axis=1)).mean()

    scale = np.sqrt(2) / std
    
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])

    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_h.T).T[:, :2]

    return pts_norm, T


#  Manual Normalized DLT

def dlt_fundamental(pts1, pts2):
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    A = []
    for (x1, y1), (x2, y2) in zip(pts1_n, pts2_n):
        A.append([
            x1*x2, x1*y2, x1,
            y1*x2, y1*y2, y1,
            x2,     y2,    1
        ])

    A = np.array(A)

    # Solve Af=0 using SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce rank 2
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize
    F = T2.T @ F @ T1
    return F / F[2, 2]


#  Manual RANSAC

def sampson_error(F, pts1, pts2):
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])

    Fx1 = F @ pts1_h.T
    Ftx2 = F.T @ pts2_h.T

    num = (pts2_h * (F @ pts1_h.T).T).sum(axis=1)**2
    denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2

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

#  Public function

def fmatrix(kps1, kps2, matches):
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    print("Running manual RANSAC...")
    F_ransac, inliers = ransac_fundamental(pts1, pts2)
    print("Manual RANSAC Inliers:", len(inliers))

    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    print("Recomputing F using DLT on inliers...")
    F_final = dlt_fundamental(pts1_in, pts2_in)

    # return F and match subset
    final_matches = [matches[i] for i in inliers]

    return F_final, final_matches
