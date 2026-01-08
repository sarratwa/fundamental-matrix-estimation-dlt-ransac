# DLT + Normalized DLT + RANSAC


# --------------------------------------------------------
# 4. Berechnung der Fundamentalmatrix F mittels DLT- oder Normalized-DLT-Algorithmus
# https://www.geeksforgeeks.org/python/python-opencv-epipolar-geometry/
# --------------------------------------------------------

# Extract coordinates of the current (relaxed) matches
pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

# Estimate the Fundamental Matrix using the 8-point / DLT method
# (no RANSAC, purely algebraic)
F_dlt, mask_dlt = cv2.findFundamentalMat(
    pts1, pts2,
    method=cv2.FM_8POINT
)

print("--------------------------------------------------------------------------------------\n")
print("4. Berechnung der Fundamentalmatrix F mittels DLT- oder Normalized-DLT-Algorithmus:\n")
print("Fundamental Matrix (DLT):\n", F_dlt)

# --------------------------------------------------------
# 5. Reduzieren Sie weitere Ausreißer durch RANSAC (automatisiert mit OpenCV)
# https://www.geeksforgeeks.org/python/step-by-step-guide-to-using-ransac-in-opencv-using-python/
# --------------------------------------------------------

# Extract coordinates of remaining matches
pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

# Estimate Fundamental Matrix with RANSAC built into OpenCV
# This internally performs the 8-point DLT and rejects outliers.
F, mask = cv2.findFundamentalMat(
    pts1, pts2,
    method=cv2.FM_RANSAC,
    ransacReprojThreshold=1.0,
    confidence=0.99
)

print("--------------------------------------------------------------------------------------\n")
print("5. Reduzieren Sie weitere Ausreißer durch RANSAC (automatisiert mit OpenCV):\n")
print("Fundamental Matrix (RANSAC via OpenCV):\n", F)
print("Number of inliers:", np.sum(mask), "out of", len(mask))

# Keep only inlier matches for later steps
matches_inliers = [m for i, m in enumerate(matches) if mask[i]]
pts1_in = pts1[mask.ravel() == 1]
pts2_in = pts2[mask.ravel() == 1]

# --------------------------------------------------------
# 6. Recompute Fundamental Matrix using all inliers (DLT)
# 6. Bestimmen Sie F neu mittels DLT-Algorithmus aus 4. unter Verwendung aller Inlier
# --------------------------------------------------------

# Use only inlier correspondences to recompute F from RANSAC
F_final, mask_final = cv2.findFundamentalMat(
    pts1_in, pts2_in,
    method=cv2.FM_8POINT
)

print("--------------------------------------------------------------------------------------\n")
print("6. Bestimmen Sie F neu mittels DLT-Algorithmus aus 4. unter Verwendung aller Inlier:\n")
print("Fundamental Matrix recomputed from all inliers:\n", F_final)


# --------------------------------------------------------
# Visualize matches 
# 7. Stellen Sie die Ergebnisse graphisch dar (Epipolarlinien und Punktkorrespondenzen in Beiden Bildern).
# --------------------------------------------------------
matched_img = cv2.drawMatches(img1, kps1, img2, kps2, matches[:50], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(matched_img, cmap='gray')
plt.title("Top 50 Feature Matches")
plt.axis('off')
plt.show()  

# sollen wir die Algorithmen manuell implementieren - schritt 4 und 5 