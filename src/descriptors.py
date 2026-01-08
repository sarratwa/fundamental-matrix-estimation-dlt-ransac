# patch extraction + SSD matcher


# --------------------------------------------------------
# Match descriptors using brute-force matcher
# 2. Korrespondenzanalyse (Matching)
# https://www.geeksforgeeks.org/machine-learning/feature-matching-using-brute-force-in-opencv/
# --------------------------------------------------------
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)
matches = sorted(matches, key=lambda m: m.distance)

print("--------------------------------------------------------------------------------------\n")
print("2. Korrespondenzanalyse (Matching):\n")
print("Total matches:", len(matches))


# --------------------------------------------------------
# Extract coordinates of matched points
# 3. Relaxation – consistency filtering of matches
# Compute disparity
# --------------------------------------------------------

# Convert matches to coordinate arrays (for distance consistency check)
# https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

# Compute disparity (difference in x-coordinates)
disparities = np.abs(pts1[:, 0] - pts2[:, 0])

# Compute mean and std of disparities
mean_disp = np.mean(disparities)
std_disp = np.std(disparities)

# Keep only matches whose disparity lies within 1 standard deviation
relaxed_matches = [m for i, m in enumerate(matches)
                   if abs(disparities[i] - mean_disp) < std_disp]

print("--------------------------------------------------------------------------------------\n")
print("3. Relaxation:\n")
print("Relaxation: kept", len(relaxed_matches), "of", len(matches), "matches")

# Replace matches with relaxed version
matches = relaxed_matches    
