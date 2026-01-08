# loads images, runs full pipeline, plots results

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Paths to the input stereo images
left_path  = r"../data/example002L.bmp"
right_path = r"../data/example002R.bmp"

# Load and convert to grayscale
# ORB basiert auf Helligkeitsunterschieden und nicht auf Farbinformationen
img1_pil = Image.open(left_path).convert("L")   # "L" = grayscale
img2_pil = Image.open(right_path).convert("L")

img1 = np.array(img1_pil)
img2 = np.array(img2_pil)

print("Shapes:", img1.shape, img2.shape)

# corner detection

# match detectors




