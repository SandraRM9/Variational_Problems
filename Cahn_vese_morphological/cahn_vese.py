import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.segmentation import morphological_chan_vese
from skimage.filters import gaussian
from skimage.color import rgb2gray
from scipy.special import erf


# --------------------------------------
# MEDICAL IMAGE LOADING (CT SCAN)
# --------------------------------------

image_path = "Chron_disease.jpg"

image = img_as_float(io.imread(image_path))

# If RGB, convert to grayscale
if image.ndim == 3:
    image = rgb2gray(image)

# Smoothing to reduce noise
image = gaussian(image, sigma=1.0)

# --------------------------------------
# MORPHOLOGICAL CHAN-VESE SIMULATION
# --------------------------------------

def chan_vese_segmentation(image, iterations=300, init="checkerboard"):
    return morphological_chan_vese(image, num_iter=iterations, init_level_set=init)

# Run segmentation
seg_result = chan_vese_segmentation(image)

# --------------------------------------
# VISUALIZATION
# --------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original Image (CT Scan)")
ax[0].axis("off")

ax[1].imshow(image, cmap="gray")
ax[1].contour(seg_result, [0.5], colors="red")
ax[1].set_title("Detected Contour (Chan-Vese)")
ax[1].axis("off")

plt.tight_layout()
plt.show()
