import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops

# -----------------------------------------------------------------------------
# 1) Load & normalize the CT image
# -----------------------------------------------------------------------------
I = imread('Chron_disease2.jpg', as_gray=True).astype(float)
I = (I - I.min()) / (I.max() - I.min())
ny, nx = I.shape

# -----------------------------------------------------------------------------
# 2) Window‐level and exclude very bright (bone) intensities
# -----------------------------------------------------------------------------
# soft‐tissue window [low, high] in normalized units
low, high = 0.1, 0.6
I_win = np.clip((I - low) / (high - low), 0, 1)

# bone exclusion threshold (normalized)
bone_thresh = 0.9

# -----------------------------------------------------------------------------
# 3) Binarize & label connected components, ignoring bone
# -----------------------------------------------------------------------------
# Otsu on windowed image
thresh = threshold_otsu(I_win)
# only those pixels in soft‐tissue range above Otsu but below bone_thresh
binary = (I_win > thresh) & (I < bone_thresh)
labels = measure.label(binary)

# -----------------------------------------------------------------------------
# 4) Chan–Vese helper functions (arctan δ, curvature)
# -----------------------------------------------------------------------------
def delta_arctan(u, eps):
    return (1/np.pi) * (eps / (u**2 + eps**2))

def curvature(u):
    ux, uy = np.gradient(u)
    norm   = np.sqrt(ux**2 + uy**2) + 1e-8
    nx_, ny_ = ux/norm, uy/norm
    nxx, _   = np.gradient(nx_)
    _, nyy   = np.gradient(ny_)
    return nxx + nyy

def chan_vese_step(u, I, eps, Dt, mu, nu, lam1, lam2):
    inside  = u >= 0
    outside = ~inside
    c1 = I[inside].mean()   if inside.sum()>0   else 0.0
    c0 = I[outside].mean()  if outside.sum()>0  else 0.0
    kappa = curvature(u)
    F     = -lam1*(I-c1)**2 + lam2*(I-c0)**2
    u_new = u + Dt * delta_arctan(u, eps) * (mu*kappa - nu + F)
    return ndimage.gaussian_filter(u_new, sigma=0.5)

# -----------------------------------------------------------------------------
# 5) Initialize seed inside inflamed region, filtering small artifacts
# -----------------------------------------------------------------------------
props = regionprops(labels, intensity_image=I_win)
# keep only regions of reasonable size
candidates = [p for p in props if 500 < p.area < 50000]
if not candidates:
    raise RuntimeError("No suitable regions—adjust area thresholds or bone_thresh.")
# choose the candidate whose mean intensity is highest (inflamed tissue)
cand = max(candidates, key=lambda p: p.mean_intensity)

cy, cx = cand.centroid
h = cand.bbox[2] - cand.bbox[0]
w = cand.bbox[3] - cand.bbox[1]
radius = 0.5 * min(h, w)

eps = 1.0
y, x = np.ogrid[:ny, :nx]
phi0 = radius - np.sqrt((y - cy)**2 + (x - cx)**2)
u = np.tanh(phi0 / eps)

# -----------------------------------------------------------------------------
# 6) Evolve and capture snapshots
# -----------------------------------------------------------------------------
nit = 500
Dt, mu, nu = 0.1, 1.0, 0.0
lam1 = lam2 = 1.0
snap_iters = [0, 100, 150, 200]
snaps = {}

for k in range(nit + 1):
    if k in snap_iters:
        snaps[k] = u.copy()
    if k == nit:
        break
    u = chan_vese_step(u, I_win, eps, Dt, mu, nu, lam1, lam2)

# -----------------------------------------------------------------------------
# 7) Plot evolution
# -----------------------------------------------------------------------------
for k in snap_iters:
    plt.figure(figsize=(6,6))
    # show original CT for context
    plt.imshow(I, cmap='gray')
    # overlay contour of level-set zero
    plt.contour(snaps[k], levels=[0], colors='r', linewidths=2)
    plt.title(f'Iteration {k}')
    plt.axis('off')
    plt.show()
