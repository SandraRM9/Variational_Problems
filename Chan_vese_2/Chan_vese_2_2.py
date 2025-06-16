import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops

# -----------------------------------------------------------------------------
# 1) Load and normalize the image
# -----------------------------------------------------------------------------
I = imread('forms.png', as_gray=True)
I = (I - I.min()) / (I.max() - I.min())
ny, nx = I.shape

# -----------------------------------------------------------------------------
# 2) Binarize and label connected components (Otsu + labeling)
# -----------------------------------------------------------------------------
thresh = threshold_otsu(I)        # Otsu's global threshold
binary = I > thresh
labels = measure.label(binary)    # label each connected component

# -----------------------------------------------------------------------------
# 3) Chan–Vese helper functions (arctan δ and curvature)
# -----------------------------------------------------------------------------
def delta_arctan(u, eps):
    """Regularized delta (arctan version)."""
    return (1/np.pi) * (eps / (u**2 + eps**2))

def curvature(u):
    """Compute curvature κ = div(∇u/|∇u|)."""
    ux, uy = np.gradient(u)
    norm = np.sqrt(ux**2 + uy**2) + 1e-8
    nx, ny = ux / norm, uy / norm
    nxx, _ = np.gradient(nx)
    _, nyy = np.gradient(ny)
    return nxx + nyy

def chan_vese_step(u, I, eps, Dt, mu, nu, lam1, lam2):
    """One explicit Chan–Vese update + Gaussian regularization."""
    inside  = u >= 0
    outside = ~inside
    # means inside/outside
    c1 = I[inside].mean()   if inside.any()   else 0
    c0 = I[outside].mean()  if outside.any()  else 0
    # curvature term
    kappa = curvature(u)
    # data fidelity
    F = -lam1 * (I - c1)**2 + lam2 * (I - c0)**2
    # update
    u_new = u + Dt * delta_arctan(u, eps) * (mu * kappa - nu + F)
    # smooth
    return ndimage.gaussian_filter(u_new, sigma=0.5)

# -----------------------------------------------------------------------------
# 4) Initialize small circular seeds inside each region
# -----------------------------------------------------------------------------
eps = 1.0
u_list = []
for prop in regionprops(labels):
    cy, cx = prop.centroid
    h = prop.bbox[2] - prop.bbox[0]
    w = prop.bbox[3] - prop.bbox[1]
    radius = 0.25 * min(h, w)   # small circle inside object
    y, x = np.ogrid[:ny, :nx]
    phi = radius - np.sqrt((y - cy)**2 + (x - cx)**2)
    u_list.append(np.tanh(phi / eps))

# -----------------------------------------------------------------------------
# 5) Evolve and capture specific snapshots
# -----------------------------------------------------------------------------
nit = 500
Dt, mu, nu = 0.1, 1.0, 0.0
lam1 = lam2 = 1.0
snap_iters = [0, 100, 200, 230, 500]
snapshots = {k: [] for k in snap_iters}

# run evolution
u_current = [u.copy() for u in u_list]
for k in range(nit + 1):
    if k in snapshots:
        snapshots[k] = [u.copy() for u in u_current]
    if k < nit:
        u_current = [chan_vese_step(u, I, eps, Dt, mu, nu, lam1, lam2)
                     for u in u_current]

# -----------------------------------------------------------------------------
# 6) Plot the evolution contours in red
# -----------------------------------------------------------------------------
for k in snap_iters:
    plt.figure(figsize=(5,5))
    plt.imshow(I, cmap='gray')
    for u_snap in snapshots[k]:
        plt.contour(u_snap, levels=[0], colors='r', linewidths=1)
    plt.axis('off')
    plt.title(f'Iteration {k}')
    plt.show()
