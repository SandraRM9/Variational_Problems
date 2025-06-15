import numpy as np
from scipy import ndimage
from scipy.special import expit
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray


# --- Definition of functions H and δ ---

def H_arctan(u, eps):
    # H(u) = 1/2 + (1/π) arctan(u/ε)
    return 0.5 + (1/np.pi)*np.arctan(u/eps)

def delta_arctan(u, eps):
    # δ(u) = (1/π) (ε / (u^2 + ε^2))
    return (1/np.pi)*(eps/(u**2 + eps**2))

def H_sigmoid(u, eps):
    """
    Smooth Heaviside using sigmoid:
    H(u) = 1 / (1 + exp(-u/ε))
    """
    return expit(u / eps)

def delta_sigmoid(u, eps):
    """
    Derivative of sigmoid using expit for stability:
    δ(u) = σ(u/ε) * (1 - σ(u/ε)) / ε
    """
    s = expit(u / eps)
    return (s * (1 - s)) / eps

def H_tanh(u, eps):
    # H(u) = ½ [1 + tanh(u/ε)]
    return 0.5 * (1 + np.tanh(u / eps))

def delta_tanh(u, eps):
    # δ(u) = (1 / (2ε)) sech²(u/ε)
    return (1 / (2 * eps)) * (1 / np.cosh(u / eps)**2)

def H_poly(u, eps):
    # Compact polynomial: 0 if x < −ε / 1 if x > ε
    # in [−ε, ε]: ½[1 + x/ε + (1/π) sin(πx/ε)]
    H = np.zeros_like(u)
    mask1 = u > eps
    mask2 = np.abs(u) <= eps
    H[mask1] = 1.0
    H[mask2] = 0.5 * (1 + u[mask2] / eps + (1 / np.pi) * np.sin(np.pi * u[mask2] / eps))
    return H

def delta_poly(u, eps):
    # Derivative of the compact polynomial inside [−ε, ε]
    D = np.zeros_like(u)
    mask = np.abs(u) <= eps
    D[mask] = (1 / (2 * eps)) * (1 + np.cos(np.pi * u[mask] / eps))
    return D


# Chan–Vese PDE evolution function
def chan_vese_pde(I, u0, num_iter, eps, Dt, mu, nu, lam1, lam2, delta):
    
    u = u0.copy().astype(float)  # Copy the initial u0 and convert to float for further calculations

    for _ in range(num_iter):
        
        # 1) Mean intensities
        inside  = u >= 0
        outside = u < 0

        sum_in, count_in   = I[inside].sum(),  inside.sum()
        sum_out, count_out = I[outside].sum(), outside.sum()
        c1 = sum_in  / float(count_in  + 1e-8)
        c0 = sum_out / float(count_out + 1e-8)
        
        # 2) Curvature κ = div(∇u/|∇u|)
        ux, uy = np.gradient(u)
        norm = np.sqrt(ux**2 + uy**2) + 1e-8
        nx, ny = ux / norm, uy / norm
        kappa = np.gradient(nx)[0] + np.gradient(ny)[1]

        # 3) Image-based term
        F = -lam1 * (I - c1)**2 + lam2 * (I - c0)**2

        # 4) Explicit evolution step
        u += Dt * delta(u, eps) * (mu * kappa - nu + F)

        # (Optional) re-regularize u, e.g., with a light Gaussian filter
        u = ndimage.gaussian_filter(u, sigma=0.5)

    return u

# Initialize a level set as a centered circle
def initialize_level_set(shape, radius=None):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    if radius is None: 
        radius = min(shape) // 4
    center = (shape[0] // 2, shape[1] // 2)
    mask = (rr - center[0])**2 + (cc - center[1])**2 < radius**2
    u0 = np.where(mask, 1.0, -1.0)
    return u0


# 1) Load and normalize images
I_circle  = imread('circle_image.png', as_gray=True)
I_hexagon = imread('hexagon_image.png', as_gray=True)
I_square= imread('square_image.png', as_gray=True)


# 2) Initialize level sets
u0_circle  = initialize_level_set(I_circle.shape)
u0_hexagon = initialize_level_set(I_hexagon.shape)
u0_square  = initialize_level_set(I_square.shape)

# 3) Parameters
eps   = 1.0
Dt    = 0.1
mu    = 1.0
nu    = 0.0
lam1  = lam2 = 1.0
nit   = 200

# Visualization

# Dictionary of delta function variants
variants = {
    'Arctan':     delta_arctan,
    'Sigmoid':    delta_sigmoid,
    'Tanh':       delta_tanh,
    'Polynomial': delta_poly
}

# --- Main loop: segmentation and visualization ---
for name, delta_fn in variants.items():

    # 1) Segmentation using each δ (implicitly using the corresponding H)
    u_circle  = chan_vese_pde(I_circle,  u0_circle,  nit, eps, Dt, mu, nu, lam1, lam2, delta_fn)
    u_hexagon = chan_vese_pde(I_hexagon, u0_hexagon, nit, eps, Dt, mu, nu, lam1, lam2, delta_fn)
    u_square  = chan_vese_pde(I_square,  u0_square,  nit, eps, Dt, mu, nu, lam1, lam2, delta_fn)

    # Visualize initial and final result for each image
    for title, I, u in [
        (f'{name} Init (circle)',   I_circle,  u0_circle),
        (f'{name} Result (circle)', I_circle,  u_circle),
        (f'{name} Init (hexagon)',  I_hexagon, u0_hexagon),
        (f'{name} Result (hexagon)',I_hexagon, u_hexagon),
        (f'{name} Init (square)',   I_square,  u0_square),
        (f'{name} Result (square)', I_square,  u_square)
    ]:
        plt.figure()
        plt.imshow(I, cmap='gray')
        plt.contour(u, levels=[0], colors='r')
        plt.title(title)
        plt.axis('off')
        plt.show()