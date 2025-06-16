import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
import matplotlib.pyplot as plt

g = 9.81
x2, y2 = 1, 0.65

def cycloid(x2, y2, N=300):
    def f(theta):
        return y2/x2 - (1 - np.cos(theta)) / (theta - np.sin(theta))
    theta2 = newton(f, np.pi/2)
    R = y2 / (1 - np.cos(theta2))
    theta = np.linspace(0, theta2, N)
    x = R * (theta - np.sin(theta))
    y = R * (1 - np.cos(theta))
    T = theta2 * np.sqrt(R / g)
    return x, y, T

def linear(x2, y2, N=300):
    m = y2 / x2
    x = np.linspace(0, x2, N)
    y = m * x
    T = np.sqrt(2 * (1 + m**2) * x2 / (g * m))
    return x, y, T

def func(x, f, fp):
    return np.sqrt((1 + fp(x)**2) / (2 * g * f(x)))

def circle(x2, y2, N=300):
    r = (x2**2 + y2**2) / (2 * x2)
    def f(x): return np.sqrt(2 * r * x - x**2)
    def fp(x): return (r - x) / f(x)
    x = np.linspace(0, x2, N)
    y = f(x)
    T = quad(func, 0, x2, args=(f, fp))[0]
    return x, y, T

def parabola(x2, y2, N=300):
    c = (y2 / x2)**2
    def f(x): return np.sqrt(c * x)
    def fp(x): return c / (2 * f(x))
    x = np.linspace(0, x2, N)
    y = f(x)
    T = quad(func, 0, x2, args=(f, fp))[0]
    return x, y, T

# Obtener curvas
curves = {}
for name in ('cycloid', 'linear', 'circle', 'parabola'):
    x, y, T = globals()[name](x2, y2)
    curves[name] = {'x': x, 'y': y, 'T': T}

# Tiempo de referencia: tiempo total de la cicloide
T_ref = curves['cycloid']['T']
progresos = [0.0, 0.25, 0.5, 0.75, 1.0]
tiempos = [p * T_ref for p in progresos]

# Crear imágenes para cada instante
for i, t in enumerate(tiempos):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Dibujar las trayectorias
    for name, data in curves.items():
        ax.plot(data['x'], data['y'], label=f'{name.capitalize()} – {data["T"]:.6f}s', lw=2.5)

    # Dibujar posición de cada bola en ese instante
    for name, data in curves.items():
        T = data['T']
        x = data['x']
        y = data['y']
        if t <= T:
            idx = min(int((t / T) * len(x)), len(x) - 1)
        else:
            idx = -1  # ya llegó
        ax.plot(x[idx], y[idx], 'o', markersize=8, label=f'{name} ball')

    ax.set_title(f'Tiempo t = {t:.2f} s')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.8, 0)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'frame_{i}.png', dpi=300)
    plt.close()

print("¡Listo! Se han guardado 5 imágenes con las posiciones sincronizadas al tiempo de la cicloide.")
