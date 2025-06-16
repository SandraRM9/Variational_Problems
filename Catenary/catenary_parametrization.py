import numpy as np
import matplotlib.pyplot as plt

# Catenary parameter (larger = flatter curve)
a = 2
x = np.linspace(-3, 3, 400)
y = a * np.cosh(x / a)

# Create figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# === LEFT PLOT ===
ax[0].plot(x, y, color='orange')
ax[0].axvline(0, color='black')  # Y-axis
ax[0].hlines(0, -3, 3, colors='black')  # X-axis

# Vertical lines (the poles)
ax[0].vlines([-3, 3], 0, a * np.cosh(3 / a), colors='lightblue')

# Lowest point A
ax[0].scatter(0, a, color='black')
ax[0].text(0, a + 0.1, 'A(0, h)', ha='center')

# Height line h
ax[0].vlines(0, 0, a, linestyles='dotted', colors='blue')
ax[0].text(0.1, a / 2, 'h', color='blue', va='center')

ax[0].set_xlim(-3.5, 3.5)
ax[0].set_ylim(0, 5)
ax[0].axis('off')

# === RIGHT PLOT ===
x_zoom = np.linspace(0, 3, 200)
y_zoom = a * np.cosh(x_zoom / a)
ax[1].plot(x_zoom, y_zoom, color='orange')
ax[1].axvline(0, color='black')  # Y-axis
ax[1].hlines(0, -1, 3.5, colors='black')  # X-axis

# Right vertical line (pole)
ax[1].vlines(3, 0, a * np.cosh(3 / a), colors='lightblue')

# Force vectors
x0, y0 = 1.5, a * np.cosh(1.5 / a)
ax[1].arrow(x0, y0, 0.5, 0.5, head_width=0.07, color='blue')  # T (tension)
ax[1].arrow(x0, y0, 0.5, 0, head_width=0.07, color='blue')    # T·cos(θ)
ax[1].arrow(x0, y0, 0, 0.5, head_width=0.07, color='blue')    # T·sin(θ)
ax[1].arrow(x0, y0, 0, -0.5, head_width=0.07, color='blue')   # Weight

# Labels
ax[1].text(x0 + 0.55, y0 + 0.55, 'T', color='blue')
ax[1].text(x0 + 0.55, y0, 'T·Cos(θ)', color='blue')
ax[1].text(x0, y0 + 0.55, 'T·Sin(θ)', color='blue')
ax[1].text(x0, y0 - 0.6, 'Weight = λ·s(x)', color='blue')
ax[1].text(-0.5, a, 'T₀', color='blue')

# Arrow for T₀
ax[1].arrow(-0.1, a, -0.4, 0, head_width=0.07, color='blue')

# Height line h
ax[1].vlines(0, 0, a, linestyles='dotted', colors='blue')
ax[1].text(0.1, a / 2, 'h', color='blue', va='center')

ax[1].axis('off')
ax[1].set_xlim(-1, 3.5)
ax[1].set_ylim(0, 5)

plt.tight_layout()
plt.show()
