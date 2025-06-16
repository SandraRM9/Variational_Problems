import numpy as np
import matplotlib.pyplot as plt

#this simulation implies different values of a to simulate the different forms of the catenary with fixed segments,
#this implies different lengths of the pieces for each case. The more value of 'a', the best case it is  

# Nuevos valores de 'a' y configuración más centrada
a_values = [0.5, 1.0, 2.0, 2.75]
n_segments = 7
x_segment = np.linspace(-1, 1, n_segments + 1)  # rango más pequeño para hacer zoom

plt.figure(figsize=(14, 8))

for i, a in enumerate(a_values, 1):
    y_segment = -a * np.cosh(x_segment / a)
    
    # Calcular longitudes de los segmentos rectos
    lengths = np.sqrt(np.diff(x_segment)**2 + np.diff(y_segment)**2)
    
    # Plot del arco con segmentos rectos
    plt.subplot(2, 2, i)
    for j in range(n_segments):
        plt.plot([x_segment[j], x_segment[j+1]], [y_segment[j], y_segment[j+1]], color='saddlebrown', lw=4)
        plt.text((x_segment[j]+x_segment[j+1])/2,
                 (y_segment[j]+y_segment[j+1])/2 + 0.05,
                 f"{lengths[j]:.2f}", fontsize=9, ha='center', color='black')
    
    plt.title(f'a = {a}', fontsize=12)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")

#plt.suptitle("Cambridge Bridge - Without Catenary, only straight segments", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
