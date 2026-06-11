import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 100, 10000)
r = theta**2

x = r * np.cos(theta)
y = r * np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.axis('equal')
plt.axis('off')
plt.show()