import matplotlib.pyplot as plt
import numpy as np

def get_random_color(r, g, b):
    r = np.random.uniform(r[0], r[1])
    g = np.random.uniform(g[0], g[1])
    b = np.random.uniform(b[0], b[1])
    return (r, g, b)

r_range = (0, 0)
g_range = (0.0, 0.5)
b_range = (0.0, 1.0) 

# Get a random color
random_color = get_random_color(r_range, g_range, b_range)

# Plotting an example using the random color
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color=random_color, label=f'Random Color: {random_color}')
plt.legend()
plt.show()