import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)  # x values from -10 to 10

# Equations of the lines
y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

plt.plot(x, y1, label="y = 2x + 1", color="blue")  # Blue line
plt.plot(x, y2, label="y = 2x + 2", color="green", linestyle="--")  # Green dashed line
plt.plot(x, y3, label="y = 2x + 3", color="red", linestyle=":")  # Red dotted line

plt.title("Linear Equations")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()  # Add a legend
plt.grid()  # Add a grid
plt.show()
