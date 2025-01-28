import numpy as np
import matplotlib.pyplot as plt

# Data for x and y
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.scatter(x, y, marker='+', color='blue')  # Use '+' to mark the points
plt.title("Scatter Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()  # Add a grid
plt.show()
