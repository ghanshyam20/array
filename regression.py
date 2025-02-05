# import numpy as np
# import matplotlib.pyplot as plt

# x_array = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
# y_array = [7, 8, 8, 9, 9, 9, 10, 11, 14, 14, 15]

# slope = sum(y_array) / sum(x_array)
# x_values = list(range(50, 151))
# y_values = [slope * x for x in x_values]

# plt.scatter(x_array, y_array, color='red'); plt.plot(x_values, y_values, label=f"y = {slope:.2f}x", color='blue'); plt.legend(); plt.grid(True); plt.show()



import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)  # Define x values
slope, intercept = 2, 3  # Define slope and intercept

# Define and plot the equations for different degrees
for power in [1, 2, 3, 4]:
    plt.plot(x, slope * x**power + intercept, label=f"y = {slope}x^{power} + {intercept}")

# Adding labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Equations with Different Degrees')
plt.legend()
plt.grid(True)
plt.show()
