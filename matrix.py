import numpy as np

# Matrix A
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# Calculate inverse
A_inv = np.linalg.inv(A)

# Check if A * A_inv and A_inv * A are identity matrices
product_1 = np.dot(A, A_inv)  # Multiply A by its inverse
product_2 = np.dot(A_inv, A)  # Multiply inverse by A

print("Matrix A:")
print(A)

print("\nInverse of A:")
print(A_inv)

print("\nA * A_inv:")
print(product_1)

print("\nA_inv * A:")
print(product_2)
