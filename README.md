# AI_PY_SGD
Stochastic Gradient Descent Algorithm Example
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (you can replace this with your own dataset)
np.random.seed(42)
X = np.random.rand(100, 1)  # Features (1D)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # Linear target variable

# Initialize weights
w = np.random.randn(1)

# Hyperparameters
learning_rate = 0.01
n_epochs = 1000

# Stochastic Gradient Descent
for epoch in range(n_epochs):
    for i in range(len(X)):
        # Randomly select a data point
        idx = np.random.randint(len(X))
        xi, yi = X[idx], y[idx]

        # Compute gradient for the selected data point
        gradient = -xi * (yi - xi * w)

        # Update weights
        w -= learning_rate * gradient

    # Compute loss (mean squared error)
    loss = np.mean((X * w - y) ** 2)

    # Display progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

# Final weight
print(f"Final weight: w = {w[0]:.4f}")

# Plot the data and the learned line
plt.scatter(X, y, label="Data points")
plt.plot(X, X * w, color="red", label="Learned line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Stochastic Gradient Descent for Linear Regression")
plt.legend()
plt.show()
