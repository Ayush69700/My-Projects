import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)

    for _ in range(epochs):
        h_x = X.dot(theta)
        errors = y - h_x
        gradients = (1 / m) * X.T.dot(errors)

        theta = theta + alpha * gradients
    return theta


# using 'Hours Studied' and 'Previous Scores' as our two features
df = pd.read_csv("Data_sets/Student_Performance.csv")

x1 = df["Hours Studied"].values
x2 = df["Previous Scores"].values
y = df["Performance Index"].values

X = np.column_stack((x1, x2))
y = df["Performance Index"].values.reshape(-1, 1)  # type: ignore

mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_scaled = (X - mu) / sigma

X_final = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

theta_initial = np.zeros((X_final.shape[1], 1))
alpha = 0.01
epochs = 1500

theta_final = gradient_descent(X_final, y, theta_initial, alpha, epochs)

print("\nFinal Weights (Theta):")
print(theta_final)

# 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# create meshgrid
x1_scaled = (x1 - x1.mean()) / x1.std()
x2_scaled = (x2 - x2.mean()) / x2.std()

x1_range = np.linspace(x1_scaled.min(), x1_scaled.max(), 10)
x2_range = np.linspace(x2_scaled.min(), x2_scaled.max(), 10)
X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)


Z_plane = theta_final[0] + (theta_final[1] * X1_mesh) + (theta_final[2] * X2_mesh)
ax.plot_surface(X1_mesh, X2_mesh, Z_plane, color="red", alpha=0.5, edgecolor="none")

feature1 = "Hours Studied"
feature2 = "Previous Scores"
target = "Performance Index"


ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_zlabel(target)
ax.set_title("3D Linear Regression: Performance Index Prediction")

plt.show()
