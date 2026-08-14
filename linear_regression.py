import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data_sets/LR_data.csv")

# Standardize the data (Mean = 0, Standard Deviation = 1)
data["SAT_scaled"] = (data["SAT"] - data["SAT"].mean()) / data["SAT"].std()


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        # Use the SCALED data here
        x = points.iloc[i].SAT_scaled
        y = points.iloc[i].GPA
        # iloc stands for integer location - used to select rows and columns from a dataframe by their numerical positions

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


m = 0
b = 0
L = 0.01
epochs = 500

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

print(f"Final m: {m}, Final b: {b}")

# Plotting with the scaled data
plt.scatter(data.SAT_scaled, data.GPA, color="black")

# Create a line based on scaled min/max
x_vals = np.linspace(data.SAT_scaled.min(), data.SAT_scaled.max(), 100)
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color="blue")

plt.show()
