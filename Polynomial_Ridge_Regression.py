import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Parameters
degree = 10
alpha = 0.1
random_seed = 0

# Generate synthetic data
np.random.seed(random_seed)
X = 6 * np.random.randn(100, 1) - 3
y = (0.5 * X**2 + X + 2 + np.random.rand(100, 1)).ravel()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial feature transformation
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Ridge Regression model
model = Ridge(alpha=alpha)
model.fit(X_train_poly, y_train)

# Predictions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Evaluation
mse_training = mean_squared_error(y_train, y_train_pred)
mse_testing = mean_squared_error(y_test, y_test_pred)

print(f'Mean Squared Error for Training Data: {mse_training:.4f}')
print(f'Mean Squared Error for Testing Data: {mse_testing:.4f}')
print(f'Difference between Training and Testing Error: {abs(mse_training - mse_testing):.4f}')

# Plotting
x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_fit_poly = poly.transform(x_fit)
y_fit = model.predict(x_fit_poly)

plt.figure(figsize=(8,6))
plt.scatter(X_train, y_train, color='red', label='Training Data', alpha=0.6, marker='o')
plt.scatter(X_test, y_test, color='skyblue', label='Testing Data', alpha=0.8, marker='^')
plt.plot(x_fit, y_fit, color='blue', label='Polynomial Fit')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Ridge Regression Fit")
plt.legend()
plt.grid(True)
plt.show()
