import numpy as np
import river
from river import linear_model, preprocessing, metrics

# Generate synthetic time series data (sine wave)
np.random.seed(42)
n_points = 1000
X = np.arange(n_points)
y = np.sin(0.02 * X) + 0.1 * np.random.randn(n_points)  # Sine wave with noise

# Model and preprocessing pipeline
model = (
    preprocessing.StandardScaler() |  # Feature scaling
    linear_model.LinearRegression(optimizer=river.optim.SGD(0.1))  # SGD optimizer
)

# Metrics for evaluation
mae_metric = metrics.MAE()

# Function to create lag features
def create_lag_features(data, lag):
    features = {}
    for i in range(1, lag + 1):
        features[f'lag_{i}'] = data[-i]
    return features

# Window size for lag features
window_size = 5

# Train the model online
for i in range(window_size, n_points):
    # Prepare the feature for current timestep by creating lagged features
    x = create_lag_features(y[i-window_size:i], window_size)
    
    # Make prediction
    y_pred = model.predict_one(x)  # Make prediction
    
    if y_pred is not None:  # Calculate error and update metrics
        mae_metric.update(y_true=y[i], y_pred=y_pred)
    
    # Update model with true value
    model.learn_one(x, y[i])

    # Output predictions every 100 steps
    if i % 100 == 0:
        print(f"Step: {i}, MAE: {mae_metric.get():.4f}")

print("Final MAE:", mae_metric.get())
