# Metrics API Reference

This page provides detailed API documentation for evaluation metrics in Samay.

---

## Overview

Samay provides a comprehensive set of metrics for evaluating time-series forecasting models. These metrics help assess prediction accuracy, error patterns, and model performance.

::: samay.metric

---

## Available Metrics

### Mean Squared Error (MSE)

Measures the average squared difference between predictions and ground truth.

```python
from samay.metric import mse

loss = mse(y_true, y_pred)
```

**Formula:** $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

**Properties:**
- Range: [0, ∞)
- Lower is better
- Sensitive to outliers
- Same units as squared target variable

---

### Root Mean Squared Error (RMSE)

Square root of MSE, providing error in original units.

```python
from samay.metric import rmse

loss = rmse(y_true, y_pred)
```

**Formula:** $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$

**Properties:**
- Range: [0, ∞)
- Lower is better
- Same units as target variable
- Interpretable in original scale

---

### Mean Absolute Error (MAE)

Measures the average absolute difference between predictions and ground truth.

```python
from samay.metric import mae

loss = mae(y_true, y_pred)
```

**Formula:** $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

**Properties:**
- Range: [0, ∞)
- Lower is better
- Less sensitive to outliers than MSE
- Same units as target variable

---

### Mean Absolute Percentage Error (MAPE)

Measures the average percentage error.

```python
from samay.metric import mape

loss = mape(y_true, y_pred)
```

**Formula:** $MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$

**Properties:**
- Range: [0, ∞)
- Lower is better
- Scale-independent
- Undefined when $y_i = 0$

---

### Symmetric Mean Absolute Percentage Error (sMAPE)

A symmetric version of MAPE that handles zero values better.

```python
from samay.metric import smape

loss = smape(y_true, y_pred)
```

**Formula:** $sMAPE = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$

**Properties:**
- Range: [0, 200%]
- Lower is better
- More robust than MAPE
- Symmetric treatment of over/under predictions

---

### Mean Absolute Scaled Error (MASE)

Scale-independent metric that compares forecast to a naive baseline.

```python
from samay.metric import mase

loss = mase(y_true, y_pred, y_train)
```

**Formula:** $MASE = \frac{\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|}{\frac{1}{n-1}\sum_{i=2}^{n}|y_i - y_{i-1}|}$

**Properties:**
- Range: [0, ∞)
- Lower is better
- Scale-independent
- Values < 1 indicate better than naive forecast

---

### R² Score (Coefficient of Determination)

Measures the proportion of variance explained by the model.

```python
from samay.metric import r2_score

score = r2_score(y_true, y_pred)
```

**Formula:** $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

**Properties:**
- Range: (-∞, 1]
- Higher is better
- 1 = perfect predictions
- 0 = as good as mean baseline

---

## Usage Examples

### Basic Usage

```python
from samay.metric import mse, mae, mape
import numpy as np

# Get predictions from model
avg_loss, y_true, y_pred, histories = model.evaluate(test_dataset)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate metrics
mse_score = mse(y_true, y_pred)
mae_score = mae(y_true, y_pred)
mape_score = mape(y_true, y_pred)

print(f"MSE: {mse_score:.4f}")
print(f"MAE: {mae_score:.4f}")
print(f"MAPE: {mape_score:.4f}%")
```

### Multiple Metrics

```python
from samay.metric import mse, mae, rmse, mape, r2_score

metrics = {
    "MSE": mse(y_true, y_pred),
    "MAE": mae(y_true, y_pred),
    "RMSE": rmse(y_true, y_pred),
    "MAPE": mape(y_true, y_pred),
    "R²": r2_score(y_true, y_pred),
}

print("Evaluation Results:")
for metric_name, value in metrics.items():
    print(f"  {metric_name}: {value:.4f}")
```

### Per-Channel Metrics

```python
import numpy as np

# y_true and y_pred shape: (num_samples, num_channels, horizon)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate metrics per channel
num_channels = y_true.shape[1]

print("Per-Channel Results:")
for ch in range(num_channels):
    y_true_ch = y_true[:, ch, :]
    y_pred_ch = y_pred[:, ch, :]
    
    mse_ch = mse(y_true_ch, y_pred_ch)
    mae_ch = mae(y_true_ch, y_pred_ch)
    
    print(f"  Channel {ch}:")
    print(f"    MSE: {mse_ch:.4f}")
    print(f"    MAE: {mae_ch:.4f}")
```

### Per-Horizon Metrics

```python
# Calculate metrics at each forecast step
horizon_len = y_true.shape[-1]

mse_per_step = []
for t in range(horizon_len):
    y_true_t = y_true[:, :, t]
    y_pred_t = y_pred[:, :, t]
    mse_t = mse(y_true_t, y_pred_t)
    mse_per_step.append(mse_t)

# Plot error over forecast horizon
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, horizon_len + 1), mse_per_step)
plt.xlabel("Forecast Step")
plt.ylabel("MSE")
plt.title("Error Over Forecast Horizon")
plt.grid(alpha=0.3)
plt.show()
```

---

## Metric Selection Guide

### Choose MSE/RMSE when:
- You want to penalize large errors more heavily
- Working with continuous variables
- Outliers are important to detect

### Choose MAE when:
- You want a robust metric less sensitive to outliers
- All errors should be weighted equally
- Interpretability is important

### Choose MAPE when:
- You need scale-independent comparison
- Percentage errors are more meaningful
- Target values are never zero

### Choose MASE when:
- Comparing across different scales
- Need scale-independent metric
- Want to compare against naive baseline

### Choose R² when:
- Want to know proportion of variance explained
- Comparing model performance
- Need a normalized metric

---

## Custom Metrics

You can define custom metrics:

```python
import numpy as np

def custom_metric(y_true, y_pred):
    """
    Custom evaluation metric.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Metric value
    """
    # Your custom logic
    error = np.abs(y_true - y_pred)
    return np.mean(error)

# Use custom metric
score = custom_metric(y_true, y_pred)
print(f"Custom Metric: {score:.4f}")
```

---

## Aggregation Strategies

### Mean Aggregation

```python
# Average across all samples and channels
overall_mse = np.mean((y_true - y_pred) ** 2)
```

### Median Aggregation

```python
# Median (robust to outliers)
median_ae = np.median(np.abs(y_true - y_pred))
```

### Weighted Aggregation

```python
# Weight recent predictions more
horizon_len = y_true.shape[-1]
weights = np.linspace(0.5, 1.0, horizon_len)  # Increasing weights
weighted_error = np.average(
    np.abs(y_true - y_pred),
    axis=-1,
    weights=weights
)
```

---

## Metric Visualization

### Error Distribution

```python
import matplotlib.pyplot as plt
import numpy as np

errors = y_true - y_pred

plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
plt.legend()

# Box plot per channel
plt.subplot(1, 2, 2)
plt.boxplot([errors[:, ch, :].flatten() for ch in range(errors.shape[1])])
plt.xlabel("Channel")
plt.ylabel("Prediction Error")
plt.title("Error Distribution by Channel")

plt.tight_layout()
plt.show()
```

### Metric Comparison

```python
import matplotlib.pyplot as plt

metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE']
metric_values = [
    mse(y_true, y_pred),
    mae(y_true, y_pred),
    rmse(y_true, y_pred),
    mape(y_true, y_pred),
]

# Normalize for comparison
normalized_values = [v / max(metric_values) for v in metric_values]

plt.figure(figsize=(10, 5))
plt.bar(metric_names, normalized_values)
plt.ylabel("Normalized Value")
plt.title("Metric Comparison (Normalized)")
plt.grid(axis='y', alpha=0.3)
plt.show()
```

---

## See Also

- [Models API](models.md): Model classes and evaluation methods
- [Datasets API](datasets.md): Dataset classes
- [Examples](../examples.md): Complete evaluation examples

