# LPTM: Large Pre-trained Time Series Model

<div align="center" style="margin: 1em 0;">
    <strong>Large Pre-trained Time Series Models for Cross-Domain Time Series Analysis Tasks</strong>
</div>

---

## Overview

**LPTM (Large Pre-trained Time Series Model)** is a foundational model designed for general-purpose time-series forecasting. It uses a transformer-based architecture with a unique segmentation module that adaptively identifies patterns in time-series data.

### Paper

[Large Pre-trained time series models for cross-domain Time series analysis tasks](https://arxiv.org/abs/2311.11413)

### Key Features

- ✅ Pre-trained on large-scale time-series data
- ✅ Adaptive segmentation for pattern discovery
- ✅ Supports forecasting, classification, and anomaly detection
- ✅ Efficient fine-tuning with frozen encoders
- ✅ Handles multivariate time series

---

## Quick Start

```python
from samay.model import LPTMModel
from samay.dataset import LPTMDataset

# Configure the model
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

# Load model
model = LPTMModel(config)

# Load dataset
train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
)

# Fine-tune
finetuned_model = model.finetune(train_dataset)

# Evaluate
test_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon=192,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

---

## Configuration Parameters

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_name` | str | `"forecasting"` | Task type: `"forecasting"`, `"classification"`, `"detection"` |
| `forecast_horizon` | int | `192` | Number of time steps to predict |
| `freeze_encoder` | bool | `True` | Whether to freeze the patch embedding layer |
| `freeze_embedder` | bool | `True` | Whether to freeze the transformer encoder |
| `freeze_head` | bool | `False` | Whether to freeze the forecasting head |
| `freeze_segment` | bool | `True` | Whether to freeze the segmentation module |
| `head_dropout` | float | `0.0` | Dropout rate for the forecasting head |
| `weight_decay` | float | `0.0` | Weight decay for regularization |
| `max_patch` | int | `16` | Maximum patch size for segmentation |

### Example Configurations

#### Zero-Shot Forecasting
```python
config = {
    "task_name": "forecasting",
    "forecast_horizon": 96,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": True,  # Keep all layers frozen
}
```

#### Fine-Tuning for Domain Adaptation
```python
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,  # Only train the head
    "head_dropout": 0.1,
    "weight_decay": 0.001,
}
```

#### Full Fine-Tuning
```python
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": False,
    "freeze_embedder": False,
    "freeze_head": False,
    "head_dropout": 0.1,
}
```

---

## Dataset

### LPTMDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `None` | Dataset name (for metadata) |
| `datetime_col` | str | `None` | Name of the datetime column |
| `path` | str | Required | Path to CSV file |
| `mode` | str | `"train"` | `"train"` or `"test"` |
| `horizon` | int | `0` | Forecast horizon length |
| `batchsize` | int | `16` | Batch size for training |
| `boundaries` | list | `[0, 0, 0]` | Custom train/val/test split indices |
| `stride` | int | `10` | Stride for sliding window |
| `seq_len` | int | `512` | Input sequence length |
| `task_name` | str | `"forecasting"` | Task type |

### Data Format

Your CSV file should have:
- A datetime column (e.g., `date`)
- One or more value columns

Example:
```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,2.009,6.082
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,5.485,1.942,5.947
...
```

---

## Training

### Fine-Tuning

```python
# Create training dataset
train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
    batchsize=16,
)

# Fine-tune the model
finetuned_model = model.finetune(
    train_dataset,
    epochs=5,
    learning_rate=1e-4,
)
```

### Custom Training Loop

For more control, you can implement a custom training loop:

```python
import torch
from torch.optim import Adam

# Get data loader
train_loader = train_dataset.get_data_loader()

# Setup optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        loss = model.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
```

---

## Evaluation

### Basic Evaluation

```python
test_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon=192,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
print(f"Average Test Loss: {avg_loss}")
```

### Custom Metrics

```python
from samay.metric import mse, mae, mape

# Get predictions
avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Calculate custom metrics
import numpy as np
trues = np.array(trues)
preds = np.array(preds)

mse_score = mse(trues, preds)
mae_score = mae(trues, preds)
mape_score = mape(trues, preds)

print(f"MSE: {mse_score:.4f}")
print(f"MAE: {mae_score:.4f}")
print(f"MAPE: {mape_score:.4f}")
```

---

## Tasks

### 1. Forecasting

Predict future values:

```python
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}
model = LPTMModel(config)
```

### 2. Anomaly Detection

Detect anomalies in time series:

```python
config = {
    "task_name": "detection",
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}
model = LPTMModel(config)

dataset = LPTMDataset(
    name="ecg",
    datetime_col="date",
    path="./data/ECG5000.csv",
    mode="train",
    task_name="detection",
)
```

### 3. Classification

Classify time series:

```python
config = {
    "task_name": "classification",
    "num_classes": 5,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}
model = LPTMModel(config)
```

---

## Advanced Usage

### Handling Multivariate Time Series

LPTM naturally handles multivariate data:

```python
# Your CSV with multiple columns
# date,sensor1,sensor2,sensor3,...
train_dataset = LPTMDataset(
    datetime_col="date",
    path="./data/multivariate.csv",
    mode="train",
    horizon=192,
)
```

### Custom Data Splits

```python
# Specify exact boundaries
train_dataset = LPTMDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
    boundaries=[0, 10000, 15000],  # Train: 0-10000, Val: 10000-15000, Test: 15000-end
)
```

### Denormalizing Predictions

```python
# Get normalized predictions
avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Denormalize using the dataset's scaler
denormalized_preds = test_dataset._denormalize_data(preds)
denormalized_trues = test_dataset._denormalize_data(trues)
```

---

## Visualization

### Plotting Forecasts

```python
import matplotlib.pyplot as plt
import numpy as np

# Get predictions
avg_loss, trues, preds, histories = model.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

# Plot a specific channel and time window
channel_idx = 0
time_index = 0

history = histories[time_index, channel_idx, :]
true = trues[time_index, channel_idx, :]
pred = preds[time_index, channel_idx, :]

plt.figure(figsize=(14, 5))
plt.plot(range(len(history)), history, label="History (512 steps)", linewidth=2)
plt.plot(
    range(len(history), len(history) + len(true)),
    true,
    label="Ground Truth (192 steps)",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    range(len(history), len(history) + len(pred)),
    pred,
    label="Prediction (192 steps)",
    linewidth=2,
)
plt.axvline(x=len(history), color='gray', linestyle=':', alpha=0.5)
plt.legend()
plt.title("LPTM Time Series Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Tips and Best Practices

### 1. Choose Appropriate Forecast Horizons
- Short-term: 24-96 steps
- Medium-term: 192-336 steps
- Long-term: 720+ steps

### 2. Fine-Tuning Strategy
- Start with frozen encoder and embedder
- Only train the forecasting head
- If results are unsatisfactory, gradually unfreeze layers

### 3. Batch Size
- Larger batch sizes (32-64) for stable training
- Smaller batch sizes (8-16) if GPU memory is limited

### 4. Data Preprocessing
- LPTM handles normalization internally
- Ensure datetime column is properly formatted
- Handle missing values before loading

---

## Common Issues

### Out of Memory

Reduce batch size or forecast horizon:
```python
config = {
    "forecast_horizon": 96,  # Instead of 192
}

dataset = LPTMDataset(
    batchsize=8,  # Instead of 16
    # ...
)
```

### Poor Performance

Try full fine-tuning:
```python
config = {
    "freeze_encoder": False,
    "freeze_embedder": False,
    "freeze_head": False,
    "head_dropout": 0.1,
    "weight_decay": 0.001,
}
```

---

## API Reference

For detailed API documentation, see:

- [LPTMModel API](../api/models.md#lptmmodel)
- [LPTMDataset API](../api/datasets.md#lptmdataset)

---

## Examples

See the [Examples](../examples.md) page for complete working examples.

