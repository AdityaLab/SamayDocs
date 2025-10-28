# TimesFM: Time Series Foundation Model

<div align="center" style="margin: 1em 0;">
    <strong>A Decoder-Only Foundation Model for Time-Series Forecasting by Google Research</strong>
</div>

---

## Overview

**TimesFM (Time Series Foundation Model)** is a decoder-only
 architecture developed by Google Research for time-series forecasting. It's designed for efficient zero-shot forecasting across diverse domains.

### Paper

[A decoder-only foundation model for time-series forecasting](https://arxiv.org/html/2310.10688v2)

### Key Features

- ✅ Decoder-only transformer architecture
- ✅ Efficient zero-shot forecasting
- ✅ Patch-based input processing
- ✅ Multiple quantile predictions
- ✅ Fast inference on GPU

---

## Quick Start

```python
from samay.model import TimesfmModel
from samay.dataset import TimesfmDataset

# Model configuration
repo = "google/timesfm-1.0-200m-pytorch"
config = {
    "context_len": 512,
    "horizon_len": 192,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 20,
    "model_dims": 1280,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

# Load model
tfm = TimesfmModel(config=config, repo=repo)

# Load dataset
train_dataset = TimesfmDataset(
    name="ett",
    datetime_col='date',
    path='data/ETTh1.csv',
    mode='train',
    context_len=config["context_len"],
    horizon_len=config["horizon_len"]
)

# Evaluate (zero-shot)
avg_loss, trues, preds, histories = tfm.evaluate(train_dataset)
print(f"Average Loss: {avg_loss}")
```

---

## Model Variants

TimesFM comes in multiple sizes:

| Model | Parameters | Repository |
|-------|------------|------------|
| TimesFM 1.0 (200M) | 200M | `google/timesfm-1.0-200m-pytorch` |
| TimesFM 2.0 (500M) | 500M | `google/timesfm-2.0-500m-pytorch` |

### Choosing a Model

```python
# Smaller, faster model
repo = "google/timesfm-1.0-200m-pytorch"

# Larger, more accurate model
repo = "google/timesfm-2.0-500m-pytorch"
```

---

## Configuration Parameters

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_len` | int | `512` | Length of historical context |
| `horizon_len` | int | `192` | Forecast horizon |
| `backend` | str | `"gpu"` | Backend: `"gpu"` or `"cpu"` |
| `per_core_batch_size` | int | `32` | Batch size per core |
| `input_patch_len` | int | `32` | Length of input patches |
| `output_patch_len` | int | `128` | Length of output patches |
| `num_layers` | int | `20` | Number of transformer layers |
| `model_dims` | int | `1280` | Model dimension |
| `quantiles` | list | `[0.1, ..., 0.9]` | Quantiles for prediction intervals |

### Example Configurations

#### Standard Configuration (200M Model)
```python
config = {
    "context_len": 512,
    "horizon_len": 192,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 20,
    "model_dims": 1280,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
```

#### Larger Model (500M)
```python
config = {
    "context_len": 512,
    "horizon_len": 192,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 50,  # More layers
    "model_dims": 1280,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
```

#### CPU Inference
```python
config = {
    "context_len": 512,
    "horizon_len": 96,
    "backend": "cpu",  # Use CPU
    "per_core_batch_size": 8,  # Smaller batch
    # ... other configs
}
```

---

## Dataset

### TimesfmDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `None` | Dataset name |
| `datetime_col` | str | `"ds"` | Name of datetime column |
| `path` | str | Required | Path to CSV file |
| `mode` | str | `"train"` | `"train"` or `"test"` |
| `context_len` | int | `128` | Length of input context |
| `horizon_len` | int | `32` | Forecast horizon |
| `freq` | str | `"h"` | Frequency: `"h"`, `"d"`, `"w"`, etc. |
| `normalize` | bool | `False` | Whether to normalize data |
| `stride` | int | `10` | Stride for sliding window |
| `batchsize` | int | `4` | Batch size |

### Data Format

CSV file with datetime and value columns:

```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,2.009,6.082
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,5.485,1.942,5.947
...
```

---

## Zero-Shot Forecasting

TimesFM excels at zero-shot forecasting:

```python
from samay.model import TimesfmModel
from samay.dataset import TimesfmDataset

# Load model
repo = "google/timesfm-1.0-200m-pytorch"
config = {
    "context_len": 512,
    "horizon_len": 192,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 20,
    "model_dims": 1280,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
tfm = TimesfmModel(config=config, repo=repo)

# Load test data
test_dataset = TimesfmDataset(
    name="ett",
    datetime_col='date',
    path='data/ETTh1.csv',
    mode='test',
    context_len=config["context_len"],
    horizon_len=config["horizon_len"]
)

# Zero-shot evaluation (no training!)
avg_loss, trues, preds, histories = tfm.evaluate(test_dataset)
print(f"Zero-shot Loss: {avg_loss}")
```

---

## Evaluation

### Basic Evaluation

```python
test_dataset = TimesfmDataset(
    name="ett",
    datetime_col='date',
    path='data/ETTh1.csv',
    mode='test',
    context_len=512,
    horizon_len=192
)

avg_loss, trues, preds, histories = tfm.evaluate(test_dataset)
```

### With Custom Metrics

```python
from samay.metric import mse, mae, mape
import numpy as np

avg_loss, trues, preds, histories = tfm.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)

print(f"MSE: {mse(trues, preds):.4f}")
print(f"MAE: {mae(trues, preds):.4f}")
print(f"MAPE: {mape(trues, preds):.4f}")
```

---

## Quantile Predictions

TimesFM provides prediction intervals via quantiles:

```python
config = {
    # ... other configs
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],  # 10%, 25%, median, 75%, 90%
}

tfm = TimesfmModel(config=config, repo=repo)

# The model will output predictions for each quantile
avg_loss, trues, preds, histories = tfm.evaluate(test_dataset)

# preds shape: (num_samples, num_channels, horizon_len, num_quantiles)
```

### Visualizing Prediction Intervals

```python
import matplotlib.pyplot as plt
import numpy as np

# Assuming preds has shape (num_samples, num_channels, horizon_len, num_quantiles)
median_idx = 2  # Index of 0.5 quantile
lower_idx = 0   # Index of 0.1 quantile
upper_idx = 4   # Index of 0.9 quantile

sample_idx = 0
channel_idx = 0

history = histories[sample_idx, channel_idx, :]
true = trues[sample_idx, channel_idx, :]

# Assuming the model returns median predictions
pred_median = preds[sample_idx, channel_idx, :]

plt.figure(figsize=(14, 5))
plt.plot(range(len(history)), history, label="History", linewidth=2)
plt.plot(
    range(len(history), len(history) + len(true)),
    true,
    label="Ground Truth",
    linestyle="--",
    linewidth=2
)
plt.plot(
    range(len(history), len(history) + len(pred_median)),
    pred_median,
    label="Prediction (Median)",
    linewidth=2
)
plt.legend()
plt.title("TimesFM Forecasting with Prediction Intervals")
plt.grid(alpha=0.3)
plt.show()
```

---

## Handling Different Frequencies

TimesFM supports various time frequencies:

```python
# Hourly data
dataset = TimesfmDataset(
    datetime_col='date',
    path='data/hourly.csv',
    freq='h',
    # ...
)

# Daily data
dataset = TimesfmDataset(
    datetime_col='date',
    path='data/daily.csv',
    freq='d',
    # ...
)

# Weekly data
dataset = TimesfmDataset(
    datetime_col='date',
    path='data/weekly.csv',
    freq='w',
    # ...
)

# Monthly data
dataset = TimesfmDataset(
    datetime_col='date',
    path='data/monthly.csv',
    freq='m',
    # ...
)
```

---

## Normalization

TimesFM can optionally normalize data:

```python
# With normalization
train_dataset = TimesfmDataset(
    name="ett",
    datetime_col='date',
    path='data/ETTh1.csv',
    mode='train',
    context_len=512,
    horizon_len=192,
    normalize=True,  # Enable normalization
)

# Denormalize predictions
avg_loss, trues, preds, histories = tfm.evaluate(train_dataset)
denormalized_preds = train_dataset._denormalize_data(preds)
```

---

## Advanced Usage

### Custom Context Lengths

```python
# Short context for fast inference
config = {
    "context_len": 128,
    "horizon_len": 64,
    # ...
}

# Long context for better accuracy
config = {
    "context_len": 1024,
    "horizon_len": 256,
    # ...
}
```

### Batch Processing

```python
# Larger batches for throughput
config = {
    "per_core_batch_size": 64,
    # ...
}

# Smaller batches for memory efficiency
config = {
    "per_core_batch_size": 8,
    # ...
}
```

---

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

avg_loss, trues, preds, histories = tfm.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

# Plot multiple channels
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i in range(4):
    ax = axes[i]
    
    history = histories[0, i, :]
    true = trues[0, i, :]
    pred = preds[0, i, :]
    
    ax.plot(range(len(history)), history, label="History", alpha=0.7)
    ax.plot(
        range(len(history), len(history) + len(true)),
        true,
        label="Ground Truth",
        linestyle="--"
    )
    ax.plot(
        range(len(history), len(history) + len(pred)),
        pred,
        label="Prediction"
    )
    ax.set_title(f"Channel {i}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Tips and Best Practices

### 1. Model Selection
- Use 200M model for faster inference
- Use 500M model for higher accuracy

### 2. Context Length
- Longer context (512-1024) for complex patterns
- Shorter context (128-256) for simpler patterns and speed

### 3. Zero-Shot vs Fine-Tuning
- TimesFM is designed for zero-shot forecasting
- Fine-tuning is not typically required

### 4. GPU Memory
- Reduce `per_core_batch_size` if OOM
- Use CPU backend for very limited memory

---

## Common Issues

### CUDA Out of Memory

```python
# Reduce batch size
config = {
    "per_core_batch_size": 8,  # Lower value
    # ...
}

# Or use CPU
config = {
    "backend": "cpu",
    # ...
}
```

### Slow Inference

```python
# Use smaller model
repo = "google/timesfm-1.0-200m-pytorch"

# Reduce context length
config = {
    "context_len": 256,  # Instead of 512
    # ...
}
```

---

## API Reference

For detailed API documentation, see:

- [TimesfmModel API](../api/models.md#timesfmmodel)
- [TimesfmDataset API](../api/datasets.md#timesfmdataset)

---

## Examples

See the [Examples](../examples.md) page for complete working examples.

