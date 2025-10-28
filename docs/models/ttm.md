# TinyTimeMixer: Fast Pre-trained Models for Time Series

<div align="center" style="margin: 1em 0;">
    <strong>Lightweight and Efficient Time-Series Foundation Model</strong>
</div>

---

## Overview

**TinyTimeMixer (TTM)** is a compact and efficient time-series forecasting model designed for fast inference and low memory footprint. It uses a mixer-based architecture that balances performance with computational efficiency, making it ideal for resource-constrained environments.

### Paper

[TinyTimeMixer: Fast Pre-trained Models for Time Series](https://arxiv.org/abs/2401.03955)

### Key Features

- ✅ Lightweight architecture (compact model size)
- ✅ Fast inference speed
- ✅ Low memory footprint
- ✅ Competitive forecasting accuracy
- ✅ Efficient training and fine-tuning
- ✅ Multivariate time-series support

---

## Quick Start

```python
from samay.model import TinyTimeMixerModel
from samay.dataset import TinyTimeMixerDataset

# Model configuration
config = {
    "context_len": 512,
    "horizon_len": 96,
    "model_size": "tiny",
}

# Load model
model = TinyTimeMixerModel(config)

# Load dataset
train_dataset = TinyTimeMixerDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    context_len=config["context_len"],
    horizon_len=config["horizon_len"],
)

# Fine-tune
finetuned_model = model.finetune(train_dataset, epochs=10)

# Evaluate
test_dataset = TinyTimeMixerDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    context_len=config["context_len"],
    horizon_len=config["horizon_len"],
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
print(f"Average Loss: {avg_loss}")
```

---

## Model Variants

TinyTimeMixer comes in different sizes:

| Variant | Parameters | Speed | Accuracy |
|---------|------------|-------|----------|
| Tiny | ~1M | Fastest | Good |
| Small | ~5M | Fast | Better |
| Base | ~15M | Moderate | Best |

---

## Configuration Parameters

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_len` | int | `512` | Length of input context |
| `horizon_len` | int | `96` | Forecast horizon |
| `model_size` | str | `"tiny"` | Model size: `"tiny"`, `"small"`, `"base"` |
| `d_model` | int | `64` | Model dimension |
| `n_heads` | int | `4` | Number of attention heads |
| `n_layers` | int | `4` | Number of mixer layers |
| `dropout` | float | `0.1` | Dropout rate |

### Example Configurations

#### Tiny Model (Fast Inference)
```python
config = {
    "context_len": 512,
    "horizon_len": 96,
    "model_size": "tiny",
    "d_model": 64,
    "n_layers": 4,
}
```

#### Small Model (Balanced)
```python
config = {
    "context_len": 512,
    "horizon_len": 96,
    "model_size": "small",
    "d_model": 128,
    "n_layers": 6,
}
```

#### Base Model (High Accuracy)
```python
config = {
    "context_len": 512,
    "horizon_len": 192,
    "model_size": "base",
    "d_model": 256,
    "n_layers": 8,
}
```

---

## Dataset

### TinyTimeMixerDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `None` | Dataset name |
| `datetime_col` | str | `"ds"` | Name of datetime column |
| `path` | str | Required | Path to CSV file |
| `mode` | str | `None` | `"train"` or `"test"` |
| `context_len` | int | `512` | Length of input context |
| `horizon_len` | int | `64` | Forecast horizon |
| `batch_size` | int | `128` | Batch size |
| `boundaries` | list | `[0, 0, 0]` | Custom split boundaries |
| `stride` | int | `10` | Stride for sliding window |

### Data Format

CSV file with datetime and value columns:

```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,2.009,6.082
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,5.485,1.942,5.947
...
```

---

## Training

### Basic Training

```python
from samay.model import TinyTimeMixerModel
from samay.dataset import TinyTimeMixerDataset

# Configure model
config = {
    "context_len": 512,
    "horizon_len": 96,
    "model_size": "tiny",
}

model = TinyTimeMixerModel(config)

# Load training data
train_dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    context_len=512,
    horizon_len=96,
    batch_size=128,
)

# Fine-tune
finetuned_model = model.finetune(
    train_dataset,
    epochs=20,
    learning_rate=1e-3,
)
```

### Training with Validation

```python
# Training dataset
train_dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    context_len=512,
    horizon_len=96,
    boundaries=[0, 10000, 15000],  # Custom split
)

# Validation dataset
val_dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="val",
    context_len=512,
    horizon_len=96,
    boundaries=[0, 10000, 15000],
)

# Fine-tune with validation
finetuned_model = model.finetune(
    train_dataset,
    val_dataset=val_dataset,
    epochs=20,
    learning_rate=1e-3,
)
```

---

## Evaluation

### Basic Evaluation

```python
test_dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    context_len=512,
    horizon_len=96,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
print(f"Average Test Loss: {avg_loss}")
```

### With Custom Metrics

```python
from samay.metric import mse, mae, mape, rmse
import numpy as np

avg_loss, trues, preds, histories = model.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)

print(f"MSE:  {mse(trues, preds):.4f}")
print(f"MAE:  {mae(trues, preds):.4f}")
print(f"RMSE: {rmse(trues, preds):.4f}")
print(f"MAPE: {mape(trues, preds):.4f}%")
```

---

## Zero-Shot Forecasting

TinyTimeMixer supports zero-shot forecasting:

```python
# Load pre-trained model
config = {
    "context_len": 512,
    "horizon_len": 96,
    "model_size": "tiny",
}

model = TinyTimeMixerModel(config)

# Test on new data without training
test_dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/new_domain.csv",
    mode="test",
    context_len=512,
    horizon_len=96,
)

# Zero-shot evaluation
avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

---

## Multivariate Forecasting

TinyTimeMixer handles multivariate data efficiently:

```python
# Your CSV with multiple value columns
dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/multivariate.csv",  # Multiple columns
    mode="train",
    context_len=512,
    horizon_len=96,
)

# Model forecasts all channels simultaneously
avg_loss, trues, preds, histories = model.evaluate(dataset)

# Results shape: (num_windows, num_channels, horizon_len)
print(f"Predictions shape: {preds.shape}")
```

---

## Advanced Usage

### Custom Context Lengths

```python
# Short context for simple patterns
config = {
    "context_len": 256,
    "horizon_len": 64,
    "model_size": "tiny",
}

# Long context for complex patterns
config = {
    "context_len": 1024,
    "horizon_len": 192,
    "model_size": "small",
}
```

### Batch Size Tuning

```python
# Large batch for faster training (if memory allows)
dataset = TinyTimeMixerDataset(
    # ...
    batch_size=256,
)

# Small batch for memory efficiency
dataset = TinyTimeMixerDataset(
    # ...
    batch_size=32,
)
```

### Stride Configuration

```python
# Smaller stride for more training samples
dataset = TinyTimeMixerDataset(
    # ...
    stride=1,  # Overlapping windows
)

# Larger stride for faster iteration
dataset = TinyTimeMixerDataset(
    # ...
    stride=96,  # Non-overlapping windows
)
```

---

## Visualization

### Single Channel Forecast

```python
import matplotlib.pyplot as plt
import numpy as np

avg_loss, trues, preds, histories = model.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

# Plot first window, first channel
window_idx = 0
channel_idx = 0

history = histories[window_idx, channel_idx, :]
true = trues[window_idx, channel_idx, :]
pred = preds[window_idx, channel_idx, :]

plt.figure(figsize=(14, 5))
plt.plot(range(len(history)), history, label="History (512 steps)", linewidth=2)
plt.plot(
    range(len(history), len(history) + len(true)),
    true,
    label="Ground Truth (96 steps)",
    linestyle="--",
    linewidth=2
)
plt.plot(
    range(len(history), len(history) + len(pred)),
    pred,
    label="TinyTimeMixer Prediction",
    linewidth=2
)
plt.axvline(x=len(history), color='gray', linestyle=':', alpha=0.5)
plt.legend()
plt.title("TinyTimeMixer Time Series Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Multiple Channels

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i in range(min(4, trues.shape[1])):
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

plt.suptitle("TinyTimeMixer Multi-Channel Forecasting")
plt.tight_layout()
plt.show()
```

### Error Distribution

```python
import matplotlib.pyplot as plt
import numpy as np

avg_loss, trues, preds, histories = model.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)

# Calculate errors
errors = trues - preds

plt.figure(figsize=(12, 5))

# Error distribution
plt.subplot(1, 2, 1)
plt.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.grid(alpha=0.3)

# Error over time
plt.subplot(1, 2, 2)
mean_abs_errors = np.mean(np.abs(errors), axis=(0, 1))
plt.plot(mean_abs_errors)
plt.xlabel("Time Step")
plt.ylabel("Mean Absolute Error")
plt.title("Error Over Forecast Horizon")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Performance Comparison

### Speed Benchmark

```python
import time

models = [
    ("Tiny", {"model_size": "tiny", "d_model": 64}),
    ("Small", {"model_size": "small", "d_model": 128}),
    ("Base", {"model_size": "base", "d_model": 256}),
]

for name, model_config in models:
    config = {
        "context_len": 512,
        "horizon_len": 96,
        **model_config
    }
    
    model = TinyTimeMixerModel(config)
    
    # Measure inference time
    start_time = time.time()
    avg_loss, trues, preds, histories = model.evaluate(test_dataset)
    elapsed_time = time.time() - start_time
    
    print(f"{name} Model:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Time: {elapsed_time:.2f}s")
    print()
```

---

## Tips and Best Practices

### 1. Model Selection
- Use **Tiny** for edge devices and real-time applications
- Use **Small** for balanced performance and speed
- Use **Base** when accuracy is more important than speed

### 2. Context Length
- Longer context captures more patterns but is slower
- Match context to your data's seasonal patterns
- Start with 512 and adjust based on results

### 3. Batch Size
- TinyTimeMixer supports large batch sizes (128-256)
- Larger batches = faster training
- Reduce batch size if OOM errors occur

### 4. Training Duration
- TinyTimeMixer trains quickly (10-20 epochs often sufficient)
- Monitor validation loss to avoid overfitting
- Early stopping is recommended

---

## Common Issues

### CUDA Out of Memory

```python
# Use smaller model
config = {
    "model_size": "tiny",
    "d_model": 64,
    # ...
}

# Reduce batch size
dataset = TinyTimeMixerDataset(
    batch_size=32,  # Instead of 128
    # ...
)

# Reduce context/horizon
config = {
    "context_len": 256,  # Instead of 512
    "horizon_len": 48,   # Instead of 96
}
```

### Slow Training

```python
# Increase batch size (if memory allows)
dataset = TinyTimeMixerDataset(
    batch_size=256,  # Larger batch
    # ...
)

# Reduce model size
config = {
    "model_size": "tiny",  # Smaller model
    # ...
}
```

### Poor Accuracy

```python
# Use larger model
config = {
    "model_size": "base",  # Larger model
    "d_model": 256,
    "n_layers": 8,
}

# Increase context length
config = {
    "context_len": 1024,  # More context
    # ...
}

# Train longer
model.finetune(train_dataset, epochs=50)  # More epochs
```

---

## Efficient Deployment

### CPU Inference

TinyTimeMixer is efficient on CPU:

```python
import torch

# Force CPU usage
device = torch.device("cpu")
model = TinyTimeMixerModel(config).to(device)

# Inference is still fast!
avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

### Model Export

Export for production deployment:

```python
# Save model
model.save("tinytimemixer_model.pt")

# Load model
loaded_model = TinyTimeMixerModel.load("tinytimemixer_model.pt")
```

### Quantization (for even faster inference)

```python
import torch

# Quantize model for faster inference
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Use quantized model
avg_loss, trues, preds, histories = quantized_model.evaluate(test_dataset)
```

---

## API Reference

For detailed API documentation, see:

- [TinyTimeMixerModel API](../api/models.md#tinytimemixermodel)
- [TinyTimeMixerDataset API](../api/datasets.md#tinytimemixerdataset)

---

## Examples

See the [Examples](../examples.md) page for complete working examples.

---

## Comparison with Other Models

| Feature | TinyTimeMixer | LPTM | TimesFM | MOMENT |
|---------|---------------|------|---------|--------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Memory** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Accuracy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Edge Deployment** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

**Use TinyTimeMixer when:**
- You need fast inference
- Memory is limited
- Deploying on edge devices
- Real-time forecasting is required

