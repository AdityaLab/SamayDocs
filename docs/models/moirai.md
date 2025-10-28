# MOIRAI: Universal Time Series Forecasting Transformer

<div align="center" style="margin: 1em 0;">
    <strong>Unified Training of Universal Time Series Forecasting Transformers by Salesforce</strong>
</div>

---

## Overview

**MOIRAI** is a universal time-series forecasting transformer developed by Salesforce Research. It's trained on a massive collection of diverse time-series data and supports various frequencies, context lengths, and prediction horizons out-of-the-box.

### Paper

[Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592)

### Key Features

- ✅ Universal architecture for diverse datasets
- ✅ Multiple frequency support (hourly, daily, weekly, monthly, etc.)
- ✅ Flexible context and horizon lengths
- ✅ Mixture-of-Experts (MoE) variants
- ✅ Strong zero-shot performance across domains

---

## Model Variants

MOIRAI comes in several variants:

| Model | Size | Type | Repository |
|-------|------|------|------------|
| MOIRAI Small | ~12M | Standard | `Salesforce/moirai-1.0-R-small` |
| MOIRAI Base | ~91M | Standard | `Salesforce/moirai-1.0-R-base` |
| MOIRAI Large | ~311M | Standard | `Salesforce/moirai-1.0-R-large` |
| MOIRAI-MoE Small | ~39M | MoE | `Salesforce/moirai-moe-1.0-R-small` |
| MOIRAI-MoE Base | ~311M | MoE | `Salesforce/moirai-moe-1.0-R-base` |

---

## Quick Start

```python
from samay.model import MoiraiTSModel
from samay.dataset import MoiraiDataset

# Model configuration
repo = "Salesforce/moirai-1.0-R-small"
config = {
    "context_len": 128,
    "horizon_len": 64,
    "model_type": "moirai",
    "model_size": "small",
}

# Load model
moirai_model = MoiraiTSModel(repo=repo, config=config)

# Load dataset
train_dataset = MoiraiDataset(
    name="ett",
    mode="train",
    path="data/ETTh1.csv",
    datetime_col="date",
    freq="h",
    context_len=config['context_len'],
    horizon_len=config['horizon_len']
)

test_dataset = MoiraiDataset(
    name="ett",
    mode="test",
    path="data/ETTh1.csv",
    datetime_col="date",
    freq="h",
    context_len=config['context_len'],
    horizon_len=config['horizon_len']
)

# Zero-shot evaluation
eval_results, trues, preds, histories = moirai_model.evaluate(
    test_dataset,
    metrics=["MSE", "MAE", "MASE"]
)
print(f"Results: {eval_results}")
```

---

## Configuration Parameters

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_len` | int | `128` | Length of historical context |
| `horizon_len` | int | `64` | Forecast horizon |
| `model_type` | str | `"moirai"` | Model type: `"moirai"` or `"moirai-moe"` |
| `model_size` | str | `"small"` | Model size: `"small"`, `"base"`, `"large"` |
| `patch_size` | int | `16` | Patch size for patching |
| `num_layers` | int | `100` | Number of transformer layers |

### Example Configurations

#### Small Model (Fast Inference)
```python
repo = "Salesforce/moirai-1.0-R-small"
config = {
    "context_len": 128,
    "horizon_len": 64,
    "model_type": "moirai",
    "model_size": "small",
}
```

#### Base Model (Balanced)
```python
repo = "Salesforce/moirai-1.0-R-base"
config = {
    "context_len": 256,
    "horizon_len": 96,
    "model_type": "moirai",
    "model_size": "base",
}
```

#### Large Model (High Accuracy)
```python
repo = "Salesforce/moirai-1.0-R-large"
config = {
    "context_len": 512,
    "horizon_len": 128,
    "model_type": "moirai",
    "model_size": "large",
}
```

#### Mixture-of-Experts (MoE)
```python
repo = "Salesforce/moirai-moe-1.0-R-small"
config = {
    "context_len": 128,
    "horizon_len": 64,
    "model_type": "moirai-moe",
    "model_size": "small",
}
```

---

## Dataset

### MoiraiDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `None` | Dataset name |
| `datetime_col` | str | `"date"` | Name of datetime column |
| `path` | str | Required | Path to CSV file |
| `mode` | str | `"train"` | `"train"`, `"val"`, or `"test"` |
| `context_len` | int | `128` | Length of input context |
| `horizon_len` | int | `32` | Forecast horizon |
| `freq` | str | `None` | Time frequency: `"h"`, `"d"`, `"w"`, `"m"`, `"q"`, `"y"` |
| `batch_size` | int | `16` | Batch size |
| `normalize` | bool | `True` | Whether to normalize data |
| `boundaries` | tuple | `(0, 0, 0)` | Custom split boundaries |
| `start_date` | str | `None` | Start date for subset |
| `end_date` | str | `None` | End date for subset |

### Data Format

CSV file with datetime and value columns:

```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,2.009,6.082
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,5.485,1.942,5.947
...
```

---

## Frequency Support

MOIRAI natively supports multiple time frequencies:

### Hourly Data
```python
dataset = MoiraiDataset(
    path="data/hourly.csv",
    datetime_col="date",
    freq="h",  # Hourly
    # ...
)
```

### Daily Data
```python
dataset = MoiraiDataset(
    path="data/daily.csv",
    datetime_col="date",
    freq="d",  # Daily
    # ...
)
```

### Weekly Data
```python
dataset = MoiraiDataset(
    path="data/weekly.csv",
    datetime_col="date",
    freq="w",  # Weekly
    # ...
)
```

### Monthly Data
```python
dataset = MoiraiDataset(
    path="data/monthly.csv",
    datetime_col="date",
    freq="m",  # Monthly
    # ...
)
```

### Quarterly Data
```python
dataset = MoiraiDataset(
    path="data/quarterly.csv",
    datetime_col="date",
    freq="q",  # Quarterly
    # ...
)
```

### Yearly Data
```python
dataset = MoiraiDataset(
    path="data/yearly.csv",
    datetime_col="date",
    freq="y",  # Yearly
    # ...
)
```

---

## Zero-Shot Forecasting

MOIRAI is designed for zero-shot forecasting across different domains:

```python
from samay.model import MoiraiTSModel
from samay.dataset import MoiraiDataset

# Load model
repo = "Salesforce/moirai-1.0-R-small"
config = {
    "context_len": 128,
    "horizon_len": 64,
    "model_type": "moirai",
    "model_size": "small",
}

moirai_model = MoiraiTSModel(repo=repo, config=config)

# Test on different domain (no training!)
test_dataset = MoiraiDataset(
    name="new_domain",
    mode="test",
    path="data/new_domain.csv",
    datetime_col="timestamp",
    freq="h",
    context_len=128,
    horizon_len=64
)

# Zero-shot evaluation
eval_results, trues, preds, histories = moirai_model.evaluate(
    test_dataset,
    metrics=["MSE", "MAE"]
)
```

---

## Evaluation Metrics

MOIRAI supports multiple evaluation metrics:

```python
# Evaluate with multiple metrics
eval_results, trues, preds, histories = moirai_model.evaluate(
    test_dataset,
    metrics=["MSE", "MAE", "MASE", "MAPE", "RMSE"]
)

print("Evaluation Results:")
for metric, value in eval_results.items():
    print(f"  {metric}: {value:.4f}")
```

### Available Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **MASE**: Mean Absolute Scaled Error
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Squared Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error

---

## Training and Fine-Tuning

While MOIRAI is designed for zero-shot forecasting, you can fine-tune it:

```python
# Load model
repo = "Salesforce/moirai-1.0-R-small"
config = {
    "context_len": 128,
    "horizon_len": 64,
    "model_type": "moirai",
    "model_size": "small",
}

moirai_model = MoiraiTSModel(repo=repo, config=config)

# Load training data
train_dataset = MoiraiDataset(
    name="ett",
    mode="train",
    path="data/ETTh1.csv",
    datetime_col="date",
    freq="h",
    context_len=128,
    horizon_len=64
)

# Fine-tune
finetuned_model = moirai_model.finetune(
    train_dataset,
    epochs=10,
    learning_rate=1e-5
)

# Evaluate
test_dataset = MoiraiDataset(
    name="ett",
    mode="test",
    path="data/ETTh1.csv",
    datetime_col="date",
    freq="h",
    context_len=128,
    horizon_len=64
)

eval_results, trues, preds, histories = moirai_model.evaluate(test_dataset)
```

---

## Normalization

MOIRAI includes built-in normalization:

```python
# With normalization (default)
dataset = MoiraiDataset(
    path="data/ETTh1.csv",
    datetime_col="date",
    freq="h",
    normalize=True,  # Normalize data
    context_len=128,
    horizon_len=64
)

# Denormalize predictions
eval_results, trues, preds, histories = moirai_model.evaluate(dataset)
denormalized_preds = dataset._denormalize_data(preds)
denormalized_trues = dataset._denormalize_data(trues)
```

---

## Handling Multivariate Data

MOIRAI naturally handles multivariate time series:

```python
# Your CSV with multiple columns
dataset = MoiraiDataset(
    path="data/multivariate.csv",
    datetime_col="date",
    freq="h",
    context_len=128,
    horizon_len=64
)

# Model will forecast each channel independently
eval_results, trues, preds, histories = moirai_model.evaluate(dataset)

# Results shape: (num_windows, num_channels, horizon_len)
```

---

## Data Subsetting

Select specific date ranges:

```python
dataset = MoiraiDataset(
    path="data/ETTh1.csv",
    datetime_col="date",
    freq="h",
    start_date="2016-07-01",  # Start from this date
    end_date="2017-12-31",    # End at this date
    context_len=128,
    horizon_len=64
)
```

---

## Visualization

### Single Channel Forecast

```python
import matplotlib.pyplot as plt
import numpy as np

eval_results, trues, preds, histories = moirai_model.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

# Plot first channel, first window
channel_idx = 0
window_idx = 0

history = histories[window_idx, channel_idx, :]
true = trues[window_idx, channel_idx, :]
pred = preds[window_idx, channel_idx, :]

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
    range(len(history), len(history) + len(pred)),
    pred,
    label="MOIRAI Prediction",
    linewidth=2
)
plt.axvline(x=len(history), color='gray', linestyle=':', alpha=0.5)
plt.legend()
plt.title("MOIRAI Time Series Forecasting")
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

plt.suptitle("MOIRAI Multi-Channel Forecasting")
plt.tight_layout()
plt.show()
```

---

## Advanced Usage

### Custom Boundaries

```python
# Specify exact train/val/test split points
dataset = MoiraiDataset(
    path="data/ETTh1.csv",
    datetime_col="date",
    freq="h",
    boundaries=(0, 10000, 15000),  # Train: 0-10k, Val: 10k-15k, Test: 15k-end
    context_len=128,
    horizon_len=64
)
```

### Different Context Lengths

```python
# Short context
config = {
    "context_len": 64,
    "horizon_len": 32,
    # ...
}

# Long context
config = {
    "context_len": 512,
    "horizon_len": 256,
    # ...
}
```

### Mixture-of-Experts Models

```python
# Use MoE variant for better performance
repo = "Salesforce/moirai-moe-1.0-R-small"
config = {
    "context_len": 128,
    "horizon_len": 64,
    "model_type": "moirai-moe",  # MoE type
    "model_size": "small",
}

moirai_moe_model = MoiraiTSModel(repo=repo, config=config)
```

---

## Tips and Best Practices

### 1. Model Selection
- Use **Small** for fast inference and experimentation
- Use **Base** for balanced performance
- Use **Large** for highest accuracy
- Use **MoE** variants for best performance at similar compute

### 2. Context Length
- Match context length to your data's pattern length
- Longer context captures more patterns but is slower
- Start with 128-256 and adjust based on results

### 3. Frequency Specification
- Always specify the correct frequency (`freq` parameter)
- Correct frequency helps MOIRAI understand temporal patterns
- MOIRAI can auto-detect frequency but explicit is better

### 4. Normalization
- Keep normalization enabled (default) for best results
- Remember to denormalize predictions for interpretation

---

## Common Issues

### CUDA Out of Memory

```python
# Use smaller model
repo = "Salesforce/moirai-1.0-R-small"

# Reduce batch size
dataset = MoiraiDataset(
    batch_size=8,  # Instead of 16
    # ...
)

# Reduce context/horizon length
config = {
    "context_len": 64,  # Instead of 128
    "horizon_len": 32,  # Instead of 64
}
```

### Incorrect Frequency

Make sure frequency matches your data:
```python
# Check your data's datetime column
import pandas as pd
df = pd.read_csv("data.csv")
df['date'] = pd.to_datetime(df['date'])
inferred_freq = pd.infer_freq(df['date'])
print(f"Inferred frequency: {inferred_freq}")

# Use correct frequency
dataset = MoiraiDataset(
    freq=inferred_freq,  # Use inferred frequency
    # ...
)
```

---

## API Reference

For detailed API documentation, see:

- [MoiraiTSModel API](../api/models.md#moiraitsmodel)
- [MoiraiDataset API](../api/datasets.md#moiraidataset)

---

## Examples

See the [Examples](../examples.md) page for complete working examples.

