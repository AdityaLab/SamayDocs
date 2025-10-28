# Chronos: Learning the Language of Time Series

<div align="center" style="margin: 1em 0;">
    <strong>Language Model-Based Time-Series Forecasting</strong>
</div>

---

## Overview

**Chronos** is a novel approach to time-series forecasting that treats time series as a language. It uses transformer architectures similar to large language models (LLMs) to tokenize and predict time-series data. This innovative approach enables zero-shot forecasting across diverse domains.

### Paper

[Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)

### Key Features

- ✅ Language model architecture for time series
- ✅ Tokenization-based approach
- ✅ Strong zero-shot capabilities
- ✅ Multiple model sizes
- ✅ Probabilistic forecasting

---

## Model Variants

Chronos comes in several sizes:

| Model | Parameters | Use Case |
|-------|------------|----------|
| Chronos-T5-tiny | ~8M | Fast inference, resource-constrained |
| Chronos-T5-mini | ~20M | Balanced performance |
| Chronos-T5-small | ~46M | Good accuracy |
| Chronos-T5-base | ~200M | High accuracy |
| Chronos-T5-large | ~800M | Best performance |

---

## Quick Start

```python
from samay.model import ChronosModel
from samay.dataset import ChronosDataset

# Model configuration
config = {
    "model_size": "small",  # tiny, mini, small, base, large
    "context_length": 512,
    "prediction_length": 64,
    "num_samples": 20,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
}

# Load model
model = ChronosModel(config)

# Load dataset
train_dataset = ChronosDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    config=config,
)

# Evaluate (zero-shot)
test_dataset = ChronosDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    config=config,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
print(f"Average Loss: {avg_loss}")
```

---

## Configuration Parameters

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | `"small"` | Model size: `"tiny"`, `"mini"`, `"small"`, `"base"`, `"large"` |
| `context_length` | int | `512` | Length of input context |
| `prediction_length` | int | `64` | Forecast horizon |
| `num_samples` | int | `20` | Number of samples for probabilistic forecasting |
| `temperature` | float | `1.0` | Sampling temperature |
| `top_k` | int | `50` | Top-k sampling parameter |
| `top_p` | float | `1.0` | Nucleus sampling parameter |
| `tokenizer_class` | str | `"MeanScaleUniformBins"` | Tokenizer type |
| `tokenizer_kwargs` | dict | `{"low_limit": -15.0, "high_limit": 15.0}` | Tokenizer parameters |

### Example Configurations

#### Fast Inference (Tiny Model)
```python
config = {
    "model_size": "tiny",
    "context_length": 256,
    "prediction_length": 32,
    "num_samples": 10,
}
```

#### Balanced Performance (Small Model)
```python
config = {
    "model_size": "small",
    "context_length": 512,
    "prediction_length": 64,
    "num_samples": 20,
    "temperature": 1.0,
}
```

#### High Accuracy (Base Model)
```python
config = {
    "model_size": "base",
    "context_length": 512,
    "prediction_length": 96,
    "num_samples": 50,
    "temperature": 0.8,
}
```

---

## Dataset

### ChronosDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `None` | Dataset name |
| `datetime_col` | str | `"ds"` | Name of datetime column |
| `path` | str | Required | Path to CSV file |
| `mode` | str | `None` | `"train"` or `"test"` |
| `batch_size` | int | `16` | Batch size |
| `boundaries` | list | `[0, 0, 0]` | Custom split boundaries |
| `stride` | int | `10` | Stride for sliding window |
| `config` | dict | `None` | Model configuration (used for context/prediction length) |

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

Chronos excels at zero-shot forecasting without any fine-tuning:

```python
from samay.model import ChronosModel
from samay.dataset import ChronosDataset

# Load model
config = {
    "model_size": "small",
    "context_length": 512,
    "prediction_length": 96,
    "num_samples": 20,
}

model = ChronosModel(config)

# Load test data directly
test_dataset = ChronosDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    config=config,
)

# Zero-shot evaluation (no training!)
avg_loss, trues, preds, histories = model.evaluate(test_dataset)
print(f"Zero-shot Loss: {avg_loss}")
```

---

## Probabilistic Forecasting

Chronos provides probabilistic forecasts through multiple samples:

```python
config = {
    "model_size": "small",
    "context_length": 512,
    "prediction_length": 96,
    "num_samples": 100,  # Generate 100 samples
    "temperature": 1.0,
}

model = ChronosModel(config)

# The model will generate multiple forecast samples
avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# preds shape: (num_windows, num_channels, prediction_length, num_samples)
```

### Analyzing Prediction Uncertainty

```python
import numpy as np
import matplotlib.pyplot as plt

# Get all samples
avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Calculate statistics
mean_pred = np.mean(preds, axis=-1)  # Mean across samples
std_pred = np.std(preds, axis=-1)    # Std across samples
lower_bound = np.percentile(preds, 10, axis=-1)
upper_bound = np.percentile(preds, 90, axis=-1)

# Plot with uncertainty bands
sample_idx = 0
channel_idx = 0

history = histories[sample_idx, channel_idx, :]
true = trues[sample_idx, channel_idx, :]
pred_mean = mean_pred[sample_idx, channel_idx, :]
pred_lower = lower_bound[sample_idx, channel_idx, :]
pred_upper = upper_bound[sample_idx, channel_idx, :]

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
    range(len(history), len(history) + len(pred_mean)),
    pred_mean,
    label="Mean Prediction",
    linewidth=2
)
plt.fill_between(
    range(len(history), len(history) + len(pred_mean)),
    pred_lower,
    pred_upper,
    alpha=0.3,
    label="80% Prediction Interval"
)
plt.legend()
plt.title("Chronos Probabilistic Forecasting")
plt.grid(alpha=0.3)
plt.show()
```

---

## Fine-Tuning

While Chronos is designed for zero-shot forecasting, you can fine-tune it on your data:

```python
# Load model
config = {
    "model_size": "small",
    "context_length": 512,
    "prediction_length": 96,
}

model = ChronosModel(config)

# Load training data
train_dataset = ChronosDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    config=config,
    batch_size=16,
)

# Fine-tune
finetuned_model = model.finetune(
    train_dataset,
    epochs=5,
    learning_rate=1e-5,  # Small learning rate
)

# Evaluate
test_dataset = ChronosDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    config=config,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

---

## Sampling Strategies

### Temperature Sampling

Control the randomness of predictions:

```python
# Lower temperature = more conservative predictions
config = {
    "temperature": 0.5,  # More deterministic
    # ...
}

# Higher temperature = more diverse predictions
config = {
    "temperature": 1.5,  # More exploratory
    # ...
}
```

### Top-K Sampling

Limit sampling to top-k most likely tokens:

```python
config = {
    "top_k": 10,  # Only sample from top 10 tokens
    # ...
}
```

### Nucleus (Top-P) Sampling

Sample from the smallest set of tokens with cumulative probability > p:

```python
config = {
    "top_p": 0.9,  # Sample from top 90% probability mass
    # ...
}
```

---

## Evaluation

### Basic Evaluation

```python
test_dataset = ChronosDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    config=config,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

### With Custom Metrics

```python
from samay.metric import mse, mae, mape
import numpy as np

avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Use mean of samples for metrics
trues = np.array(trues)
preds = np.mean(np.array(preds), axis=-1)  # Average across samples

print(f"MSE: {mse(trues, preds):.4f}")
print(f"MAE: {mae(trues, preds):.4f}")
print(f"MAPE: {mape(trues, preds):.4f}")
```

---

## Advanced Usage

### Custom Context Lengths

```python
# Short context for simpler patterns
config = {
    "context_length": 128,
    "prediction_length": 32,
    # ...
}

# Long context for complex patterns
config = {
    "context_length": 1024,
    "prediction_length": 128,
    # ...
}
```

### Multivariate Forecasting

Chronos handles multivariate data by forecasting each channel independently:

```python
# Your CSV with multiple columns
dataset = ChronosDataset(
    datetime_col="date",
    path="./data/multivariate.csv",
    mode="test",
    config=config,
)

# Model will forecast all channels
avg_loss, trues, preds, histories = model.evaluate(dataset)
```

---

## Tokenization

Chronos uses a tokenization approach similar to NLP:

### Mean-Scale Uniform Bins

The default tokenizer normalizes values and bins them:

```python
config = {
    "tokenizer_class": "MeanScaleUniformBins",
    "tokenizer_kwargs": {
        "low_limit": -15.0,  # Lower bound for binning
        "high_limit": 15.0,   # Upper bound for binning
    },
    # ...
}
```

### Custom Tokenizer

You can customize the tokenization:

```python
config = {
    "tokenizer_class": "MeanScaleUniformBins",
    "tokenizer_kwargs": {
        "low_limit": -10.0,
        "high_limit": 10.0,
        "n_tokens": 4096,  # Vocabulary size
    },
}
```

---

## Visualization

### Multiple Forecast Samples

```python
import matplotlib.pyplot as plt
import numpy as np

avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Plot multiple samples
sample_idx = 0
channel_idx = 0
num_samples_to_plot = 10

history = histories[sample_idx, channel_idx, :]
true = trues[sample_idx, channel_idx, :]

plt.figure(figsize=(14, 5))
plt.plot(range(len(history)), history, label="History", linewidth=2, color='blue')
plt.plot(
    range(len(history), len(history) + len(true)),
    true,
    label="Ground Truth",
    linestyle="--",
    linewidth=2,
    color='green'
)

# Plot multiple forecast samples
for i in range(num_samples_to_plot):
    pred_sample = preds[sample_idx, channel_idx, :, i]
    plt.plot(
        range(len(history), len(history) + len(pred_sample)),
        pred_sample,
        alpha=0.3,
        color='red'
    )

# Plot mean prediction
pred_mean = np.mean(preds[sample_idx, channel_idx, :, :], axis=-1)
plt.plot(
    range(len(history), len(history) + len(pred_mean)),
    pred_mean,
    label="Mean Prediction",
    linewidth=2,
    color='red'
)

plt.legend()
plt.title("Chronos: Multiple Forecast Samples")
plt.grid(alpha=0.3)
plt.show()
```

---

## Tips and Best Practices

### 1. Model Selection
- Use **tiny/mini** for fast inference and experimentation
- Use **small/base** for production applications
- Use **large** for highest accuracy (if resources allow)

### 2. Context Length
- Longer context captures more patterns but is slower
- Start with 512, adjust based on your data

### 3. Number of Samples
- More samples = better uncertainty estimation
- 20-50 samples is usually sufficient
- Use fewer samples for faster inference

### 4. Zero-Shot vs Fine-Tuning
- Try zero-shot first (Chronos is designed for this)
- Fine-tune only if zero-shot performance is insufficient
- Use very small learning rates when fine-tuning

---

## Common Issues

### CUDA Out of Memory

```python
# Use smaller model
config = {
    "model_size": "tiny",  # Instead of "base"
    # ...
}

# Reduce batch size
dataset = ChronosDataset(
    batch_size=4,  # Instead of 16
    # ...
)

# Reduce context length
config = {
    "context_length": 256,  # Instead of 512
    # ...
}
```

### Slow Inference

```python
# Use smaller model
config = {
    "model_size": "mini",
    # ...
}

# Reduce number of samples
config = {
    "num_samples": 10,  # Instead of 50
    # ...
}
```

### Poor Predictions

```python
# Try larger model
config = {
    "model_size": "base",  # Instead of "small"
    # ...
}

# Increase context length
config = {
    "context_length": 1024,  # More context
    # ...
}

# Adjust temperature
config = {
    "temperature": 0.8,  # Less randomness
    # ...
}
```

---

## API Reference

For detailed API documentation, see:

- [ChronosModel API](../api/models.md#chronosmodel)
- [ChronosDataset API](../api/datasets.md#chronosdataset)

---

## Examples

See the [Examples](../examples.md) page for complete working examples.

