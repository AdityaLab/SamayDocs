# Datasets API Reference

This page provides detailed API documentation for all dataset classes in Samay.

---

## LPTMDataset

Dataset class for LPTM model.

::: samay.dataset.LPTMDataset

---

## TimesfmDataset

Dataset class for TimesFM model.

::: samay.dataset.TimesfmDataset

---

## MomentDataset

Dataset class for MOMENT model supporting multiple tasks.

::: samay.dataset.MomentDataset

---

## ChronosDataset

Dataset class for Chronos model with tokenization.

::: samay.dataset.ChronosDataset

---

## MoiraiDataset

Dataset class for MOIRAI model with frequency support.

::: samay.dataset.MoiraiDataset

---

## TinyTimeMixerDataset

Dataset class for TinyTimeMixer model.

::: samay.dataset.TinyTimeMixerDataset

---

## BaseDataset

All datasets inherit from the base dataset class:

::: samay.dataset.BaseDataset

---

## Usage Examples

### Loading a Dataset

```python
from samay.dataset import LPTMDataset

train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
    batchsize=16,
)
```

### Custom Data Splits

```python
# Specify exact boundaries
dataset = LPTMDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
    boundaries=[0, 10000, 15000],  # Train: 0-10k, Val: 10k-15k, Test: 15k-end
)
```

### Getting Data Loader

```python
# Get PyTorch DataLoader
train_loader = train_dataset.get_data_loader()

for batch in train_loader:
    # Process batch
    pass
```

### Accessing Dataset Properties

```python
# Dataset length
print(f"Dataset size: {len(dataset)}")

# Get a single item
sample = dataset[0]

# Number of channels
print(f"Number of channels: {dataset.n_channels}")

# Sequence length
print(f"Sequence length: {dataset.seq_len}")
```

### Denormalizing Predictions

```python
# If dataset normalizes data
normalized_preds = model.evaluate(dataset)[2]

# Denormalize for interpretation
denormalized_preds = dataset._denormalize_data(normalized_preds)
```

---

## Common Parameters

Most dataset classes share these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `None` | Dataset name (for metadata) |
| `datetime_col` | str | Varies | Name of the datetime column in CSV |
| `path` | str | Required | Path to CSV file |
| `mode` | str | `"train"` | Mode: `"train"` or `"test"` |
| `batchsize` | int | Varies | Batch size for DataLoader |
| `boundaries` | list | `[0, 0, 0]` | Custom train/val/test split indices |
| `stride` | int | `10` | Stride for sliding window |

---

## Data Format Requirements

### CSV Structure

All datasets expect CSV files with:
1. A datetime column (configurable name)
2. One or more value columns

Example:
```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,2.009,6.082
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,5.485,1.942,5.947
...
```

### Datetime Formats

Supported datetime formats:
- ISO 8601: `2016-07-01 00:00:00`
- Date only: `2016-07-01`
- Custom formats (parsed by pandas)

### Missing Values

- Some datasets handle missing values automatically
- Others require preprocessing
- Check individual dataset documentation

---

## Model-Specific Dataset Features

### LPTMDataset

- Supports forecasting, classification, and detection
- Configurable sequence length (default: 512)
- Adaptive segmentation

```python
dataset = LPTMDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
    seq_len=512,  # Configurable
    task_name="forecasting",
)
```

### TimesfmDataset

- Frequency specification
- Optional normalization
- Patch-based processing

```python
dataset = TimesfmDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    context_len=512,
    horizon_len=192,
    freq="h",  # Frequency
    normalize=True,  # Optional normalization
)
```

### MomentDataset

- Multi-task support
- Task-specific preprocessing
- Label handling for classification

```python
# Forecasting
dataset = MomentDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon_len=192,
    task_name="forecasting",
)

# Classification
dataset = MomentDataset(
    datetime_col="date",
    path="./data/classification.csv",
    mode="train",
    task_name="classification",
    label_col="label",
)
```

### ChronosDataset

- Tokenization support
- Configurable vocab size
- Drop probability for training

```python
from samay.models.chronosforecasting.chronos import ChronosConfig

config = ChronosConfig(
    context_length=512,
    prediction_length=64,
    # ... other configs
)

dataset = ChronosDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    config=config,
)
```

### MoiraiDataset

- Frequency specification (required)
- Date range filtering
- Built-in normalization

```python
dataset = MoiraiDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    freq="h",  # Required
    context_len=128,
    horizon_len=64,
    start_date="2016-01-01",  # Optional
    end_date="2017-12-31",    # Optional
    normalize=True,
)
```

### TinyTimeMixerDataset

- Large batch support
- Efficient windowing
- Fast data loading

```python
dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    context_len=512,
    horizon_len=96,
    batch_size=128,  # Supports large batches
)
```

---

## Common Methods

All datasets implement these methods:

### `__len__()`

Returns the number of samples in the dataset.

### `__getitem__(idx)`

Returns a single sample at the given index.

### `get_data_loader()`

Returns a PyTorch DataLoader for the dataset.

**Returns:**
- `torch.utils.data.DataLoader`

### `_denormalize_data(data)`

Denormalizes data (if normalization was applied).

**Parameters:**
- `data` (np.ndarray): Normalized data

**Returns:**
- `np.ndarray`: Denormalized data

---

## Data Split Strategies

### Default Split

When `boundaries=[0, 0, 0]`:
- Train: 60% of data
- Validation: 20% of data
- Test: 20% of data

### Custom Split

```python
# Specify exact indices
dataset = LPTMDataset(
    boundaries=[0, 10000, 15000],
    # Train: 0-10000
    # Val: 10000-15000
    # Test: 15000-end
)
```

### Use All Data

```python
# Use entire dataset for training
dataset = LPTMDataset(
    boundaries=[-1, -1, -1],
)
```

---

## Performance Tips

### 1. Batch Size

Larger batch sizes improve throughput:
```python
dataset = LPTMDataset(
    batchsize=64,  # Larger batch
    # ...
)
```

### 2. Stride

Smaller stride creates more samples but is slower:
```python
# More samples (slower)
dataset = LPTMDataset(
    stride=1,
    # ...
)

# Fewer samples (faster)
dataset = LPTMDataset(
    stride=96,
    # ...
)
```

### 3. Normalization

Enable normalization for better performance:
```python
dataset = TimesfmDataset(
    normalize=True,
    # ...
)
```

---

## See Also

- [Models API](models.md): Model classes
- [Metrics API](metrics.md): Evaluation metrics
- [Getting Started](../getting-started.md): Basic usage guide

