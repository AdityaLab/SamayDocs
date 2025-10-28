# Models API Reference

This page provides detailed API documentation for all model classes in Samay.

---

## LPTMModel

Large Pre-trained Time Series Model for forecasting, classification, and anomaly detection.

::: samay.model.LPTMModel

---

## TimesfmModel

Google's Time Series Foundation Model for zero-shot forecasting.

::: samay.model.TimesfmModel

---

## MomentModel

Multi-task time-series foundation model supporting forecasting, classification, detection, and imputation.

::: samay.model.MomentModel

---

## ChronosModel

Language model-based time-series forecasting with tokenization.

::: samay.model.ChronosModel

---

## MoiraiTSModel

Salesforce's universal time series forecasting transformer.

::: samay.model.MoiraiTSModel

---

## TinyTimeMixerModel

Lightweight and efficient time-series forecasting model.

::: samay.model.TinyTimeMixerModel

---

## Usage Examples

### Loading a Model

```python
from samay.model import LPTMModel

config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = LPTMModel(config)
```

### Fine-Tuning

```python
from samay.dataset import LPTMDataset

train_dataset = LPTMDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
)

finetuned_model = model.finetune(train_dataset, epochs=10)
```

### Evaluation

```python
test_dataset = LPTMDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon=192,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

### Saving and Loading

```python
# Save model
model.save("model_checkpoint.pt")

# Load model
loaded_model = LPTMModel.load("model_checkpoint.pt")
```

---

## Common Parameters

Most model classes share these common parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | dict | Configuration dictionary with model-specific parameters |
| `device` | str | Device to run model on: `"cuda"` or `"cpu"` |
| `repo` | str | Hugging Face repository for pre-trained weights (some models) |

---

## Common Methods

All models implement these methods:

### `finetune(dataset, epochs=5, learning_rate=1e-4, **kwargs)`

Fine-tune the model on a dataset.

**Parameters:**
- `dataset`: Training dataset object
- `epochs` (int): Number of training epochs
- `learning_rate` (float): Learning rate for optimizer
- `**kwargs`: Additional training arguments

**Returns:**
- Trained model

### `evaluate(dataset, metrics=None, **kwargs)`

Evaluate the model on a dataset.

**Parameters:**
- `dataset`: Test dataset object
- `metrics` (list): List of metric names to compute
- `**kwargs`: Additional evaluation arguments

**Returns:**
- `avg_loss` (float): Average loss
- `trues` (np.ndarray): Ground truth values
- `preds` (np.ndarray): Predicted values
- `histories` (np.ndarray): Historical context

### `predict(input_data, **kwargs)`

Make predictions on new data.

**Parameters:**
- `input_data`: Input time series data
- `**kwargs`: Additional prediction arguments

**Returns:**
- Predictions array

### `save(path)`

Save model to disk.

**Parameters:**
- `path` (str): Path to save model

### `load(path)` (class method)

Load model from disk.

**Parameters:**
- `path` (str): Path to load model from

**Returns:**
- Loaded model instance

---

## See Also

- [Datasets API](datasets.md): Dataset classes for data loading
- [Metrics API](metrics.md): Evaluation metrics
- [Model Guides](../models/lptm.md): Detailed guides for each model

