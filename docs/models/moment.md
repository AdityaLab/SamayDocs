# MOMENT: A Family of Open Time-Series Foundation Models

<div align="center" style="margin: 1em 0;">
    <strong>Multi-Task Time-Series Foundation Model</strong>
</div>

---

## Overview

**MOMENT** is a family of open-source time-series foundation models designed to handle multiple tasks including forecasting, classification, anomaly detection, and imputation. It uses a masked autoencoder architecture with a focus on versatility across different time-series domains.

### Paper

[MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885)

### Key Features

- ✅ Multi-task learning (forecasting, classification, detection, imputation)
- ✅ Masked autoencoder architecture
- ✅ Pre-trained on diverse time-series datasets
- ✅ Flexible fine-tuning strategies
- ✅ Open-source and community-driven

---

## Quick Start

### Forecasting

```python
from samay.model import MomentModel
from samay.dataset import MomentDataset

# Model configuration
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

# Load model
model = MomentModel(config)

# Load dataset
train_dataset = MomentDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon_len=192,
    task_name="forecasting",
)

# Fine-tune
finetuned_model = model.finetune(train_dataset)

# Evaluate
test_dataset = MomentDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon_len=192,
    task_name="forecasting",
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

---

## Supported Tasks

MOMENT supports four main tasks:

### 1. Forecasting

Predict future time-series values:

```python
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = MomentModel(config)

dataset = MomentDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon_len=192,
    task_name="forecasting",
)
```

### 2. Classification

Classify time-series sequences:

```python
config = {
    "task_name": "classification",
    "num_classes": 5,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = MomentModel(config)

dataset = MomentDataset(
    datetime_col="date",
    path="./data/classification_data.csv",
    mode="train",
    task_name="classification",
    label_col="label",
)
```

### 3. Anomaly Detection

Detect anomalies in time series:

```python
config = {
    "task_name": "detection",
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = MomentModel(config)

dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_data.csv",
    mode="train",
    task_name="detection",
)
```

### 4. Imputation

Fill missing values:

```python
config = {
    "task_name": "imputation",
    "freeze_encoder": True,
    "freeze_embedder": False,  # May need to train encoder
    "freeze_head": False,
}

model = MomentModel(config)

dataset = MomentDataset(
    datetime_col="date",
    path="./data/incomplete_data.csv",
    mode="train",
    task_name="imputation",
)
```

---

## Configuration Parameters

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_name` | str | `"forecasting"` | Task: `"forecasting"`, `"classification"`, `"detection"`, `"imputation"` |
| `forecast_horizon` | int | `192` | Horizon length for forecasting |
| `num_classes` | int | `None` | Number of classes for classification |
| `freeze_encoder` | bool | `True` | Whether to freeze the encoder |
| `freeze_embedder` | bool | `True` | Whether to freeze the embedder |
| `freeze_head` | bool | `False` | Whether to freeze the task head |
| `dropout` | float | `0.1` | Dropout rate |
| `learning_rate` | float | `1e-4` | Learning rate for fine-tuning |

---

## Dataset

### MomentDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `None` | Dataset name |
| `datetime_col` | str | `None` | Name of datetime column |
| `path` | str | Required | Path to CSV file |
| `mode` | str | `"train"` | `"train"` or `"test"` |
| `horizon_len` | int | `0` | Forecast horizon |
| `task_name` | str | `"forecasting"` | Task type |
| `label_col` | str | `"label"` | Label column for classification |
| `batchsize` | int | `64` | Batch size |
| `boundaries` | list | `[0, 0, 0]` | Custom split boundaries |
| `stride` | int | `10` | Stride for sliding window |

### Data Format

#### Forecasting/Imputation

```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,2.009,6.082
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,5.485,1.942,5.947
...
```

#### Classification

```
date,feature1,feature2,feature3,label
2016-07-01 00:00:00,5.827,2.009,1.599,0
2016-07-01 01:00:00,5.693,2.076,1.492,1
...
```

#### Anomaly Detection

```
date,value,anomaly
2016-07-01 00:00:00,5.827,0
2016-07-01 01:00:00,5.693,0
2016-07-01 02:00:00,12.456,1
...
```

---

## Training

### Forecasting

```python
from samay.model import MomentModel
from samay.dataset import MomentDataset

config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = MomentModel(config)

train_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon_len=192,
    task_name="forecasting",
    batchsize=64,
)

# Fine-tune
finetuned_model = model.finetune(train_dataset, epochs=10)
```

### Classification

```python
config = {
    "task_name": "classification",
    "num_classes": 5,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = MomentModel(config)

train_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_classification.csv",
    mode="train",
    task_name="classification",
    label_col="label",
    batchsize=32,
)

finetuned_model = model.finetune(train_dataset, epochs=20)
```

### Anomaly Detection

```python
config = {
    "task_name": "detection",
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = MomentModel(config)

train_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_anomaly.csv",
    mode="train",
    task_name="detection",
    batchsize=32,
)

finetuned_model = model.finetune(train_dataset, epochs=15)
```

---

## Evaluation

### Forecasting Evaluation

```python
test_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon_len=192,
    task_name="forecasting",
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Calculate metrics
from samay.metric import mse, mae
import numpy as np

trues = np.array(trues)
preds = np.array(preds)

print(f"MSE: {mse(trues, preds):.4f}")
print(f"MAE: {mae(trues, preds):.4f}")
```

### Classification Evaluation

```python
test_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_classification.csv",
    mode="test",
    task_name="classification",
    label_col="label",
)

avg_loss, labels, predictions, _ = model.evaluate(test_dataset)

# Calculate accuracy
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
```

### Anomaly Detection Evaluation

```python
test_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_anomaly.csv",
    mode="test",
    task_name="detection",
)

avg_loss, true_labels, anomaly_scores, _ = model.evaluate(test_dataset)

# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(true_labels, anomaly_scores)
print(f"ROC-AUC: {auc:.4f}")
```

---

## Advanced Usage

### Multi-Task Transfer Learning

Train on one task, fine-tune on another:

```python
# First, train on forecasting
config_forecast = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": False,
    "freeze_embedder": False,
    "freeze_head": False,
}

model = MomentModel(config_forecast)
forecast_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon_len=192,
    task_name="forecasting",
)

model.finetune(forecast_dataset, epochs=10)

# Now switch to classification
config_class = {
    "task_name": "classification",
    "num_classes": 5,
    "freeze_encoder": True,  # Freeze what we learned
    "freeze_embedder": True,
    "freeze_head": False,
}

# Update model config
model.update_config(config_class)

class_dataset = MomentDataset(
    datetime_col="date",
    path="./data/classification.csv",
    mode="train",
    task_name="classification",
    label_col="label",
)

model.finetune(class_dataset, epochs=5)
```

### Custom Training Loop

```python
import torch
from torch.optim import AdamW

model = MomentModel(config)
train_loader = train_dataset.get_data_loader()

optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Unpack batch based on task
        if config["task_name"] == "forecasting":
            timeseries, input_mask, forecast = batch
            predictions = model(timeseries, input_mask)
            loss = criterion(predictions, forecast)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")
```

---

## Visualization

### Forecasting Results

```python
import matplotlib.pyplot as plt
import numpy as np

avg_loss, trues, preds, histories = model.evaluate(test_dataset)

trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

# Plot multiple channels
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

plt.suptitle("MOMENT Multi-Channel Forecasting")
plt.tight_layout()
plt.show()
```

### Anomaly Detection Results

```python
import matplotlib.pyplot as plt

# Get anomaly scores
avg_loss, true_labels, anomaly_scores, sequences = model.evaluate(test_dataset)

plt.figure(figsize=(14, 6))

# Plot time series
plt.subplot(2, 1, 1)
plt.plot(sequences[0], label="Time Series")
plt.scatter(
    np.where(true_labels[0] == 1)[0],
    sequences[0][true_labels[0] == 1],
    color='red',
    label='True Anomalies',
    s=100
)
plt.legend()
plt.title("Time Series with Anomalies")

# Plot anomaly scores
plt.subplot(2, 1, 2)
plt.plot(anomaly_scores[0], label="Anomaly Score")
plt.axhline(y=np.percentile(anomaly_scores[0], 95), color='r', linestyle='--', label='Threshold (95%)')
plt.legend()
plt.title("Anomaly Scores")

plt.tight_layout()
plt.show()
```

---

## Tips and Best Practices

### 1. Task Selection
- Use **forecasting** for predicting future values
- Use **classification** for categorizing sequences
- Use **detection** for identifying anomalies
- Use **imputation** for filling missing data

### 2. Fine-Tuning Strategy
- Start with frozen encoder and embedder
- Gradually unfreeze if performance is poor
- Use higher learning rates for task heads

### 3. Sequence Length
- MOMENT uses fixed sequence length of 512
- Longer sequences capture more context
- Shorter sequences are faster

### 4. Multi-Task Learning
- Pre-train on related tasks
- Fine-tune on target task
- Freeze learned features when switching tasks

---

## Common Issues

### Poor Classification Performance

Try unfreezing more layers:
```python
config = {
    "task_name": "classification",
    "num_classes": 5,
    "freeze_encoder": False,  # Unfreeze encoder
    "freeze_embedder": False, # Unfreeze embedder
    "freeze_head": False,
}
```

### High Memory Usage

Reduce batch size:
```python
dataset = MomentDataset(
    # ...
    batchsize=16,  # Instead of 64
)
```

### Missing Values

MOMENT handles missing values through imputation:
```python
config = {
    "task_name": "imputation",
    # ...
}
```

---

## API Reference

For detailed API documentation, see:

- [MomentModel API](../api/models.md#momentmodel)
- [MomentDataset API](../api/datasets.md#momentdataset)

---

## Examples

See the [Examples](../examples.md) page for complete working examples of all tasks.

