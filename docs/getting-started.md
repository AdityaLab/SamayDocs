# Getting Started with Samay

This guide will help you install Samay and run your first time-series forecasting model.

---

## Installation

### Standard Installation

Install Samay directly from GitHub using pip:

```bash
pip install git+https://github.com/AdityaLab/Samay.git
```

This will install Samay and all its dependencies.

---

### Development Installation

If you want to contribute to Samay or modify the code, follow these steps:

```bash
# Clone the repository
git clone https://github.com/AdityaLab/Samay.git
cd Samay

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --reinstall
```

---

## System Requirements

- **Python Version**: 3.11, 3.12, or 3.13
- **Operating Systems**:
    - Linux (CPU and GPU)
    - macOS (CPU only)
- **GPU Support**: NVIDIA GPUs (CUDA-enabled)

!!! warning "Platform Limitations"
    Windows and Apple Silicon GPU support is currently under development.

---

## Your First Forecasting Model

Let's start with a simple example using the **LPTM** model to forecast time-series data.

### Step 1: Import Libraries

```python
from samay.model import LPTMModel
from samay.dataset import LPTMDataset
```

### Step 2: Configure the Model

```python
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,  # Predict 192 time steps ahead
    "freeze_encoder": True,   # Keep encoder frozen
    "freeze_embedder": True,  # Keep embedder frozen
    "freeze_head": False,     # Train the forecasting head
}
```

### Step 3: Load the Pre-trained Model

```python
model = LPTMModel(config)
```

The model is automatically downloaded from the Hugging Face Hub on first use.

### Step 4: Prepare Your Dataset

Samay expects your data in CSV format with:
- A datetime column
- One or more value columns

Example data format (`ETTh1.csv`):

```
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.599,0.462,5.677,2.009,6.082
2016-07-01 01:00:00,5.693,2.076,1.492,0.426,5.485,1.942,5.947
...
```

### Step 5: Load the Dataset

```python
train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
)
```

### Step 6: Fine-tune the Model (Optional)

```python
finetuned_model = model.finetune(train_dataset)
```

During fine-tuning, you'll see training progress:

```
Epoch 0: Train loss: 0.594
Epoch 1: Train loss: 0.504
Epoch 2: Train loss: 0.479
...
```

### Step 7: Evaluate the Model

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

### Step 8: Visualize Results

```python
import matplotlib.pyplot as plt
import numpy as np

# Convert to numpy arrays
trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

# Pick a random example
channel_idx = 0
time_index = 0

history = histories[time_index, channel_idx, :]
true = trues[time_index, channel_idx, :]
pred = preds[time_index, channel_idx, :]

plt.figure(figsize=(12, 4))
plt.plot(range(len(history)), history, label="History", c="darkblue")
plt.plot(
    range(len(history), len(history) + len(true)),
    true,
    label="Ground Truth",
    color="green",
    linestyle="--",
)
plt.plot(
    range(len(history), len(history) + len(pred)),
    pred,
    label="Prediction",
    color="red",
)
plt.legend()
plt.title("Time Series Forecasting with LPTM")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.show()
```

---

## Complete Example

Here's the complete code in one place:

```python
from samay.model import LPTMModel
from samay.dataset import LPTMDataset
import matplotlib.pyplot as plt
import numpy as np

# Configure model
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

# Load model
model = LPTMModel(config)

# Load datasets
train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
)

test_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon=192,
)

# Fine-tune
finetuned_model = model.finetune(train_dataset)

# Evaluate
avg_loss, trues, preds, histories = model.evaluate(test_dataset)
print(f"Average Test Loss: {avg_loss}")

# Visualize
trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

channel_idx = 0
time_index = 0

history = histories[time_index, channel_idx, :]
true = trues[time_index, channel_idx, :]
pred = preds[time_index, channel_idx, :]

plt.figure(figsize=(12, 4))
plt.plot(range(len(history)), history, label="History")
plt.plot(range(len(history), len(history) + len(true)), true, label="Ground Truth", linestyle="--")
plt.plot(range(len(history), len(history) + len(pred)), pred, label="Prediction")
plt.legend()
plt.show()
```

---

## Zero-Shot Forecasting

You can also use pre-trained models directly without fine-tuning:

```python
# Skip the finetune step
avg_loss, trues, preds, histories = model.evaluate(test_dataset)
```

This is called **zero-shot forecasting**, where the pre-trained model makes predictions without any task-specific training.

---

## Dataset Boundaries

By default, Samay splits your data into:
- **Training**: First 60% of data
- **Validation**: Next 20% of data
- **Testing**: Last 20% of data

You can customize these boundaries:

```python
train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
    boundaries=[0, 10000, 15000],  # Custom split points
)
```

---

## Next Steps

Now that you've run your first model, explore:

- **[Model Guides](models/lptm.md)**: Learn about other models (TimesFM, MOMENT, Chronos, etc.)
- **[API Reference](api/models.md)**: Detailed API documentation
- **[Examples](examples.md)**: More advanced use cases

---

## Common Issues

### CUDA Out of Memory

If you encounter GPU memory issues:

```python
config = {
    "task_name": "forecasting",
    "forecast_horizon": 96,  # Reduce horizon
    # ... other configs
}

train_dataset = LPTMDataset(
    # ...
    batchsize=8,  # Reduce batch size
)
```

### Missing Dependencies

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install --upgrade git+https://github.com/AdityaLab/Samay.git
```

---

## Getting Help

- **Documentation**: You're reading it!
- **Examples**: Check the [examples](examples.md) section
- **GitHub Issues**: [Report bugs](https://github.com/AdityaLab/Samay/issues)
- **Email**: <hkamarthi3@gatech.edu>, <badityap@cc.gatech.edu>

