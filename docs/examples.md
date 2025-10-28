# Examples

This page provides complete, working examples for using Samay with different models and tasks.

---

## Table of Contents

1. [Basic Forecasting with LPTM](#basic-forecasting-with-lptm)
2. [Zero-Shot Forecasting with TimesFM](#zero-shot-forecasting-with-timesfm)
3. [Multi-Task Learning with MOMENT](#multi-task-learning-with-moment)
4. [Probabilistic Forecasting with Chronos](#probabilistic-forecasting-with-chronos)
5. [Universal Forecasting with MOIRAI](#universal-forecasting-with-moirai)
6. [Fast Forecasting with TinyTimeMixer](#fast-forecasting-with-tinytimemixer)
7. [Anomaly Detection](#anomaly-detection)
8. [Time Series Classification](#time-series-classification)
9. [Multi-Horizon Forecasting](#multi-horizon-forecasting)
10. [Cross-Domain Transfer Learning](#cross-domain-transfer-learning)

---

## Basic Forecasting with LPTM

A complete example of loading data, fine-tuning, and evaluating LPTM.

```python
from samay.model import LPTMModel
from samay.dataset import LPTMDataset
from samay.metric import mse, mae, mape
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Configure the model
config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
    "head_dropout": 0.1,
}

# Step 2: Load the model
print("Loading LPTM model...")
model = LPTMModel(config)

# Step 3: Load training data
print("Loading training data...")
train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
    batchsize=16,
)

# Step 4: Fine-tune the model
print("Fine-tuning model...")
finetuned_model = model.finetune(
    train_dataset,
    epochs=10,
    learning_rate=1e-4,
)

# Step 5: Load test data
print("Loading test data...")
test_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon=192,
)

# Step 6: Evaluate
print("Evaluating model...")
avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Step 7: Calculate metrics
trues = np.array(trues)
preds = np.array(preds)
histories = np.array(histories)

print("\n=== Evaluation Results ===")
print(f"Average Loss: {avg_loss:.4f}")
print(f"MSE: {mse(trues, preds):.4f}")
print(f"MAE: {mae(trues, preds):.4f}")
print(f"MAPE: {mape(trues, preds):.4f}%")

# Step 8: Visualize results
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
    color="green"
)
plt.plot(
    range(len(history), len(history) + len(pred)),
    pred,
    label="Prediction (192 steps)",
    linewidth=2,
    color="red"
)
plt.axvline(x=len(history), color='gray', linestyle=':', alpha=0.5)
plt.legend()
plt.title("LPTM Time Series Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lptm_forecast.png", dpi=300)
plt.show()

print("\nForecast plot saved as 'lptm_forecast.png'")
```

---

## Zero-Shot Forecasting with TimesFM

Example of using TimesFM without any training.

```python
from samay.model import TimesfmModel
from samay.dataset import TimesfmDataset
from samay.metric import mse, mae
import numpy as np

# Configure TimesFM
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

print("Loading TimesFM model...")
tfm = TimesfmModel(config=config, repo=repo)

# Load test data (no training needed!)
print("Loading test data...")
test_dataset = TimesfmDataset(
    name="ett",
    datetime_col='date',
    path='data/ETTh1.csv',
    mode='test',
    context_len=config["context_len"],
    horizon_len=config["horizon_len"],
    freq="h",
)

# Zero-shot evaluation
print("Running zero-shot evaluation...")
avg_loss, trues, preds, histories = tfm.evaluate(test_dataset)

# Calculate metrics
trues = np.array(trues)
preds = np.array(preds)

print("\n=== Zero-Shot Results ===")
print(f"Average Loss: {avg_loss:.4f}")
print(f"MSE: {mse(trues, preds):.4f}")
print(f"MAE: {mae(trues, preds):.4f}")
```

---

## Multi-Task Learning with MOMENT

Example showing forecasting and anomaly detection with the same model.

```python
from samay.model import MomentModel
from samay.dataset import MomentDataset
import numpy as np

# ===== Task 1: Forecasting =====
print("=== Task 1: Forecasting ===\n")

config_forecast = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

model = MomentModel(config_forecast)

train_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon_len=192,
    task_name="forecasting",
)

print("Fine-tuning for forecasting...")
model.finetune(train_dataset, epochs=10)

test_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon_len=192,
    task_name="forecasting",
)

avg_loss, trues, preds, _ = model.evaluate(test_dataset)
print(f"Forecasting Loss: {avg_loss:.4f}\n")

# ===== Task 2: Anomaly Detection =====
print("=== Task 2: Anomaly Detection ===\n")

config_detection = {
    "task_name": "detection",
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

# Update model for new task
model.update_config(config_detection)

anomaly_train = MomentDataset(
    datetime_col="date",
    path="./data/ecg_anomaly.csv",
    mode="train",
    task_name="detection",
)

print("Fine-tuning for anomaly detection...")
model.finetune(anomaly_train, epochs=10)

anomaly_test = MomentDataset(
    datetime_col="date",
    path="./data/ecg_anomaly.csv",
    mode="test",
    task_name="detection",
)

avg_loss, labels, scores, _ = model.evaluate(anomaly_test)

# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(labels, scores)
print(f"Anomaly Detection AUC: {auc:.4f}")
```

---

## Probabilistic Forecasting with Chronos

Example of generating prediction intervals with Chronos.

```python
from samay.model import ChronosModel
from samay.dataset import ChronosDataset
import matplotlib.pyplot as plt
import numpy as np

# Configure for probabilistic forecasting
config = {
    "model_size": "small",
    "context_length": 512,
    "prediction_length": 96,
    "num_samples": 100,  # Generate 100 samples
    "temperature": 1.0,
}

print("Loading Chronos model...")
model = ChronosModel(config)

# Load test data
test_dataset = ChronosDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    config=config,
)

# Generate probabilistic forecasts
print("Generating probabilistic forecasts...")
avg_loss, trues, preds, histories = model.evaluate(test_dataset)

# Calculate statistics across samples
mean_pred = np.mean(preds, axis=-1)
std_pred = np.std(preds, axis=-1)
lower_10 = np.percentile(preds, 10, axis=-1)
lower_25 = np.percentile(preds, 25, axis=-1)
upper_75 = np.percentile(preds, 75, axis=-1)
upper_90 = np.percentile(preds, 90, axis=-1)

# Visualize with prediction intervals
sample_idx = 0
channel_idx = 0

history = histories[sample_idx, channel_idx, :]
true = trues[sample_idx, channel_idx, :]
pred_mean = mean_pred[sample_idx, channel_idx, :]
pred_lower_10 = lower_10[sample_idx, channel_idx, :]
pred_lower_25 = lower_25[sample_idx, channel_idx, :]
pred_upper_75 = upper_75[sample_idx, channel_idx, :]
pred_upper_90 = upper_90[sample_idx, channel_idx, :]

plt.figure(figsize=(14, 6))

# Plot history
plt.plot(range(len(history)), history, label="History", linewidth=2, color='blue')

# Plot ground truth
forecast_range = range(len(history), len(history) + len(true))
plt.plot(forecast_range, true, label="Ground Truth", linestyle="--", linewidth=2, color='green')

# Plot mean prediction
plt.plot(forecast_range, pred_mean, label="Mean Prediction", linewidth=2, color='red')

# Plot prediction intervals
plt.fill_between(
    forecast_range,
    pred_lower_10,
    pred_upper_90,
    alpha=0.2,
    color='red',
    label="80% Prediction Interval"
)
plt.fill_between(
    forecast_range,
    pred_lower_25,
    pred_upper_75,
    alpha=0.3,
    color='red',
    label="50% Prediction Interval"
)

plt.axvline(x=len(history), color='gray', linestyle=':', alpha=0.5)
plt.legend()
plt.title("Chronos Probabilistic Forecasting with Prediction Intervals")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("chronos_probabilistic.png", dpi=300)
plt.show()

print("\nPrediction interval plot saved as 'chronos_probabilistic.png'")
```

---

## Universal Forecasting with MOIRAI

Example of using MOIRAI across different frequencies.

```python
from samay.model import MoiraiTSModel
from samay.dataset import MoiraiDataset
import numpy as np

# Configure MOIRAI
repo = "Salesforce/moirai-1.0-R-small"
config = {
    "context_len": 128,
    "horizon_len": 64,
    "model_type": "moirai",
    "model_size": "small",
}

print("Loading MOIRAI model...")
moirai_model = MoiraiTSModel(repo=repo, config=config)

# Test on different frequencies
frequencies = [
    ("Hourly", "h", "data/hourly.csv"),
    ("Daily", "d", "data/daily.csv"),
    ("Weekly", "w", "data/weekly.csv"),
]

results = {}

for freq_name, freq_code, data_path in frequencies:
    print(f"\n=== Testing on {freq_name} Data ===")
    
    test_dataset = MoiraiDataset(
        mode="test",
        path=data_path,
        datetime_col="date",
        freq=freq_code,
        context_len=128,
        horizon_len=64,
    )
    
    eval_results, trues, preds, histories = moirai_model.evaluate(
        test_dataset,
        metrics=["MSE", "MAE", "MASE"]
    )
    
    results[freq_name] = eval_results
    
    print(f"Results for {freq_name}:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")

# Summary
print("\n=== Summary Across Frequencies ===")
for freq_name, metrics in results.items():
    print(f"{freq_name}: MAE = {metrics['MAE']:.4f}")
```

---

## Fast Forecasting with TinyTimeMixer

Example emphasizing speed and efficiency.

```python
from samay.model import TinyTimeMixerModel
from samay.dataset import TinyTimeMixerDataset
import time
import numpy as np

# Configure for speed
config = {
    "context_len": 512,
    "horizon_len": 96,
    "model_size": "tiny",
}

print("Loading TinyTimeMixer model...")
model = TinyTimeMixerModel(config)

# Load data with large batch size
train_dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    context_len=512,
    horizon_len=96,
    batch_size=256,  # Large batch for speed
)

# Fast training
print("Training (10 epochs)...")
start_time = time.time()
model.finetune(train_dataset, epochs=10)
train_time = time.time() - start_time
print(f"Training time: {train_time:.2f}s")

# Fast inference
test_dataset = TinyTimeMixerDataset(
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    context_len=512,
    horizon_len=96,
    batch_size=256,
)

print("Running inference...")
start_time = time.time()
avg_loss, trues, preds, histories = model.evaluate(test_dataset)
inference_time = time.time() - start_time

print(f"\n=== Performance Summary ===")
print(f"Training time: {train_time:.2f}s")
print(f"Inference time: {inference_time:.2f}s")
print(f"Test Loss: {avg_loss:.4f}")
```

---

## Anomaly Detection

Complete anomaly detection example with visualization.

```python
from samay.model import MomentModel
from samay.dataset import MomentDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# Configure for anomaly detection
config = {
    "task_name": "detection",
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

print("Loading MOMENT for anomaly detection...")
model = MomentModel(config)

# Load training data
train_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_anomaly.csv",
    mode="train",
    task_name="detection",
)

print("Training anomaly detector...")
model.finetune(train_dataset, epochs=15)

# Load test data
test_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_anomaly.csv",
    mode="test",
    task_name="detection",
)

# Detect anomalies
print("Detecting anomalies...")
avg_loss, true_labels, anomaly_scores, sequences = model.evaluate(test_dataset)

# Calculate metrics
auc = roc_auc_score(true_labels, anomaly_scores)
precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)

print(f"\n=== Anomaly Detection Results ===")
print(f"ROC-AUC: {auc:.4f}")

# Find optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Best F1 Score: {f1_scores[optimal_idx]:.4f}")

# Visualize results
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot time series with anomalies
axes[0].plot(sequences[0], label="Time Series", linewidth=1)
anomaly_indices = np.where(true_labels[0] == 1)[0]
axes[0].scatter(
    anomaly_indices,
    sequences[0][anomaly_indices],
    color='red',
    s=100,
    label='True Anomalies',
    zorder=5
)
axes[0].set_title("Time Series with True Anomalies")
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel("Value")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot anomaly scores
axes[1].plot(anomaly_scores[0], label="Anomaly Score", color='orange')
axes[1].axhline(
    y=optimal_threshold,
    color='r',
    linestyle='--',
    label=f'Threshold ({optimal_threshold:.2f})'
)
axes[1].set_title("Anomaly Scores")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Anomaly Score")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot precision-recall curve
axes[2].plot(recall, precision, linewidth=2)
axes[2].scatter(
    recall[optimal_idx],
    precision[optimal_idx],
    color='red',
    s=100,
    zorder=5,
    label=f'Optimal Point (F1={f1_scores[optimal_idx]:.3f})'
)
axes[2].set_title("Precision-Recall Curve")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("anomaly_detection.png", dpi=300)
plt.show()

print("\nAnomal detection plot saved as 'anomaly_detection.png'")
```

---

## Time Series Classification

Example of classifying time series sequences.

```python
from samay.model import MomentModel
from samay.dataset import MomentDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure for classification
config = {
    "task_name": "classification",
    "num_classes": 5,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

print("Loading MOMENT for classification...")
model = MomentModel(config)

# Load training data
train_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_classification.csv",
    mode="train",
    task_name="classification",
    label_col="label",
)

print("Training classifier...")
model.finetune(train_dataset, epochs=20)

# Load test data
test_dataset = MomentDataset(
    datetime_col="date",
    path="./data/ecg_classification.csv",
    mode="test",
    task_name="classification",
    label_col="label",
)

# Classify
print("Classifying test data...")
avg_loss, true_labels, predictions, _ = model.evaluate(test_dataset)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
print(f"\n=== Classification Results ===")
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(true_labels, predictions))

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[f'Class {i}' for i in range(config["num_classes"])],
    yticklabels=[f'Class {i}' for i in range(config["num_classes"])]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("classification_confusion_matrix.png", dpi=300)
plt.show()

print("\nConfusion matrix saved as 'classification_confusion_matrix.png'")
```

---

## Multi-Horizon Forecasting

Example of forecasting at different horizons.

```python
from samay.model import LPTMModel
from samay.dataset import LPTMDataset
from samay.metric import mae
import matplotlib.pyplot as plt
import numpy as np

# Test different forecast horizons
horizons = [96, 192, 336, 720]
results = {}

for horizon in horizons:
    print(f"\n=== Testing Horizon: {horizon} ===")
    
    # Configure model
    config = {
        "task_name": "forecasting",
        "forecast_horizon": horizon,
        "freeze_encoder": True,
        "freeze_embedder": True,
        "freeze_head": False,
    }
    
    model = LPTMModel(config)
    
    # Load data
    train_dataset = LPTMDataset(
        datetime_col="date",
        path="./data/ETTh1.csv",
        mode="train",
        horizon=horizon,
    )
    
    test_dataset = LPTMDataset(
        datetime_col="date",
        path="./data/ETTh1.csv",
        mode="test",
        horizon=horizon,
    )
    
    # Train and evaluate
    model.finetune(train_dataset, epochs=10)
    avg_loss, trues, preds, _ = model.evaluate(test_dataset)
    
    # Calculate MAE
    mae_score = mae(np.array(trues), np.array(preds))
    results[horizon] = mae_score
    
    print(f"MAE for horizon {horizon}: {mae_score:.4f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o', linewidth=2, markersize=8)
plt.xlabel("Forecast Horizon")
plt.ylabel("MAE")
plt.title("Forecasting Performance vs. Horizon Length")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("multi_horizon_results.png", dpi=300)
plt.show()

print("\nMulti-horizon results saved as 'multi_horizon_results.png'")
```

---

## Cross-Domain Transfer Learning

Example of transferring knowledge across domains.

```python
from samay.model import LPTMModel
from samay.dataset import LPTMDataset
from samay.metric import mae
import numpy as np

# Step 1: Train on source domain
print("=== Step 1: Training on Source Domain ===")

config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": False,  # Train all layers
    "freeze_embedder": False,
    "freeze_head": False,
}

model = LPTMModel(config)

source_train = LPTMDataset(
    datetime_col="date",
    path="./data/source_domain.csv",
    mode="train",
    horizon=192,
)

print("Training on source domain...")
model.finetune(source_train, epochs=20)

# Evaluate on source domain
source_test = LPTMDataset(
    datetime_col="date",
    path="./data/source_domain.csv",
    mode="test",
    horizon=192,
)

_, trues, preds, _ = model.evaluate(source_test)
source_mae = mae(np.array(trues), np.array(preds))
print(f"Source domain MAE: {source_mae:.4f}")

# Step 2: Transfer to target domain
print("\n=== Step 2: Transferring to Target Domain ===")

# Freeze encoder and embedder, only train head
config["freeze_encoder"] = True
config["freeze_embedder"] = True
config["freeze_head"] = False

model.update_config(config)

target_train = LPTMDataset(
    datetime_col="date",
    path="./data/target_domain.csv",
    mode="train",
    horizon=192,
)

print("Fine-tuning on target domain...")
model.finetune(target_train, epochs=5)

# Evaluate on target domain
target_test = LPTMDataset(
    datetime_col="date",
    path="./data/target_domain.csv",
    mode="test",
    horizon=192,
)

_, trues, preds, _ = model.evaluate(target_test)
target_mae = mae(np.array(trues), np.array(preds))
print(f"Target domain MAE: {target_mae:.4f}")

# Step 3: Compare with training from scratch
print("\n=== Step 3: Training from Scratch on Target ===")

config_scratch = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "freeze_encoder": False,
    "freeze_embedder": False,
    "freeze_head": False,
}

model_scratch = LPTMModel(config_scratch)

print("Training from scratch on target domain...")
model_scratch.finetune(target_train, epochs=5)

_, trues_scratch, preds_scratch, _ = model_scratch.evaluate(target_test)
scratch_mae = mae(np.array(trues_scratch), np.array(preds_scratch))
print(f"From-scratch MAE: {scratch_mae:.4f}")

# Summary
print("\n=== Transfer Learning Summary ===")
print(f"Source domain MAE: {source_mae:.4f}")
print(f"Target with transfer: {target_mae:.4f}")
print(f"Target from scratch: {scratch_mae:.4f}")
improvement = ((scratch_mae - target_mae) / scratch_mae) * 100
print(f"Improvement from transfer learning: {improvement:.2f}%")
```

---

## Additional Resources

For more examples, check out:

- [Jupyter Notebooks](https://github.com/AdityaLab/Samay/tree/main/example) in the repository
- [Model Guides](models/lptm.md) for model-specific examples
- [API Reference](api/models.md) for detailed method documentation

---

## Tips for Examples

1. **Start Simple**: Begin with basic forecasting before trying advanced tasks
2. **Visualize Results**: Always plot your predictions to verify they make sense
3. **Monitor Training**: Watch for overfitting by tracking validation loss
4. **Experiment**: Try different model configurations and hyperparameters
5. **Save Models**: Save checkpoints of well-performing models for reuse

---

## Need Help?

If you encounter issues with these examples:

- Check the [Getting Started](getting-started.md) guide for setup
- Review [Model Guides](models/lptm.md) for model-specific details
- Open an issue on [GitHub](https://github.com/AdityaLab/Samay/issues)
- Contact us at <hkamarthi3@gatech.edu> or <badityap@cc.gatech.edu>

