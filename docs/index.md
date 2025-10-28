# Samay: Time-series Foundational Models Library

<div align="center" style="margin: 2em 0;">
    <h2>A Unified Interface for Multiple Time-Series Foundational Models</h2>
</div>

## Welcome to Samay

**Samay** is a comprehensive Python library that provides a unified, easy-to-use interface for training and evaluating state-of-the-art time-series foundational models. Whether you're working on forecasting, classification, anomaly detection, or imputation tasks, Samay simplifies the process of leveraging powerful pre-trained models.

---

## ✨ Key Features

- **Unified Interface**: Work with multiple foundational models through a consistent API
- **Pre-trained Models**: Access state-of-the-art pre-trained models ready for zero-shot forecasting
- **Fine-tuning Support**: Easily fine-tune models on your custom datasets
- **Multiple Tasks**: Support for forecasting, classification, anomaly detection, and imputation
- **Flexible Data Handling**: Built-in dataset classes for common time-series formats
- **Easy Integration**: Simple pip installation and minimal code to get started

---

## 🚀 Supported Models

Samay currently supports the following foundational models:

| Model | Paper | Strengths |
|-------|-------|-----------|
| **[LPTM](models/lptm.md)** | [Large Pre-trained Time Series Models](https://arxiv.org/abs/2311.11413) | General-purpose forecasting with segmentation |
| **[MOMENT](models/moment.md)** | [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885) | Multi-task learning (forecasting, classification, anomaly detection) |
| **[TimesFM](models/timesfm.md)** | [A decoder-only foundation model for time-series forecasting](https://arxiv.org/html/2310.10688v2) | Decoder-only architecture by Google Research |
| **[Chronos](models/chronos.md)** | [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815) | Language model-based approach |
| **[MOIRAI](models/moirai.md)** | [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592) | Universal transformer by Salesforce |
| **[TinyTimeMixer](models/ttm.md)** | [TinyTimeMixer: Fast Pre-trained Models for Time Series](https://arxiv.org/abs/2401.03955) | Lightweight and efficient |

---

## 📦 Quick Installation

Install Samay directly from GitHub:

```bash
pip install git+https://github.com/AdityaLab/Samay.git
```

---

## 🔥 Quick Start

Here's a simple example using LPTM for time-series forecasting:

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

# Load the pre-trained model
model = LPTMModel(config)

# Load your dataset
train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="train",
    horizon=192,
)

# Fine-tune the model
finetuned_model = model.finetune(train_dataset)

# Evaluate on test data
test_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/ETTh1.csv",
    mode="test",
    horizon=192,
)

avg_loss, trues, preds, histories = model.evaluate(test_dataset)
print(f"Average Loss: {avg_loss}")
```

---

## 📚 What's Next?

- **[Getting Started](getting-started.md)**: Detailed installation and first steps
- **[Model Guides](models/lptm.md)**: In-depth guides for each supported model
- **[API Reference](api/models.md)**: Complete API documentation
- **[Examples](examples.md)**: Real-world examples and use cases

---

## 🎯 Use Cases

Samay is perfect for:

- **Time-Series Forecasting**: Predict future values from historical data
- **Anomaly Detection**: Identify unusual patterns in time-series data
- **Classification**: Classify time-series sequences into categories
- **Data Imputation**: Fill missing values in time-series data
- **Transfer Learning**: Leverage pre-trained models for your domain-specific tasks

---

## 💡 Why Samay?

Traditional time-series modeling requires extensive expertise and computational resources. Samay democratizes access to state-of-the-art foundational models by:

1. Providing a **unified interface** across multiple model architectures
2. Offering **pre-trained models** that work out-of-the-box
3. Enabling **easy fine-tuning** with minimal code
4. Supporting **multiple tasks** with the same infrastructure

---

## 🤝 Community and Support

- **GitHub**: [AdityaLab/Samay](https://github.com/AdityaLab/Samay)
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/AdityaLab/Samay/issues)
- **Email**: <hkamarthi3@gatech.edu>, <badityap@cc.gatech.edu>

---

## 📝 Citation

If you use Samay in your research, please cite:

```bibtex
@inproceedings{
kamarthi2024large,
title={Large Pre-trained time series models for cross-domain Time series analysis tasks},
author={Harshavardhan Kamarthi and B. Aditya Prakash},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=vMMzjCr5Zj}
}
```

---

## 📋 System Requirements

- **Python**: 3.11-3.13
- **OS**: Linux (CPU + GPU), macOS (CPU)
- **GPU**: NVIDIA GPUs supported

!!! note "Platform Support"
    Windows and Apple Silicon GPU support is planned for future releases.

