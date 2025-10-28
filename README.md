# Samay Documentation

This repository contains comprehensive documentation for the Samay time-series forecasting library.

**View the documentation online:** [https://adityalab.github.io/SamayDocs/](https://adityalab.github.io/SamayDocs/)

## Overview

Samay is a unified Python library that provides a consistent interface for multiple time-series foundational models including LPTM, MOMENT, TimesFM, Chronos, MOIRAI, and TinyTimeMixer.

## Documentation Structure

The documentation includes:

- **Getting Started Guide** - Installation and first forecasting example
- **Model Guides** - Detailed documentation for all 6 supported models
- **API Reference** - Auto-generated API documentation for models, datasets, and metrics
- **Examples** - Complete working examples for forecasting, anomaly detection, classification, and more

## Documentation Features

- Material theme with custom styling
- Full-text search functionality
- Auto-generated API documentation from source code
- Complete code examples with syntax highlighting
- External links automatically open in new tabs
- Mobile-responsive design

## Repository Structure

```
.
├── mkdocs.yml                  # MkDocs configuration
├── requirements.txt            # Documentation dependencies
├── docs/
│   ├── index.md               # Homepage
│   ├── getting-started.md     # Installation and quick start
│   ├── examples.md            # Complete examples
│   ├── models/                # Model-specific guides
│   │   ├── lptm.md
│   │   ├── timesfm.md
│   │   ├── moment.md
│   │   ├── chronos.md
│   │   ├── moirai.md
│   │   └── ttm.md
│   ├── api/                   # API reference
│   │   ├── models.md
│   │   ├── datasets.md
│   │   └── metrics.md
│   └── javascripts/
│       └── external-links.js  # External link handler
└── Samay/                     # Source code
```

## Main Repository

The main Samay library repository: [github.com/AdityaLab/Samay](https://github.com/AdityaLab/Samay)

## License

This documentation follows the same license as the Samay project.
