# Samay Documentation

This repository contains comprehensive documentation for the Samay time-series forecasting library.

**View the documentation online:** [https://adityalab.github.io/SamayDocs/](https://adityalab.github.io/SamayDocs/)

## Overview

Samay is a unified Python library that provides a consistent interface for multiple time-series foundational models including LPTM, MOMENT, TimesFM, Chronos, MOIRAI, and TinyTimeMixer.

## Setup

### Cloning the Repository

To clone this repository along with the Samay submodule, use:

```bash
git clone --recurse-submodules https://github.com/AdityaLab/SamayDocs.git
cd SamayDocs
```

### Installing Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install all dependencies with:

```bash
uv sync
```

This will install all dependencies including the Samay library from the workspace submodule.

### Building and Serving Documentation Locally

To preview the documentation locally with live reload (changes will automatically refresh):

```bash
uv run mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000` in your browser.

To build the static documentation site:

```bash
uv run mkdocs build
```

The built site will be in the `site/` directory.

### Making Changes

To contribute to the documentation:

1. **Edit documentation files**: Modify the Markdown files in the `docs/` directory
2. **Update API documentation**: API docs are auto-generated from source code in `Samay/src/samay/`
3. **Preview changes**: Use `uv run mkdocs serve` to see your changes in real-time
4. **Test the build**: Run `uv run mkdocs build` to ensure everything builds correctly

Key files to know:
- `docs/index.md` - Homepage
- `docs/getting-started.md` - Installation guide
- `docs/models/` - Model-specific documentation
- `docs/api/` - API reference pages
- `mkdocs.yml` - Configuration file

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