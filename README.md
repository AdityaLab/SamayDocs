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

### Deploying to GitHub Pages

To deploy the documentation to the `gh-pages` branch (which makes it available at the GitHub Pages URL):

```bash
uv run mkdocs gh-deploy
```

This command will:
1. Build the documentation site
2. Commit the built site to the `gh-pages` branch
3. Push the changes to GitHub

After deployment, the documentation will be available at the configured GitHub Pages URL (typically `https://<username>.github.io/<repository-name>/`).

**Note:** Ensure you have push access to the repository and that GitHub Pages is enabled in your repository settings.

### Making Changes

This repository includes the Samay library as a submodule, so changes may involve updating both repositories. Follow these guidelines:

#### Updating the Samay Submodule

If you've made changes to the Samay library or need to pull the latest changes:

```bash
# Navigate to the submodule directory
cd Samay

# Pull latest changes from the Samay repository
git pull origin main  # or the appropriate branch

# Return to the docs repository
cd ..
```

Or update the submodule from the root directory:

```bash
git submodule update --remote Samay
```

#### Working with Documentation Changes

1. **Edit documentation files**: Modify the Markdown files in the `docs/` directory
2. **Update API documentation**: API docs are auto-generated from source code in `Samay/src/samay/`. If you modify Samay source code, the API docs will automatically reflect those changes.
3. **Preview changes**: Use `uv run mkdocs serve` to see your changes in real-time
4. **Test the build**: Run `uv run mkdocs build` to ensure everything builds correctly

#### Committing Changes

When making changes that affect both repositories:

**If you only modified documentation files:**
```bash
# In the SamayDocs repository root
git add docs/
git commit -m "Update documentation"
git push
```

**If you modified both Samay and documentation:**

1. **First, commit and push Samay changes:**
   ```bash
   cd Samay
   git add .
   git commit -m "Update Samay code"
   git push
   cd ..
   ```

2. **Then, update the submodule reference in SamayDocs:**
   ```bash
   git add Samay
   git commit -m "Update Samay submodule to latest version"
   git push
   ```

**If you only need to update the submodule to point to a new commit:**
```bash
# After pulling latest changes in Samay submodule
git add Samay
git commit -m "Update Samay submodule"
git push
```

#### Pulling Latest Changes

To get the latest changes from both repositories:

```bash
# Pull SamayDocs changes
git pull

# Update submodule to latest commit
git submodule update --init --recursive

# Or pull latest changes in the submodule
cd Samay
git pull
cd ..
```

#### Key Files to Know

- `docs/index.md` - Homepage
- `docs/getting-started.md` - Installation guide
- `docs/models/` - Model-specific documentation
- `docs/api/` - API reference pages (auto-generated from `Samay/src/samay/`)
- `mkdocs.yml` - Configuration file
- `Samay/src/samay/` - Samay source code (submodule)

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