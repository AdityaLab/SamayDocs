# Samay Documentation

This directory contains the source files for the Samay documentation website.

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `mkdocs-material` - Material theme for MkDocs
- `mkdocstrings[python]` - Python API documentation generator
- `mkdocs-jupyter` - Jupyter notebook support

### Local Development

To preview the documentation locally:

```bash
mkdocs serve
```

This will start a local server at `http://127.0.0.1:8000/`. The site will automatically reload when you make changes to the documentation files.

### Building the Site

To build the static site:

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML files.

### Deploying to GitHub Pages

To deploy to GitHub Pages:

```bash
mkdocs gh-deploy
```

This builds the documentation and pushes it to the `gh-pages` branch.

## Documentation Structure

```
docs/
├── index.md                    # Homepage
├── getting-started.md          # Installation and first steps
├── models/                     # Model-specific guides
│   ├── lptm.md
│   ├── timesfm.md
│   ├── moment.md
│   ├── chronos.md
│   ├── moirai.md
│   └── ttm.md
├── api/                        # API reference
│   ├── models.md
│   ├── datasets.md
│   └── metrics.md
└── examples.md                 # Complete examples
```

## Writing Documentation

### Markdown Files

Documentation is written in Markdown with some extensions:

- **Code blocks**: Use triple backticks with language identifier
- **Admonitions**: Use `!!! note`, `!!! warning`, `!!! tip`, etc.
- **Tables**: Use standard Markdown tables
- **Math**: Use LaTeX math with `\(` and `\)` for inline, `\[` and `\]` for blocks

### API Documentation

API documentation is auto-generated using `mkdocstrings`. To include API docs for a class or function:

```markdown
::: samay.model.LPTMModel
```

This will automatically generate documentation from the docstrings.

### Code Examples

Include code examples with syntax highlighting:

````markdown
```python
from samay.model import LPTMModel

model = LPTMModel(config)
```
````

## Customization

### Theme Configuration

The theme is configured in `mkdocs.yml`. Key settings:

- `theme.palette.primary`: Primary color (indigo)
- `theme.palette.accent`: Accent color (blue)
- `theme.features`: Enabled features (navigation, search, etc.)

### Navigation

The navigation structure is defined in the `nav` section of `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Models:
      - LPTM: models/lptm.md
      # ... more models
  - API Reference:
      - Models: api/models.md
      # ... more API pages
  - Examples: examples.md
```

## Contributing

When contributing to documentation:

1. Follow the existing structure and style
2. Include code examples for new features
3. Update the API reference if adding new classes/methods
4. Test locally with `mkdocs serve` before committing
5. Ensure all links work correctly

## Tips

- Use **bold** for emphasis on important terms
- Use `code` for inline code, function names, and parameters
- Use admonitions for warnings, tips, and notes
- Include working code examples
- Add visualizations where helpful
- Keep sections focused and concise

## Getting Help

If you need help with the documentation:

- Check the [MkDocs documentation](https://www.mkdocs.org/)
- Review [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- Look at [mkdocstrings](https://mkdocstrings.github.io/)
- Open an issue on GitHub

## License

The documentation is part of the Samay project and follows the same license.

