# Building imgcom

This document describes how to build and distribute imgcom.

## Prerequisites

- Python 3.6 or higher
- pip
- build tools: `pip install build wheel setuptools`

## Building the Package

### Using the build script (recommended)

```bash
./build.sh
```

This will:
1. Clean previous builds
2. Install build dependencies
3. Build both source distribution (sdist) and wheel
4. Display the results

### Using Make

```bash
make build
```

### Using Python build directly

```bash
python3 -m build
```

## Build Output

After building, you'll find the distribution packages in the `dist/` directory:

- `imgcom-1.0.0.tar.gz` - Source distribution
- `imgcom-1.0.0-py3-none-any.whl` - Universal wheel

## Installing Locally

To install the built package locally:

```bash
pip install dist/imgcom-1.0.0-py3-none-any.whl
```

Or install in development mode:

```bash
pip install -e .
```

With all optional dependencies:

```bash
pip install -e ".[full]"
```

## Testing the Build

After building, test the installation:

```bash
make test
```

Or manually:

```bash
python3 -c "import imgcom; print('✓ Module imported')"
imgcom --help
```

## Distribution

### Upload to PyPI (Test)

```bash
python3 -m pip install twine
python3 -m twine upload --repository testpypi dist/*
```

### Upload to PyPI (Production)

```bash
python3 -m twine upload dist/*
```

## Project Structure

```
imgcom/
├── imgcom.py              # Main application
├── setup.py               # Setuptools configuration
├── pyproject.toml         # Modern Python packaging config
├── requirements.txt       # Dependencies
├── MANIFEST.in           # Files to include in distribution
├── LICENSE               # MIT License
├── README.md             # Documentation
├── batch_config.example.json  # Example batch config
├── Makefile              # Build automation
├── build.sh              # Build script
└── dist/                 # Build output (generated)
```

## Version Management

To update the version, edit:
- `setup.py` - `version` field
- `pyproject.toml` - `version` field

Both should match!

## Clean Build Artifacts

```bash
make clean
```

Or manually:

```bash
rm -rf build/ dist/ *.egg-info __pycache__
```

