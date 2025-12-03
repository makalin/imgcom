#!/bin/bash
# Build script for imgcom

set -e

echo "🔨 Building imgcom..."
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info __pycache__

# Install build dependencies
echo ""
echo "Installing build dependencies..."
pip3 install --upgrade build wheel setuptools

# Build the package
echo ""
echo "Building package..."
python3 -m build

# Show results
echo ""
echo "✅ Build complete!"
echo ""
echo "Distribution packages:"
ls -lh dist/

echo ""
echo "To install locally:"
echo "  pip3 install dist/imgcom-*.whl"
echo ""
echo "To upload to PyPI:"
echo "  python3 -m twine upload dist/*"

