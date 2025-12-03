.PHONY: help install install-dev build clean test dist upload

help:
	@echo "imgcom - Professional Image Combiner & Stitcher"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install imgcom in development mode"
	@echo "  install-dev  - Install with all optional dependencies"
	@echo "  build        - Build distribution packages"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run basic tests"
	@echo "  dist         - Create source and wheel distributions"
	@echo "  upload       - Upload to PyPI (requires credentials)"

install:
	pip install -e .

install-dev:
	pip install -e ".[full]"

build:
	python3 -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf __pycache__
	rm -f *.pyc
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

test:
	@echo "Testing imgcom installation..."
	python3 -c "import imgcom; print('✓ imgcom module imported successfully')"
	@echo "Testing CLI..."
	imgcom --help > /dev/null && echo "✓ CLI command works" || echo "✗ CLI command failed"

dist: clean
	python3 -m build
	@echo ""
	@echo "Distribution packages created in dist/:"
	@ls -lh dist/

upload: dist
	python3 -m twine upload dist/*

