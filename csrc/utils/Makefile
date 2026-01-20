# Makefile for building CUDA extensions in utils directory
#
# Usage:
#   make              - Build all extensions (create_block_mask)
#   make all          - Same as above
#   make block_mask   - Build only create_block_mask extension
#   make test         - Run create_block_mask tests (manual scripts)
#   make pytest            - Run create_block_mask pytest tests
#   make clean        - Clean create_block_mask build artifacts

.PHONY: all block_mask test pytest bench bench_block_mask clean

# Default target: build all
all: block_mask

# Build create_block_mask extension
block_mask:
	@echo "=============================================="
	@echo "Building create_block_mask CUDA extension..."
	@echo "=============================================="
	cd create_block_mask && pip install -e . --no-build-isolation
	@echo ""

# Run all tests
test: test_block_mask

tt: test_block_mask # for convenience cjerry
# Run create_block_mask tests (manual script)
test_block_mask:
	@echo "=============================================="
	@echo "Testing create_block_mask..."
	@echo "=============================================="
	cd create_block_mask && python test_create_block_mask.py
	@echo ""


pytest: pytest_block_mask
# Run create_block_mask pytest (full test suite)
pytest_block_mask:
	@echo "=============================================="
	@echo "Running pytest for create_block_mask..."
	@echo "=============================================="
	cd create_block_mask && pytest test_create_block_mask.py -v --tb=short
	@echo ""

# Run all pytest tests
pytest: pytest_block_mask
	@echo ""

# Run all benchmarks
bench: bench_block_mask

# Run create_block_mask benchmark
bench_block_mask:
	@echo "=============================================="
	@echo "Benchmarking create_block_mask..."
	@echo "=============================================="
	cd create_block_mask && python benchmark_create_block_mask.py --seqlen 220000 --pattern diagonal
	@echo ""

# Clean all build artifacts
clean: clean_block_mask

# Clean create_block_mask build artifacts
clean_block_mask:
	@echo "Cleaning create_block_mask build artifacts..."
	rm -rf create_block_mask/build
	rm -rf create_block_mask/dist
	rm -rf create_block_mask/*.egg-info
	rm -rf create_block_mask/*.so
	rm -rf create_block_mask/__pycache__

# Build and test all
build_and_test: all test

# Help
help:
	@echo "Available targets:"
	@echo "  all               - Build all extensions (default)"
	@echo "  block_mask        - Build create_block_mask extension"
	@echo "  test              - Run all tests (manual scripts)"
	@echo "  test_block_mask   - Run create_block_mask tests (manual script)"
	@echo "  pytest            - Run all pytest tests"
	@echo "  pytest_block_mask - Run create_block_mask pytest (full test suite)"
	@echo "  bench             - Run all benchmarks"
	@echo "  bench_block_mask  - Run create_block_mask benchmark"
	@echo "  clean             - Clean all build artifacts"
	@echo "  clean_block_mask  - Clean create_block_mask artifacts"
	@echo "  build_and_test    - Build all and run all tests"
	@echo "  help              - Show this help message"

