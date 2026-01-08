# Makefile for building CUDA extensions in utils directory
#
# Usage:
#   make              - Build all extensions (create_block_mask and magi_to_hstu)
#   make all          - Same as above
#   make block_mask   - Build only create_block_mask extension
#   make magi_to_hstu - Build only magi_to_hstu extension
#   make test         - Run all tests (manual scripts)
#   make test_block_mask   - Run create_block_mask tests (manual script)
#   make test_magi_to_hstu - Run magi_to_hstu tests
#   make pytest            - Run all pytest tests
#   make pytest_block_mask - Run create_block_mask pytest (full test suite)
#   make clean        - Clean all build artifacts
#   make clean_block_mask   - Clean create_block_mask build artifacts
#   make clean_magi_to_hstu - Clean magi_to_hstu build artifacts

.PHONY: all block_mask magi_to_hstu test test_block_mask test_magi_to_hstu pytest pytest_block_mask clean clean_block_mask clean_magi_to_hstu

# Default target: build all
all: block_mask magi_to_hstu

# Build create_block_mask extension
block_mask:
	@echo "=============================================="
	@echo "Building create_block_mask CUDA extension..."
	@echo "=============================================="
	cd create_block_mask && pip install -e . --no-build-isolation
	@echo ""

# Build magi_to_hstu extension
magi_to_hstu:
	@echo "=============================================="
	@echo "Building magi_to_hstu CUDA extension..."
	@echo "=============================================="
	cd magi_to_hstu && pip install -e . --no-build-isolation
	@echo ""

# Run all tests
test: test_block_mask test_magi_to_hstu

tt: test_block_mask # for convenience cjerry
# Run create_block_mask tests (manual script)
test_block_mask:
	@echo "=============================================="
	@echo "Testing create_block_mask..."
	@echo "=============================================="
	cd create_block_mask && python test_create_block_mask.py
	@echo ""

# Run create_block_mask pytest (full test suite)
pytest_block_mask:
	@echo "=============================================="
	@echo "Running pytest for create_block_mask..."
	@echo "=============================================="
	cd create_block_mask && pytest test_create_block_mask.py -v --tb=short
	@echo ""

# Run magi_to_hstu tests
test_magi_to_hstu:
	@echo "=============================================="
	@echo "Testing magi_to_hstu..."
	@echo "=============================================="
	cd magi_to_hstu && python test_magi_to_hstu.py
	@echo ""

# Run all pytest tests
pytest: pytest_block_mask
	@echo ""

# Clean all build artifacts
clean: clean_block_mask clean_magi_to_hstu

# Clean create_block_mask build artifacts
clean_block_mask:
	@echo "Cleaning create_block_mask build artifacts..."
	rm -rf create_block_mask/build
	rm -rf create_block_mask/dist
	rm -rf create_block_mask/*.egg-info
	rm -rf create_block_mask/*.so
	rm -rf create_block_mask/__pycache__

# Clean magi_to_hstu build artifacts
clean_magi_to_hstu:
	@echo "Cleaning magi_to_hstu build artifacts..."
	rm -rf magi_to_hstu/build
	rm -rf magi_to_hstu/dist
	rm -rf magi_to_hstu/*.egg-info
	rm -rf magi_to_hstu/*.so
	rm -rf magi_to_hstu/__pycache__

# Build and test all
build_and_test: all test

# Help
help:
	@echo "Available targets:"
	@echo "  all               - Build all extensions (default)"
	@echo "  block_mask        - Build create_block_mask extension"
	@echo "  magi_to_hstu      - Build magi_to_hstu extension"
	@echo "  test              - Run all tests (manual scripts)"
	@echo "  test_block_mask   - Run create_block_mask tests (manual script)"
	@echo "  test_magi_to_hstu - Run magi_to_hstu tests"
	@echo "  pytest            - Run all pytest tests"
	@echo "  pytest_block_mask - Run create_block_mask pytest (full test suite)"
	@echo "  clean             - Clean all build artifacts"
	@echo "  clean_block_mask  - Clean create_block_mask artifacts"
	@echo "  clean_magi_to_hstu- Clean magi_to_hstu artifacts"
	@echo "  build_and_test    - Build all and run all tests"
	@echo "  help              - Show this help message"

