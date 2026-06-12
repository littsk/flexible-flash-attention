# Makefile for building CUDA extensions in utils directory
#
# Usage:
#   make              - Build all extensions (create_block_mask, magi_to_hstu)
#   make all          - Same as above
#   make block_mask   - Build only create_block_mask extension
#   make create_block_mask - Alias for block_mask
#   make magi_to_hstu - Build only magi_to_hstu extension
#   make test         - Run utility extension tests (manual scripts)
#   make pytest       - Run utility extension pytest tests
#   make clean        - Clean utility extension build artifacts

.PHONY: all install block_mask create_block_mask magi_to_hstu test test_block_mask test_create_block_mask test_magi_to_hstu pytest pytest_block_mask pytest_magi_to_hstu bench bench_block_mask clean clean_block_mask clean_magi_to_hstu

# Default target: build all
all: block_mask magi_to_hstu

install: all

# Build create_block_mask extension
block_mask:
	@echo "=============================================="
	@echo "Building create_block_mask CUDA extension..."
	@echo "=============================================="
	cd create_block_mask && pip install --no-user -e . --no-build-isolation
	@echo ""

create_block_mask: block_mask

magi_to_hstu:
	@echo "=============================================="
	@echo "Building magi_to_hstu CUDA extension..."
	@echo "=============================================="
	cd magi_to_hstu && pip install --no-user -e . --no-build-isolation
	@echo ""

# Run all tests
test: test_block_mask test_magi_to_hstu

# Run create_block_mask tests (manual script)
test_block_mask: block_mask
	@echo "=============================================="
	@echo "Testing create_block_mask..."
	@echo "=============================================="
	cd create_block_mask && python test_create_block_mask.py
	@echo ""

test_create_block_mask: pytest_block_mask

test_magi_to_hstu: magi_to_hstu
	@echo "=============================================="
	@echo "Testing magi_to_hstu..."
	@echo "=============================================="
	cd magi_to_hstu && python test_magi_to_hstu.py
	@echo ""

pytest: pytest_block_mask pytest_magi_to_hstu
# Run create_block_mask pytest (full test suite)
pytest_block_mask: block_mask
	@echo "=============================================="
	@echo "Running pytest for create_block_mask..."
	@echo "=============================================="
	cd create_block_mask && pytest test_create_block_mask.py -v --tb=short
	@echo ""

pytest_magi_to_hstu: magi_to_hstu
	@echo "=============================================="
	@echo "Running pytest for magi_to_hstu..."
	@echo "=============================================="
	cd magi_to_hstu && pytest test_magi_to_hstu.py -v --tb=short
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
clean: clean_block_mask clean_magi_to_hstu

# Clean create_block_mask build artifacts
clean_block_mask:
	@echo "Cleaning create_block_mask build artifacts..."
	rm -rf create_block_mask/build
	rm -rf create_block_mask/dist
	rm -rf create_block_mask/*.egg-info
	rm -rf create_block_mask/*.so
	rm -rf create_block_mask/__pycache__
	rm -rf create_block_mask/.pytest_cache

clean_magi_to_hstu:
	@echo "Cleaning magi_to_hstu build artifacts..."
	rm -rf magi_to_hstu/build
	rm -rf magi_to_hstu/dist
	rm -rf magi_to_hstu/*.egg-info
	rm -rf magi_to_hstu/*.so
	rm -rf magi_to_hstu/__pycache__
	rm -rf magi_to_hstu/.pytest_cache

# Build and test all
build_and_test: all test

# Help
help:
	@echo "Available targets:"
	@echo "  all               - Build all extensions (default)"
	@echo "  install           - Build/install all extensions"
	@echo "  block_mask        - Build create_block_mask extension"
	@echo "  create_block_mask - Alias for block_mask"
	@echo "  magi_to_hstu      - Build magi_to_hstu extension"
	@echo "  test              - Run all tests (manual scripts)"
	@echo "  test_block_mask   - Run create_block_mask tests (manual script)"
	@echo "  test_magi_to_hstu - Run magi_to_hstu tests (manual script)"
	@echo "  test_create_block_mask - Run create_block_mask pytest"
	@echo "  pytest            - Run all pytest tests"
	@echo "  pytest_block_mask - Run create_block_mask pytest (full test suite)"
	@echo "  pytest_magi_to_hstu - Run magi_to_hstu pytest"
	@echo "  bench             - Run all benchmarks"
	@echo "  bench_block_mask  - Run create_block_mask benchmark"
	@echo "  clean             - Clean all build artifacts"
	@echo "  clean_block_mask  - Clean create_block_mask artifacts"
	@echo "  clean_magi_to_hstu - Clean magi_to_hstu artifacts"
	@echo "  build_and_test    - Build all and run all tests"
	@echo "  help              - Show this help message"
