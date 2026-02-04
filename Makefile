.PHONY: install tt vt bm fm clean_dist create_dist upload_package

# Set ARBITRARY=1 to use test_arbitrary_mask.py, otherwise use test_flash_attn.py
# Example: make fm ARBITRARY=1
ARBITRARY ?= 1
ifeq ($(ARBITRARY),1)
    TEST_FILE = tests/cute/test_arbitrary_mask.py
else
    TEST_FILE = tests/cute/test_flash_attn.py
endif

# Flex attention reference: 0=disable (use CUDA kernel only), 1=enable (use PyTorch flex_attention)
# Set to 0 if your PyTorch version doesn't support flex_attention
FLEXATTN ?= 1
ifeq ($(FLEXATTN),1)
    export DISABLE_FLEX_ATTENTION := FALSE
else
    export DISABLE_FLEX_ATTENTION := TRUE
endif

install:
	bash install.sh editable

tt:
	PYTHONPATH=${PWD} python $(TEST_FILE)

vt:
	PYTHONPATH=${PWD} pytest $(TEST_FILE)::test_arbitrary_mask -v

# profile fwd and bwd
bm:
	PYTHONPATH=${PWD} ncu --set full --nvtx --nvtx-include "flash_attn_bwd_kernel/" --nvtx-include "flash_attn_fwd_kernel/"  -f -o flash_bwd.%p  python $(TEST_FILE)

# profile fwd
fm:
	PYTHONPATH=${PWD} ncu --set full --nvtx --nvtx-include "flash_attn_fwd_kernel/"  -f -o flash_fwd.%p  python $(TEST_FILE)

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*
