.PHONY: install tt vt bm fm clean_dist create_dist upload_package

install:
	bash install.sh

tt:
	PYTHONPATH=${PWD} python tests/cute/test_flash_attn.py

vt:
	PYTHONPATH=${PWD} pytest tests/cute/test_flash_attn.py::test_flash_attn_output -v

# profile fwd and bwd
bm:
	PYTHONPATH=${PWD} ncu --set full --nvtx --nvtx-include "flash_attn_bwd_kernel/" --nvtx-include "flash_attn_fwd_kernel/"  -f -o flash_bwd.%p  python tests/cute/test_flash_attn.py

# profile fwd
fm:
	PYTHONPATH=${PWD} ncu --set full --nvtx --nvtx-include "flash_attn_fwd_kernel/"  -f -o flash_fwd.%p  python tests/cute/test_flash_attn.py

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*
