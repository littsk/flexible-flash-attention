
tt:
	PYTHONPATH=$(PWD) python tests/cute/test_arbitrary_mask.py
fm:
	PYTHONPATH=$(PWD) ncu --set full --clock-control none -f -o baseline --kernel-id ::regex:kernel_cutlass_kernel: python tests/cute/test_arbitrary_mask.py

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*
