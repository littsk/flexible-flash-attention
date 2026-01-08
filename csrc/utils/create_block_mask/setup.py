# Setup script for create_block_mask CUDA extension
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="create_block_mask_cuda",
    version="0.1",
    description="CUDA kernels for creating block masks (FlexAttention style)",
    ext_modules=[
        CUDAExtension(
            name="create_block_mask_cuda",
            sources=[
                "create_block_mask_api.cpp",
                "create_block_mask.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_80,code=sm_80",
                    "-gencode", "arch=compute_90,code=sm_90",
                    "-gencode", "arch=compute_100,code=sm_100",
                ],
            },
            include_dirs=[this_dir],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

