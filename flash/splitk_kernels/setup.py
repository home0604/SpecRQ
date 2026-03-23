import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
# System nvcc (11.8)와 PyTorch CUDA (12.8) version mismatch 우회
os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] = "1"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Monkey-patch: CUDA version check 비활성화
import torch.utils.cpp_extension as _ext
_ext._check_cuda_version = lambda *a, **kw: None

setup(
    name="splitk_kernels",
    version="0.1.0",
    description="Split-K Flash Decoding CUDA kernels",
    ext_modules=[
        CUDAExtension(
            name="_C",
            sources=["csrc/kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math",
                         "-gencode=arch=compute_80,code=sm_80"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=[],  # _C extension + __init__.py가 같은 디렉토리에 있으므로 패키지 불필요
)
