#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Detect ROCm build of PyTorch.  When using PyTorch built with ROCm the
# compiler is hipcc and the "nvcc" extra_compile_args key is still honoured
# (PyTorch maps it automatically).  The only difference is the GLM force-cuda
# macro: use GLM_FORCE_HIP on ROCm to avoid nvcc-specific pragmas.
try:
    import torch
    _IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
except Exception:
    _IS_ROCM = False

_glm_flag = "-DGLM_FORCE_HIP" if _IS_ROCM else "-DGLM_FORCE_CUDA"

setup(
    name="diff_gaussian_rasterization_fastgs",
    packages=['diff_gaussian_rasterization_fastgs'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_fastgs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/adam.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": [
                    _glm_flag,
                    "-I" + os.path.join(_HERE, "third_party/glm/"),
                ]
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
