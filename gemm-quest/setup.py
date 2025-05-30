import os

import setuptools
from torch.utils import cpp_extension


CUTLASS_PATH = os.path.join('.', 'cutlass')

setuptools.setup(
    name='quest',
    version='0.0.1',
    description='QuEST: Accurate Quantized Gradient Estimation for Training Low-Bitwith Large Language Models',
    install_requires=['torch'],
    packages=setuptools.find_packages(exclude=['docs', 'examples', 'tests']),
    ext_modules=[cpp_extension.CUDAExtension(
        'quest_c',
        [
            'quest/pybind.cpp',
            'quest/matmul.cpp',
            'quest/matmul_kernel.cu',
            'quest/quantize.cpp',
            'quest/quantize_kernel.cu',
            'quest/fused_matmul_dequantize.cpp',
            'quest/fused_matmul_dequantize_kernel.cu',
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                # '-march=native',
                # '-mtune=native',
                # '-funroll-loops',
                # '-flto',
                # '-fprefetch-loop-arrays',
                # '-falign-functions=16',
                # '-pthread',
                # '-ffast-math',
                # '-ftz=true',
                # '-prec-div=false',
                # '-prec-sqrt=false',
            ],
            'nvcc': [
                '-O3',
                # '-gencode', 'arch=compute_86,code=sm_86',  # Add GPU architecture 8.6 if desired
                # '-Xptxas', '--opt-level=3',
                # '--disable-warnings',
                # '-restrict',
                # '--expt-relaxed-constexpr',
                # '--expt-extended-lambda',
                # '--use_fast_math',
            ]
        }
    )],
    include_dirs=[
        os.path.join(CUTLASS_PATH, 'include'),
        os.path.join(CUTLASS_PATH, 'tools', 'util', 'include'),
        os.path.join('.', 'include'),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
