import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if os.getenv('USE_FLASH_MINIMAL', 'false').lower() == 'true':
    setup(
        name='minimal_attn',
        ext_modules=[
            CUDAExtension(
                name='minimal_attn',
                sources=['kernels/bind.cpp', 'kernels/flash_minimal.cu'],
                extra_cuda_cflags=['-O2', '--arch=sm_80']
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        },
        options={
            'build_ext': {
                'build_lib': os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama")
            }
        }
    )
if os.getenv('USE_FLASH_V1', 'false').lower() == 'true':
    setup(
        name='minimal_attn',
        ext_modules=[
            CUDAExtension(
                name='minimal_attn',
                sources=['kernels/bind.cpp', 'kernels/flash_attn_v1.cu'],
                extra_cuda_cflags=['-O2', '--arch=sm_80']
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        },
        options={
            'build_ext': {
                'build_lib': os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama")
            }
        }
    )