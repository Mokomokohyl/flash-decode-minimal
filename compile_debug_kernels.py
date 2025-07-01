import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='minimal_attn',
    ext_modules=[
        CUDAExtension(
            name='minimal_attn',
            sources=['kernels/bind.cpp', 'kernels/flash_minimal.cu'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2', '-arch=sm_80', '-DDEBUG']
            }
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