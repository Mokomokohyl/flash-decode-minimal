import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

is_debug = os.getenv('MODE', 'bench').lower() == 'debug'

base_nvcc_flags = ['-O2', '-arch=sm_80']
if is_debug:
    base_nvcc_flags.extend(['-Xptxas', '-v', '-DDEBUG'])

ALL_KERNELS = {
    "minimal": {
        "sources": ['kernels/bind.cpp', 'kernels/flash_attn_minimal.cu'],
        "extra_compile_args": {'cxx': ['-O2'], 'nvcc': base_nvcc_flags}
    },
    "v1": {
        "sources": ['kernels/bind.cpp', 'kernels/flash_attn_v1.cu'],
        "extra_compile_args": {'cxx': ['-O2'], 'nvcc': base_nvcc_flags}
    },
    "v2": {
        "sources": ['kernels/bind.cpp', 'kernels/flash_attn_v2.cu'],
        "extra_compile_args": {'cxx': ['-O2'], 'nvcc': base_nvcc_flags}
    },
    "minimal_v2": {
        "sources": ['kernels/bind.cpp', 'kernels/flash_attn_minimal_v2.cu'],
        "extra_cuda_cflags": base_nvcc_flags
    },
    "fdm": {
        "sources": ['kernels/bind.cpp', 'kernels/flash_decode_minimal.cu'],
        "extra_compile_args": {'cxx': ['-O2'], 'nvcc': base_nvcc_flags}
    },
    "fdm_splitkv": {
        "sources": ['kernels/bind.cpp', 'kernels/flash_decode_minimal_splitKV.cu'],
        "extra_compile_args": {'cxx': ['-O2'], 'nvcc': base_nvcc_flags}
    },
}

def get_ext_modules():
    kernels_to_compile_str = os.getenv('KERNELS', 'all')
    
    if kernels_to_compile_str == 'all':
        kernels_to_compile = list(ALL_KERNELS.keys())
    else:
        kernels_to_compile = [k.strip() for k in kernels_to_compile_str.split(',')]

    ext_modules = []
    for kernel_name in kernels_to_compile:
        if kernel_name in ALL_KERNELS:
            print(f"Preparing to compile kernel: {kernel_name}")
            kernel_info = ALL_KERNELS[kernel_name]
            
            ext_modules.append(
                CUDAExtension(
                    name=f"kernels.{kernel_name}",
                    **kernel_info
                )
            )
        else:
            print(f"Warning: Kernel '{kernel_name}' not found in definitions.")
            
    return ext_modules


ext_modules = get_ext_modules()
if ext_modules:
    setup(
        name='flash_decode_kernels',
        ext_package='llama',
        ext_modules=ext_modules,
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        },
    )