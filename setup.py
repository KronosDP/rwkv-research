from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_wkv_kernel',
    ext_modules=[
        CUDAExtension(
            name='custom_wkv_kernel', # Must match the PYBIND11_MODULE name and import name
            sources=['wkv_kernel.cu'],
            extra_compile_args={'cxx': ['-g', '-O2'], # Example flags
                                'nvcc': ['-O2']}    # Example flags
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })