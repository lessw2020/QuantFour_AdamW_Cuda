
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name=f"quantfour",
    description="4-bit AdamW",
    keywords="optimizers ",
    version="1.0.02052024",
    url="https://github.com/lessw2020/4Bit_AdamW_Triton",
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        CUDAExtension(
            'quantfour_cuda',
            ['cuda_fused_interface.cc', 'cuda_adamw_kernels.cu']
        ),



    ],
)
