from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#sudo apt install libeigen3-dev
eigen_include_dir = '/usr/include/eigen3' 

setup(
    name='lda_module',
    ext_modules=[
        CppExtension(
            name='lda_module',
            sources=['lda.cpp'],
            include_dirs=[eigen_include_dir],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
