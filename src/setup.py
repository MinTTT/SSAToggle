from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = [Extension("toggle", ["toggle.pyx", "SSAToggle.cpp"], language='c++',
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=["/openmp"]
                         )]

setup(cmdclass={'build_ext': build_ext}, ext_modules=cythonize(ext_modules))
