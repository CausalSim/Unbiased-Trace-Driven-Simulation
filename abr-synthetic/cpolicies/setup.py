from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from setuptools.command.build_ext import build_ext


# Avoid a gcc warning below:
# cc1plus: warning: command line option ‘-Wstrict-prototypes’ is valid
# for C/ObjC but not for C++
class BuildExtDebug(build_ext):
    def build_extensions(self):
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()


extension = cythonize(
    [
        Extension('mpc',
                  ['mpc.pyx'],
                  language="c++",
                  extra_compile_args=["-std=c++14"],
                  include_dirs=[numpy.get_include()]
                  )
    ],
    compiler_directives={'language_level': "3"}
)

setup(
    # Information
    name="cpol",
    ext_modules=extension,
    cmdclass={'build_ext': BuildExtDebug}
)
