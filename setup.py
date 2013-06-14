from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
 
setup(ext_modules=[Extension("fastremap",
                             ["fastremap.pyx"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             language="c",),
                   Extension("clahe",
                             ["clahe.pyx"],
                             language="c++",)],
      cmdclass = {'build_ext': build_ext})
