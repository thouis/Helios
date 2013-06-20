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
                             include_dirs=['/n/lichtmanfs1/thouis_jones/opencv-install/include'],
                             libraries=['opencv_core', 'opencv_imgproc'],
                             library_dirs=['/n/lichtmanfs1/thouis_jones/opencv-install/lib'],
                             extra_link_args=['-Wl,-rpath,/n/lichtmanfs1/thouis_jones/opencv-install/lib'],
                             language="c++",)],
      cmdclass = {'build_ext': build_ext})
