cimport cython
import time
import sys

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat(int, int, int, void *, int) nogil
        Mat() nogil

cdef extern from "opencv2/core/core.hpp":
    int CV_32F
    int CV_8U

cdef extern from "adapthisteq.cpp":
   void adapthisteq(Mat &_in, Mat &_out, float maxderiv) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _clahe(int rows, int cols,
                 unsigned char [:, :] imin, unsigned char [:, :] imout,
                 float maxderiv) nogil:
    cdef Mat Min, Mout

    Min = Mat(rows, cols, CV_8U,
              &(imin[0, 0]), imin.strides[0])
    Mout = Mat(rows, cols, CV_8U,
              &(imout[0, 0]), imout.strides[0])

    adapthisteq(Min, Mout, maxderiv)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef clahe(unsigned char [:, :] imin, unsigned char [:, :] imout,
            float maxderiv):
    assert imin.shape[0] == imout.shape[0]
    assert imin.shape[1] == imout.shape[1]
    with nogil:
        _clahe(imin.shape[0], imin.shape[1],
               imin, imout, maxderiv)
