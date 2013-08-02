cimport cython
from cython.parallel cimport prange
import time

ctypedef fused pixeltype:
    cython.uchar
    cython.float
    cython.double


@cython.cdivision(True)
cdef float lerp(float a, float b, float t) nogil:
    return (b - a) * t + a

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef pixeltype lerp2d(pixeltype [:, :] input,
                      float i, float j) nogil:
    cdef int inti, intj
    cdef float deltai, deltaj

    inti = <int> i
    intj = <int> j
    if inti == input.shape[0] - 1:
        inti -= 1
    if intj == input.shape[1] - 1:
        intj -= 1
    deltai = i - inti
    deltaj = j - intj

    return <pixeltype> lerp(lerp(<double> input[inti, intj], <double>input[inti + 1, intj], deltai),
                            lerp(<double>input[inti, intj + 1], <double> input[inti + 1, intj + 1], deltai),
                            deltaj)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _remap(pixeltype [:, :] input,
                 pixeltype [:, :] output,
                 double [:, :] R, double [:, :] T,
                 bint repeat) nogil:
   cdef int i, j
   cdef float interped_row, interped_col

   for i in prange(output.shape[0], schedule='static', num_threads=8):
       for j in range(output.shape[1]):
           # apply rigid transformation
           interped_row = i * R[0, 0] + j * R[0, 1] + T[0, 0]
           interped_col = i * R[1, 0] + j * R[1, 1] + T[1, 0]

           # deal with clamping and repeat
           if interped_row < 0:
               if not repeat:
                   continue
               interped_row = 0
           elif interped_row > input.shape[0] - 1:
               if not repeat:
                   continue
               interped_row = input.shape[0] - 1
           if interped_col < 0:
               if not repeat:
                   continue
               interped_col = 0
           elif interped_col > input.shape[1] - 1:
               if not repeat:
                   continue
               interped_col = input.shape[1] - 1

           output[i, j] = lerp2d(input, interped_row, interped_col)

cpdef remap(pixeltype [:, :] input,
            pixeltype [:, :] output,
            double [:, :] R, double [:, :] T,
            bint repeat):
    _remap(input, output, R, T, repeat)
