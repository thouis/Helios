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

    return <pixeltype> lerp(lerp(input[inti, intj], input[inti + 1, intj], deltai),
                            lerp(input[inti, intj + 1], input[inti + 1, intj + 1], deltai),
                            deltaj)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _remap(pixeltype [:, :] input,
                 pixeltype [:, :] output,
                 double [:, :] R, double [:, :] T,
                 float [:, :] row_warp, float [:, :] column_warp,
                 bint repeat) nogil:
   cdef int i, j
   cdef float normalized_i, normalized_j, warp_i, warp_j, \
       adjust_i, adjust_j, interped_row, interped_col

   for i in prange(output.shape[0], schedule='static', num_threads=8):
       normalized_i = (<float> i) / (output.shape[0] - 1)
       warp_i = normalized_i * (row_warp.shape[0] - 1)
       for j in range(output.shape[1]):
           normalized_j = (<float> j) / (output.shape[1] - 1)
           warp_j = normalized_j * (row_warp.shape[1] - 1)

           # interpolate warp coords
           adjust_i = lerp2d(row_warp, warp_i, warp_j)
           adjust_j = lerp2d(column_warp, warp_i, warp_j)
           # apply rigid transformation
           interped_row = adjust_i + normalized_i * R[0, 0] + normalized_j * R[0, 1] + T[0, 0]
           interped_col = adjust_j + normalized_i * R[1, 0] + normalized_j * R[1, 1] + T[1, 0]

           # deal with clamping and repeat
           interped_row = interped_row * (input.shape[0] - 1)
           interped_col = interped_col * (input.shape[1] - 1)
           if interped_row < 0:
               if not repeat:
                   continue
               interped_row = 0
           if interped_row > input.shape[0] - 1:
               if not repeat:
                   continue
               interped_row = input.shape[0] - 1
           if interped_col < 0:
               if not repeat:
                   continue
               interped_col = 0
           if interped_col > input.shape[1] - 1:
               if not repeat:
                   continue
               interped_col = input.shape[1] - 1

           output[i, j] = lerp2d(input, interped_row, interped_col)

cpdef remap(pixeltype [:, :] input,
            pixeltype [:, :] output,
            double [:, :] R, double [:, :] T,
            float [:, :] row_warp, float [:, :] column_warp,
            bint repeat):
    _remap(input, output, R, T, row_warp, column_warp, repeat)
