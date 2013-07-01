import sys
import cv2
from helios import Flow
import numpy as np

def lerp(a, b, t):
    return (b - a) * t + a

keypoints = [1, 19, 41, 60, 78, 100, 119, 140, 162, 183, 204, 225, 244, 266, 282]
correct_u = 0
correct_v = 0
for k1, k2 in zip(keypoints, keypoints[1:]):
    flow12 = Flow.load('cache_%02d_%02d.hdf5' % (k1, k2))
    old_cor_u = correct_u
    old_cor_v = correct_v
    correct_u += np.median(flow12.u)
    correct_v += np.median(flow12.v)
    for idx in range(k1, k2):
        im = cv2.imread(sys.argv[idx], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if idx > k1:
            fromk1 = Flow.load('cache_%02d_%02d.hdf5' % (k1, idx))
            fromk2 = Flow.load('cache_%02d_%02d.hdf5' % (k2, idx))
            t = (idx - k1) / float(k2 - k1)
            av = fromk1.average(fromk2, t)
            av.u += lerp(old_cor_u, correct_u, t)
            av.v += lerp(old_cor_v, correct_v, t)
        else:
            av = Flow(flow12.u.shape)
            av.u += old_cor_u
            av.v += old_cor_u
        av.resize(im.shape)
        im = av.warp(im)
        c = (av.u.shape[0] // 2, av.u.shape[1] // 2)
        print idx, "CENTER", c, av.u.shape, av.u[c], av.v[c]
        cv2.imwrite("out_%03d.tif" % idx, im)

# write final frame
av = Flow(flow12.u.shape)
av.u += correct_u
av.v += correct_v
im = cv2.imread(sys.argv[idx], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
av.resize(im.shape)
im = av.warp(im)
cv2.imwrite("out_%03d.tif" % idx, im)
c = (av.u.shape[0] // 2, av.u.shape[1] // 2)
print idx + 1, "CENTER", c, av.u.shape, av.u[c], av.v[c]
