from helios import Flow
import cv2
import numpy as np
import pylab
import os
import sys

flocache = {}

def load_flow(a, b):
    if (a, b) in flocache:
        return flocache[a, b]
    if os.path.exists("cache_%02d_%02d.hdf5" % (a, b)):
        flocache[a, b] = Flow.load("cache_%02d_%02d.hdf5" % (a, b))
        return flocache[a, b]
    try:
        flocache[a, b] = Flow.load("%02d_%02d.hdf5" % (a, b))
        return flocache[a, b]
    except:
        print "could not load", a, b
        return None

def generate_chains(a, b, maxstep, dir):
    if a != b:
        yield (a, b)
        for s in range(1, maxstep + 1):
            if dir * (b - a) > s:
                for c in generate_chains(b - s * dir, b, maxstep, dir):
                    yield (a,) + c

def image_diff(flo, a, b):
    flo = flo.resize(a.shape)
    return np.sum(np.abs(a - flo.warp(b)))

def best_flow(a, b, dir=1):
    ''' best a to b'''
    ima = cv2.imread(sys.argv[a], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    imb = cv2.imread(sys.argv[b], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    def best_of(*choices):
        def load_chain(c):
            if (((dir == 1) and (c[1] <= c[0])) or
                ((dir == -1) and (c[1] >= c[0]))):
                return None
            flow = load_flow(c[0], c[1])
            c.pop(0)
            while len(c) > 1:
                flow = flow.chain(load_flow(c[0], c[1]))
                c.pop(0)
            return flow

        flocache.clear()
        flows = [load_chain(list(c)) for c in choices]
        fdc = [(image_diff(f, ima, imb), f, c) for (f, c) in zip(flows, choices) if f is not None]
        fdc.sort()
        best = fdc[0]
        return best[1], best[2]

    return best_of(*generate_chains(a, b, 5, dir))

def cache(a, b, fl):
    fl.save("cache_%02d_%02d.hdf5" % (a, b))

if __name__ == "__main__":
    keypoints = [1, 19, 41, 60, 78, 100, 119, 140, 162, 183, 204, 225, 244, 266, 282]
    keypoints = [100, 119, 140, 162, 183, 204, 225, 244, 266, 282]
    for k1, k2 in zip(keypoints, keypoints[1:]):
        for idx in range(k1 + 1, k2 + 1):
            b, chain = best_flow(k1, idx)
            print "FLOW", k1, idx, b.distortion(), chain
            cache(k1, idx, b)
        for idx in range(k2 - 1, k1, -1):
            b, chain = best_flow(k2, idx, dir=-1)
            print "FLOW", k2, idx, b.distortion(), chain
            cache(k2, idx, b)

