import sys

import numpy as np
import cv2
import fastremap
import clahe
import scipy.sparse as sparse
import scipy.sparse.linalg as spspl
import pylab

from cg import cg

s_hstack = sparse.hstack
s_vstack = sparse.vstack

class Flow(object):
    def __init__(self, shape, u=None, v=None):
        if u is None:
            self.u = np.zeros(shape, np.float32)
        else:
            self.u = u

        if v is None:
            self.v = np.zeros(shape, np.float32)
        else:
            self.v = v

    def resize(self, newshape):
        scaleu = float(newshape[1]) / self.u.shape[1]
        scalev = float(newshape[0]) / self.v.shape[0]
        return Flow(newshape,
                    scaleu * cv2.resize(self.u, newshape[::-1]),
                    scalev * cv2.resize(self.v, newshape[::-1]))

def warp(im, flow):
    ybase, xbase = np.mgrid[:im.shape[0], :im.shape[1]]
    return cv2.remap(im,
                     (xbase + flow.u).astype(np.float32),
                     (ybase + flow.v).astype(np.float32),
                     cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=np.nan)

def scalespace(im, octaves):
    sp = {}
    for o in range(octaves + 1):
        sp[o] = im
        if o < octaves:
            im = cv2.GaussianBlur(im, (0, 0), sigmaX=1.0)
            im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    return sp

# combined central difference and forward difference
filter = (np.array([[0, 1, -8, 0,       8, -1,     0]]) / 12.0 +
          np.array([[0, 0,  0, -11/6.0, 3, -3/2.0, 1/3.0]])) / 2.0

def dx(im):
    return cv2.filter2D(im, cv2.CV_32F, filter, borderType=cv2.BORDER_REPLICATE)

def dy(im):
    return cv2.filter2D(im, cv2.CV_32F, filter.T, borderType=cv2.BORDER_REPLICATE)

def derivs(im):
    return dx(im), dy(im)

def deriv_operators(shape):
    scoo = sparse.coo_matrix
    npixels = np.prod(shape)
    linearcoords = np.arange(npixels, dtype=np.int32)
    def linearize(ic, jc):
        return (ic * shape[1] + jc).ravel()
    i, j = np.ogrid[:shape[0], :shape[1]]
    Dx = 0
    Dy = 0
    for w, offset in zip(filter.flat, range(-3, 4)):
        if w == 0:
            continue
        weights = w * np.ones(npixels) / 12.0
        dx = np.clip(j + offset, 0, shape[1] - 1)
        Dx = Dx + scoo((weights, (linearcoords, (linearize(i, dx)))), shape=(npixels, npixels))
        dy = np.clip(i + offset, 0, shape[0] - 1)
        Dy = Dy + scoo((weights, (linearcoords, (linearize(dy, j)))), shape=(npixels, npixels))
    return Dx.tocsr(), Dy.tocsr()

def vectorize(im):
    return im.ravel().reshape((-1, 1))

def diag(im):
    npixels = np.prod(im.shape)
    linearcoords = np.arange(npixels, dtype=np.int32)
    return sparse.coo_matrix((im.ravel(), (linearcoords, linearcoords)))

def imshow(t, im):
    im = im.astype(np.float32)
    im -= im.min()
    im /= im.max()
    cv2.imshow(t, im)

# Penalty functions

def phi_prime(x, epsilon=0.001):
    '''spatial: phi(x) = sqrt(x + epsilon**2)'''
    return np.sqrt(x) / np.sqrt(x + epsilon * epsilon)

'''data penalty same as spatial penalty'''
psi_prime = phi_prime

def compute_flow(im1, im2, previous_flow=None,
                 average_derivs=True,
                 flow_iters=10,
                 alpha=10.0):
    # See Ce Liu's thesis, appendix A for notation

    assert im1.shape == im2.shape

    # compute image derivatives
    Ix, Iy = derivs(im2)

    # warp im2 and derivs by existing flow
    if previous_flow is not None:
        cur_flow = previous_flow.resize(im1.shape)
        I2 = warp(im2, cur_flow)
        Ix = warp(Ix, cur_flow)
        Iy = warp(Iy, cur_flow)
    else:
        cur_flow = Flow(im1.shape)
        I2 = im2

    # Average derivatives
    if average_derivs:
        temp_Ix, temp_Iy = derivs(im1)
        Ix = (Ix + temp_Ix) / 2.0
        Iy = (Iy + temp_Iy) / 2.0

    # temporal derivative
    Iz = I2 - im1

    # mask nonoverlapping areas
    Iz[np.isnan(Iz)] = 0
    Ix[np.isnan(Ix)] = 0
    Iy[np.isnan(Iy)] = 0

    # setup
    Dx, Dy = deriv_operators(im1.shape)
    Ix = vectorize(Ix)
    Iy = vectorize(Iy)
    Iz = vectorize(Iz)
    U = vectorize(cur_flow.u)
    V = vectorize(cur_flow.v)
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    prev_x = None
    for i in range(flow_iters):
        # Compute data and spatial weighting terms
        g = (Dx * (U + dU)) ** 2 + (Dy * (U + dU)) ** 2 + (Dx * (V + dV)) ** 2 + (Dy * (V + dV)) ** 2
        f = (Iz + Ix * dU + Iy * dV) ** 2

        Phi = phi_prime(g)
        Psi = psi_prime(f)

        L = Dx.T * diag(Phi) * Dx + Dy.T * diag(Phi) * Dy

        UL = diag(Psi * (Ix ** 2)) + alpha * L
        UR = LL = diag(Psi * Ix * Iy)
        LR = diag(Psi * (Iy ** 2)) + alpha * L
        A = s_vstack((s_hstack((UL, UR)),
                      s_hstack((LL, LR)))).tocsc()
        preA = sparse.diags(1.0 / A[range(A.shape[0]), range(A.shape[0])].A.ravel(),
                            0)
        bU = Psi * Ix * Iz + alpha * L * U
        bL = Psi * Iy * Iz + alpha * L * V
        b = - np.vstack((bU, bL))

        x, st = cg(A, b, x0=prev_x, M=preA, tol=0.05 / np.linalg.norm(b))
        print i, np.median(np.abs(x)), st
        dU = x[:dU.shape[0]].reshape(dU.shape)
        dV = x[dU.shape[0]:].reshape(dU.shape)
        prev_x = x
        if st <= 3 and i > 1:
            break
    cur_flow.u += dU.reshape(cur_flow.u.shape)
    cur_flow.v += dV.reshape(cur_flow.v.shape)
    cur_flow.u = cv2.medianBlur(cur_flow.u, 5)
    cur_flow.v = cv2.medianBlur(cur_flow.v, 5)
    return cur_flow

if __name__ == "__main__":
    im1 = cv2.imread(sys.argv[1], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im2 = cv2.imread(sys.argv[2], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im1 = im1[3249:47465, 5099:34750]
    im2 = im2[3249:47465, 5099:34750]

    im1 = cv2.resize(im1, (im.shape[1] // 16, im.shape[0] // 16))
    im2 = cv2.resize(im2, (im.shape[1] // 16, im.shape[0] // 16))

    # reduce noise
    for i in range(2):
        im1 = cv2.medianBlur(im1, 3)
        im2 = cv2.medianBlur(im2, 3)

    # equalize histogram
    clahe.clahe(im1, im1, 1.5)
    clahe.clahe(im2, im2, 1.5)
    im1 = im1.astype(np.float32) / 255
    im2 = im2.astype(np.float32) / 255

    # keep about 32 pixels on the shortest side
    octaves = max(0, int(np.log2(min(*im1.shape)) - 5))
    print "Downsampling to", octaves, "x".join(str(s * (0.5 ** octaves)) for s in im1.shape)
    pyramid1 = scalespace(im1, octaves)
    pyramid2 = scalespace(im2, octaves)

    flow = compute_flow(pyramid1[octaves], pyramid2[octaves])
    for o in range(octaves):
        print "OCTAVE", octaves - o - 1, octaves
        flow = compute_flow(pyramid1[octaves - o - 1],
                            pyramid2[octaves - o - 1],
                            previous_flow=flow)


notes = '''

E = rho(I1 - warp(I2)) + rho(u_ij - u_i1j) + rho(u_ij - u_ij1) + rho(v_ij - v_i1j) + rho(vij - v_ij1)
linearize to approximate I1 - warp(I2) with I1 - warp(I2) + u * dI1/dx + v * dI2/dy (??? + u * v * ddI/dxdy)

Equation 10 of Papenberg et al. 2006
http://www.mia.uni-saarland.de/Publications/papenberg-ijcv06.pdf

http://hci.iwr.uni-heidelberg.de/Staff/bgoldlue/fvia_ws_2011/fvia_ws_2011_02_gradient_descent.pdf
def gradient(im):
    dx = - im
    dx[:-1, :] += im[1:, :]
    dx[-1, :] = 0

dPsiData * (dIdx**2 * newu + dIdx * dIdy * newv + dIdt * dIdx) - alpha * div(dPsiSmooth * grad(u0 + newu)) = 0
                                                               - alpha * div(dPsiSmooth * 

See equations 6-8 on page 11 of Deqing's thesis.
http://cs.brown.edu/~dqsun/pubs/Deqing_Sun_dissertation.pdf

Ce's thesis, particularly Appendix A
http://people.csail.mit.edu/celiu/Thesis/CePhDThesis.pdf
'''
