import sys

import numpy as np
import cv2
import skimage.morphology as skm
import fastremap
import clahe
import scipy.sparse as sparse
import scipy.sparse.linalg as spspl
from scipy.interpolate import griddata
import h5py

from cg import cg

def pyshowim(im, name):
    pylab.figure()
    pylab.imshow(im)
    pylab.title(name)
    pylab.colorbar()

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

    def save(self, filename):
        f = h5py.File(filename, "w")
        f.create_dataset("u", self.u.shape, data=self.u, compression='gzip')
        f.create_dataset("v", self.v.shape, data=self.v, compression='gzip')
        f.close()

    @classmethod
    def load(cls, filename):
        f = h5py.File(filename, "r")
        fl = cls(f["u"].shape, f["u"][...], f["v"][...])
        f.close()
        return fl

    def warp(self, im, repeat=False):
        ybase, xbase = np.mgrid[:im.shape[0], :im.shape[1]]
        if repeat:
            return cv2.remap(im,
                             (xbase + self.u).astype(np.float32),
                             (ybase + self.v).astype(np.float32),
                             cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        return cv2.remap(im,
                         (xbase + self.u).astype(np.float32),
                         (ybase + self.v).astype(np.float32),
                         cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=np.nan)

    def chain(self, other):
        return Flow(self.u.shape,
                    self.u + self.warp(other.u, repeat=True),
                    self.v + self.warp(other.v, repeat=True))

    def average(self, other, otherweight):
        if otherweight == 0.0:
            return self
        if otherweight == 1.0:
            return other
        return Flow(self.u.shape,
                    (other.u - self.u) * otherweight + self.u,
                    (other.v - self.v) * otherweight + self.v)

    def distortion(self):
        return np.sum(np.abs(np.diff(self.u, axis=0))) + \
            np.sum(np.abs(np.diff(self.u, axis=1))) + \
            np.sum(np.abs(np.diff(self.v, axis=0))) + \
            np.sum(np.abs(np.diff(self.v, axis=1)))

def warp(im, flow):
    return flow.warp(im)

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
                 alpha=1.0,
                 constraints=None):
    # See Ce Liu's thesis, appendix A for notation

    assert im1.shape == im2.shape, "mismatch" + str(im1.shape)  + " " + str(im2.shape)

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
    print "Median Abs error", np.median(np.abs(Iz[np.isfinite(Iz)]))

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

        bU = Psi * Ix * Iz + alpha * L * U
        bL = Psi * Iy * Iz + alpha * L * V
        b = - np.vstack((bU, bL))

        # preconditioner
        di = A[range(A.shape[0]), range(A.shape[0])].A.ravel()
        di[di == 0] = 1.0
        preA = sparse.diags(1.0 / di, 0)

        x, st = spspl.gmres(A, b, x0=prev_x, M=preA, tol=0.05, maxiter=10)
        print i, "med change", np.median(np.abs(x)), "med change", np.median(np.abs(x - prev_x)) if prev_x is not None else ""
        if prev_x is not None:
            if np.max(np.abs(prev_x - x)) < 0.05:
                break
        dU = x[:dU.shape[0]].reshape(dU.shape)
        dV = x[dU.shape[0]:].reshape(dU.shape)
        prev_x = x
    cur_flow.u += dU.reshape(cur_flow.u.shape)
    cur_flow.v += dV.reshape(cur_flow.v.shape)

    if constraints is not None:
        # correct for constraints
        correction_u = np.zeros_like(cur_flow.u)
        correction_v = np.zeros_like(cur_flow.v)
        gi, gj = np.mgrid[0:cur_flow.u.shape[0], 0:cur_flow.u.shape[1]]
        src_is = [src_i for src_i, src_j, dest_i, dest_j in constraints]
        corner_is = [0, 0, cur_flow.u.shape[0], cur_flow.u.shape[0]]
        src_js = [src_j for src_i, src_j, dest_i, dest_j in constraints]
        corner_js = [0, cur_flow.u.shape[1], cur_flow.u.shape[1], 0]
        u_fixes = np.array([(dest_j - src_j - cur_flow.u[int(src_i), int(src_j)]) 
                            for src_i, src_j, dest_i, dest_j in constraints])
        v_fixes = np.array([(dest_i - src_i - cur_flow.v[int(src_i), int(src_j)]) 
                            for src_i, src_j, dest_i, dest_j in constraints])
        corner_u_fixes = griddata((src_is, src_js), u_fixes, (corner_is, corner_js), method='nearest')
        corner_v_fixes = griddata((src_is, src_js), v_fixes, (corner_is, corner_js), method='nearest')
        src_is += corner_is
        src_js += corner_js
        u_fixes = np.hstack((u_fixes, corner_u_fixes))
        v_fixes = np.hstack((v_fixes, corner_v_fixes))
        cur_flow.u += griddata((src_is, src_js), u_fixes, (gi, gj), method='linear')
        cur_flow.v += griddata((src_is, src_js), v_fixes, (gi, gj), method='linear')
    cur_flow.u = cv2.medianBlur(cur_flow.u, 5)
    cur_flow.v = cv2.medianBlur(cur_flow.v, 5)
    return cur_flow

def find_marks(im):
    marks = {}
    # marks are places with 255 in one channel and 0 in another
    for r in [0, 255]:
        for g in [0, 255]:
            for b in [0, 255]:
                if r == b == g:
                    continue
                mask = (im[:, :, 2] == r) & (im[:, :, 1] == g) & (im[:, :, 0] == b)
                if np.any(mask):
                    nzi, nzj = np.nonzero(mask)
                    marks[(r, g, b)] = (np.mean(nzi), np.mean(nzj))
    return marks

def scale_constraints(marks1, marks2, octaves):
    def gen():
        sc = 0.5 ** octaves
        for k in set(marks1.keys()) &  set(marks2.keys()):
            i1, j1 = marks1[k]
            i2, j2 = marks2[k]
            yield(sc * i1, sc * j1, sc * i2, sc * j2)
    temp = [c for c in gen()]
    return temp if len(temp) else None

if __name__ == "__main__":
    im1 = cv2.imread(sys.argv[1])
    im2 = cv2.imread(sys.argv[2])

    # Find fiducial marks
    marks_1 = find_marks(im1)
    marks_2 = find_marks(im2)

    # convert to grayscale
    im1 = np.mean(im1, axis=2).astype(np.uint8)
    im2 = np.mean(im2, axis=2).astype(np.uint8)

    print "Size", im1.shape, im2.shape


    out = sys.argv[3]

    im1 = im1.astype(np.float32) / 255
    im2 = im2.astype(np.float32) / 255

    l1 = skm.label(im1 == 0)
    im1[l1 == l1[0, 0]] = np.nan
    l2 = skm.label(im2 == 0)
    im2[l2 == l2[0, 0]] = np.nan

    octaves = 2
    print "Downsampling %d times to" % (octaves), "x".join(str(int(s * (0.5 ** octaves))) for s in im1.shape)
    pyramid1 = scalespace(im1, octaves)
    pyramid2 = scalespace(im2, octaves)

    flow = compute_flow(pyramid1[octaves],
                        pyramid2[octaves],
                        alpha=3.0,
                        constraints=scale_constraints(marks_1, marks_2, octaves))

    for o in range(octaves):
        alpha = 5.0 + o * 20.0 / (octaves - 1)
        print "OCTAVE", octaves - o - 1, octaves
        constraints = scale_constraints(marks_1, marks_2, octaves - o - 1)
        if octaves - o < 5:
            print "no corrections"
            constraints = None
        flow = compute_flow(pyramid1[octaves - o - 1],
                            pyramid2[octaves - o - 1],
                            previous_flow=flow,
                            alpha=alpha,
                            constraints=constraints)

    print "saving"
    flow.save(out)



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
