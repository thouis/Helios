from numpy import inner, conjugate, asarray, int, ceil
from numpy.linalg import norm
from scipy.sparse.linalg.isolve.utils import make_system
from warnings import warn

def cg(A, b, x0=None, tol=1e-5, maxiter=None, xtype=None, M=None, callback=None, residuals=None):
    '''Conjugate Gradient on A x = b
    Left preconditioning is supported

    Parameters
    ----------
    A : {array, matrix, sparse matrix, LinearOperator}
        n x n, linear system to solve
    b : {array, matrix}
        right hand side, shape is (n,) or (n,1)
    x0 : {array, matrix}
        initial guess, default is a vector of zeros
    tol : float
        relative convergence tolerance, i.e. tol is scaled by ||b||
    maxiter : int
        maximum number of allowed iterations
    xtype : type
        dtype for the solution, default is automatic type detection
    M : {array, matrix, sparse matrix, LinearOperator}
        n x n, inverted preconditioner, i.e. solve M A A.H x = b.
    callback : function
        User-supplied funtion is called after each iteration as
        callback(xk), where xk is the current solution vector
    residuals : list
        residuals has the residual norm history,
        including the initial residual, appended to it

    Returns
    -------    
    (xNew, info)
    xNew : an updated guess to the solution of Ax = b
    info : halting status of cg

            ==  ======================================= 
            0   successful exit
            >0  convergence to tolerance not achieved,
                return iteration count instead.  
            <0  numerical breakdown, or illegal input
            ==  ======================================= 

    Notes
    -----
    The LinearOperator class is in scipy.sparse.linalg.interface.
    Use this class if you prefer to define A or M as a mat-vec routine
    as opposed to explicitly constructing the matrix.  A.psolve(..) is
    still supported as a legacy.

    Examples
    --------
    >>>from pyamg.krylov import *
    >>>from scipy import rand
    >>>import pyamg
    >>>A = pyamg.poisson((50,50))
    >>>b = rand(A.shape[0],)
    >>>(x,flag) = cg(A,b,maxiter=200, tol=1e-8)
    >>>print pyamg.util.linalg.norm(b - A*x)

    References
    ----------
    Yousef Saad, "Iterative Methods for Sparse Linear Systems, 
    Second Edition", SIAM, pp. 262-67, 2003

    '''
    A,M,x,b,postprocess = make_system(A,M,x0,b,xtype=None)

    n = len(b)
    # Determine maxiter
    if maxiter is None:
        maxiter = int(1.3*len(b)) + 2
    elif maxiter < 1:
        raise ValueError('Number of iterations must be positive')

    # Scale tol by normb
    normb = norm(b) 
    if normb != 0:
        tol = tol*normb

    # setup method
    r  = b - A*x
    z  = M*r
    p  = z.copy()
    rz = inner(conjugate(r), z)

    normr = norm(r)

    if residuals is not None:
        residuals[:] = [normr] #initial residual 

    if normr < tol:
        return (postprocess(x), 0)

    iter = 0

    while True:
        Ap = A*p

        rz_old = rz

        alpha = rz/inner(conjugate(Ap), p)  # 3  (step # in Saad's pseudocode)
        x    += alpha * p                   # 4
        r    -= alpha * Ap                  # 5
        z     = M*r                         # 6
        rz    = inner(conjugate(r), z)          
        beta  = rz/rz_old                   # 7
        p    *= beta                        # 8
        p    += z

        iter += 1

        normr = norm(r)

        if residuals is not None:
            residuals.append(normr)

        if callback is not None:
            callback(x)

        if normr < tol:
            return (postprocess(x), iter)

        if iter == maxiter:
            return (postprocess(x), iter)
