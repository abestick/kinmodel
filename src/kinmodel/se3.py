# This library is provided under the GNU LGPL v3.0 license, with the 
# additional provision that any use in work leading to academic 
# publication requires at least one of the following conditions be met:
# (1) You credit all authors of the library, by name, in the publication
# (2) You reference at least one relevant paper by each of the authors
# 
# (C) Shai Revzen, U Penn. 2011
# (C) Sam Burden, UC Berkeley, 2013

"""
File se3.py contains a library of functions for performing operations
in the classical Lie Group SE(3) and associated algebra se(3).

In the implementation, special care has been taken to provide vectorized
functions -- functions which operate in batch on multiple parameters.
In most cases, this takes the form of using the first indices of an
array and reusing the remaining indices, e.g. skew() applied to an array
of shape (4,7,3) would return result with shape (4,7,3,3), such that for
all i,j: result[i,j,:,:] = skew( parameter[i,j,:] )

Additionally, many functions have an _UNSAFE variant which skips all
validity checking on the inputs. These variants are provided for 
performance reasons. In general, USE THE SAFE FUNCTIONS unless you are
hitting a performance bottleneck and you are sure that the preconditions
for the computation are met.

NOTE to developers:
This library uses python doctest to ensure that it performs as
specified in the documentation. Please run with -v if you think
something isn't working right, and send the failure case report
"""

from numpy import zeros, zeros_like, asarray, array, allclose, resize, ones, ones_like, empty, identity, prod, sqrt, isnan, pi, cos, sin, newaxis, diag, sum, arange, dot

import numpy as np

from numpy.linalg import inv, eig

def skew( v ):
    """
    Convert a 3-vector to a skew matrix such that 
        dot(skew(x),y) = cross(x,y)
    
    The function is vectorized, such that:
    INPUT:
        v -- 3 x ... x N -- input vectors
    OUTPUT:
        3 x 3 x ... x N
    """
    v = asarray(v)
    z = zeros_like(v[0,...])
    return array([[ z,       -v[2,...], v[1,...]],
                  [ v[2,...], z,       -v[0,...]],
                  [-v[1,...], v[0,...], z,     ]])

def unskew( S ):
    """
    Convert a skew matrix to a 3-vector
    
    The function is vectorized, such that:
    INPUT:
        S -- 3 x 3 x N... -- input skews
    OUTPUT:
        3 x N...    
    
    This is the "safe" function -- it tests for skewness first.
    Use unskew_UNSAFE(S) to skip this check
    
    Example:
    >>> x = array(range(24)).reshape(2,1,4,3); allclose(unskew(skew(x)),x)
    True
    >>> unskew([[1,2,3],[4,5,6],[7,8,9]])
    Traceback (most recent call last):
    ...
    AssertionError: S is skew
    """
    S = asarray(S)
    assert allclose(S.T.transpose((1,0)+tuple(range(2,S.ndim))),-S.T),"S is skew"
    return unskew_UNSAFE(S)

def unskew_UNSAFE(S):
    """
    Convert a skew matrix to a 3-vector
    
    The function is vectorized, such that:
    INPUT:
        S -- N... x 3 x 3 -- input skews
    OUTPUT:
        N... x 3    
    
    This is the "unsafe" function -- it does not test for skewness first.
    Use unskew(S) under normal circumstances
    """
    S = asarray(S)
    return array([S[2,1,...],S[0,2,...],S[1,0,...]])

def hat_( v ):
    """
    Convert a 6-vector twist to homogeneous twist matrix
    
    INPUT:
        v - 6 - twist vector (om, v)
    OUTPUT:
        4 x 4   
    """
    v = asarray(v)
    z = zeros((4,4))
    return array([
            [ 0,    -v[5], v[4],  v[0] ],
            [ v[5], 0,     -v[3], v[1] ],
            [-v[4], v[3],  0,     v[2] ],
            [ 0,    0,     0,     0    ]])

def hat( v ):
    """
    Convert a 6-vector twist to homogeneous twist matrix
    
    The function is vectorized, such that:
    INPUT:
        v -- 6 x ... x N -- twist vectors (om, v)
    OUTPUT:
        4 x 4 x ... x N
    """
    v = asarray(v)
    z = zeros_like(v[0,...])
    return array([
            [ z,       -v[5,...], v[4,...],  v[0,...] ],
            [ v[5,...], z,        -v[3,...], v[1,...] ],
            [-v[4,...], v[3,...], z,         v[2,...] ],
            [ z,        z,        z,         z      ] ])

def unhat( S ):
    """
    Convert a homogeneous twist matrix to a 6-vector twist 
    
    INPUT:
        S - 4 x 4 x ... x N - twist matrices
    OUTPUT:
        xi - 6 x ... x N - twist vectors (om, v)
    """
    #S = asarray(S)
    #return asarray([S[2,1],S[0,2],S[1,0],S[0,3],S[1,3],S[2,3]])
    S = asarray(S)
    return array([S[0,3,...],S[1,3,...],S[2,3,...],S[2,1,...],S[0,2,...],S[1,0,...]])

def screw( v ):
    """
    Convert a 6-vector to a screw matrix 
    
    The function is vectorized, such that:
    INPUT:
        v -- N... x 6 -- input vectors
    OUTPUT:
        N... x 4 x 4    
    """
    v = asarray(v)
    z = zeros_like(v[0,...])
    o = ones_like(v[0,...])
    return array([
            [ z,        -v[...,5], v[...,4],  v[...,0]],
            [ v[...,5], z,         -v[...,3], v[...,1]],
            [-v[...,4], v[...,3],  z,         v[...,2]],
            [ z,        z,         z,         o       ]])

#def unscrew( S ):
#   """
#   Convert a screw matrix to a 6-vector
#   
#   The function is vectorized, such that:
#   INPUT:
#       S -- N... x 4 x 4 -- input screws
#   OUTPUT:
#       N... x 6
#   
#   This is the "safe" function -- it tests for screwness first.
#   Use unscrew_UNSAFE(S) to skip this check
#   """
#   S = asarray(S)
#   assert allclose(S[...,:3,:3].transpose(0,1),-S[...,:3,:3]),"S[...,:3,:3] is skew"
#   assert allclose(S[...,3,:3],0),"Bottom row is 0"
#   assert allclose(S[...,3,3],1),"Corner is 1"
#   return unscrew_UNSAFE(S)

#def unscrew_UNSAFE(S):
#   """
#   Convert a screw matrix to a 6-vector
#   
#   The function is vectorized, such that:
#   INPUT:
#       S -- N... x 4 x 4 -- input screws
#   OUTPUT:
#       N... x 6 
#   
#   This is the "unsafe" function -- it does not test for screwness first.
#   Use unscrew(S) under normal circumstances
#   """
#   S = asarray(S)
#   return array([S[...,1,2],S[...,2,0],S[...,0,1],
#       S[...,0,3],S[...,1,3],S[...,2,3]])

def aDot( A, B ):
    """Similar to dot(...) but allows B.shape[-2] to equal A.shape[-1]-1
    for affine operations.
    
    Example:
    >>> aDot( seToSE([0,pi/2,0,1,1,1.0]), identity(3) )
    array([[ 1.,    1., 0.],
             [ 1.,  2., 1.],
             [ 2.,  1., 1.]])
    """
    if B.shape[-2] != A.shape[-1]-1:
        return dot(A,B)
    C = dot( A[...,:-1,:-1],B )
    C += A[...,:-1,[-1]]
    return C

def aaToRot( aa ):
    """
    Convert skew vector into a rotation matrix using Rodregues' formula
    i.e. Eqn. 2.14 in MLS
    INPUT:
        aa - 3 x ... x N
    OUTPUT:
        3 x 3 x ... x N
    
    >>> diag(aaToRot([3.1415926,0,0])).round(2)
    array([ 1., -1., -1.])
    >>> aaToRot([0,3.1415926/4,0]).round(2)
    array([[ 0.71,  0.  , -0.71],
                 [ 0.   ,   1.  ,   0.  ],
                 [ 0.71,    0.  ,   0.71]])
    """
    aa = asarray(aa)
    t = sqrt(sum(aa * aa,0))
    k = aa / t[newaxis,...]
    k[isnan(k)]=0
    kkt = k[:,newaxis,...] * k[newaxis,:,...]
    I = identity(3)
    # Note: (a.T+b.T).T is not a+b -- index broadcasting is different
    R = ((sin(t)*skew(k) + (cos(t)-1)*(I-kkt.T).T).T + I).T
    return R

def expso3(om, th=None, dbg=False):
    """
    expm( skew( om ) )
    i.e. Eqn. 2.14 in MLS
    """
    om = np.asarray(om).flatten()
    assert om.size == 3 # so(3)
    om = om[:3][:,np.newaxis]
    I = np.identity(3)
    if allclose(om,0):
        R = I
    else:
        if th is None:
            th = np.sqrt((om**2).sum())
            om = om/th
        Om = skew( om.flatten() ) 
        R = I + np.sin(th) * Om + (1 - np.cos(th)) * np.dot(Om, Om)
    if dbg:
        1/0
    return R

def expse3(xi, th=1., dbg=False):
    """
    expm( hat( xi ) * th )
    """
    xi = np.asarray(xi).flatten()
    assert xi.size == 6 # se3 twist
    v, om = xi[:3][:,np.newaxis],xi[3:][:,np.newaxis]
    I = np.identity(3)
    # Case 1 (om = 0) in MLS Prop 2.8
    if allclose(om,0):
        # Eqn. 2.32 in MLS
        p = v*th
        R = I
    # Case 2 (om \ne 0) in MLS Prop 2.8
    else:
        ph = np.sqrt((om**2).sum())
        om = om/ph; v = v/ph
        R = expso3( om, ph*th )
        #R = aaToRot( ph*th*om.flatten() )
        # Eqn. 2.36 in MLS
        p = (np.dot( I - R, np.dot( skew( om.flatten() ), v ) ) 
                 + om * np.dot(om.T, v) * ph*th )
        if dbg:
            1/0
    return homog(p,R)

def expse3_(xi, th=1., dbg=False):
    """
    expm( hat( xi ) * th )
    i.e. Prob 2.8 in MLS

    Input:
        xi - 6 x ... x N - (om, v)
        th - ... x N
    Output:
        4 x 4 x ... x N - homogeneous matrix, expm( hat( xi ) * th )
    """
    #xi = np.asarray(xi).flatten()
    xi = np.asarray(xi)
    #assert xi.size == 6 # se3 twist
    #v, om = xi[:3][:,np.newaxis],xi[3:][:,np.newaxis]
    v, om = xi[:3],xi[3:]
    I = np.identity(3)
    # Case 2 (om \ne 0) in MLS Prop 2.8
    ph = np.sqrt(np.sum(om * om,0))
    om = om / ph[np.newaxis,...]; v = v / ph[np.newaxis,...]
    R = aaToRot( ph*th*om )
    # Eqn. 2.36 in MLS
    #a = np.einsum('ij...,j...->i...', skew(om), v)
    #b = np.einsum('ij...,j...->i...', (I - R.T).T, a)
    #c = np.sum(om * v,0) * om * ph*th
    #p = b + c
    p = np.einsum('ij...,j...->i...', (I - R.T).T, np.einsum('ij...,j...->i...', skew(om), v)) + np.sum(om * v,0) * om * ph*th
    if dbg:
        1/0
    return homog_(p,R)

def dexpse3(xi, th=1., dbg=False):
    """
    (dxi, dth) expm( hat( xi ) * th )

    Eqn. (33) and (14) of:
    @article{HeZhaoYangYang2010,
                    Title = {Kinematic-Parameter Identification for Serial-Robot Calibration Based on POE Formula},
                    Author = {Ruibo He and Yingjun Zhao and Shunian Yang and Shuzi Yang},
                    Journal = {IEEE Transactions on Robotics},
                    Volume = {26},
                    Number = {3},
                    Pages = {411-423},
                    Year = {2010}}
    """
    xi = np.asarray(xi).flatten()
    assert xi.size == 6 # se3 twist
    v, om = xi[:3][:,np.newaxis],xi[3:][:,np.newaxis]
    I = np.identity(6)
    # om = 0; Eqn. (33) in \cite{HeZhaoYangYang2010}
    if allclose(om,0):
        dxi = th*I*[1.,1.,1.,0.,0.,0.]
        dth = xi[:,np.newaxis]
    # om \ne 0; Eqn (14) in \cite{HeZhaoYangYang2010}
    else:
        Om = ad(xi)
        Om2 = np.dot(Om,Om); Om3 = np.dot(Om2,Om); Om4 = np.dot(Om3,Om)
        s = np.sqrt((om**2).sum())
        ph = s*th
        dxi = ( th*I + (4       - ph*np.sin(ph) -   4*np.cos(ph)) * Om  / (2*s**2)
                                 + (4*ph -  5*np.sin(ph) + ph*np.cos(ph)) * Om2 / (2*s**3)
                                 + (2       - ph*np.sin(ph) -   2*np.cos(ph)) * Om3 / (2*s**4)
                                 + (2*ph -  3*np.sin(ph) + ph*np.cos(ph)) * Om4 / (2*s**5) )
        dth = xi[:,np.newaxis]
        if dbg:
            1/0
    return dxi,dth

def dexpse3_(xi, th=1., dbg=False):
    """
    derivative of rigid body transformation represented in Lie algebra
    ((dxi, dth) expm( hat( xi ) * th )) expm( hat( xi ) * (-th) ) 
    
    For derivative in spatial coordinates, right-multiply by expm(hat(xi)*th)

    Eqn. (14) of:
    @article{HeZhaoYangYang2010,
                    Title = {Kinematic-Parameter Identification for Serial-Robot Calibration Based on POE Formula},
                    Author = {Ruibo He and Yingjun Zhao and Shunian Yang and Shuzi Yang},
                    Journal = {IEEE Transactions on Robotics},
                    Volume = {26},
                    Number = {3},
                    Pages = {411-423},
                    Year = {2010}}

    Input:
        xi - 6 x ... x N - (om, v)
        th - ... x N
    Output:
        dxi - 6 x 6 x ... x N - derivatives as twists 
        dth - 6 x 1 x ... x N - derivatives as twists 
    """
    xi = np.asarray(xi)
    v, om = xi[:3],xi[3:]
    sh = list(xi.shape[1:])
    I = np.identity(6).reshape([6,6]+[1]*(len(xi.shape)-1))

    Om = ad_(xi)
    Om2 = np.einsum('ij...,jk...->ik...',Om ,Om)
    Om3 = np.einsum('ij...,jk...->ik...',Om2,Om)
    Om4 = np.einsum('ij...,jk...->ik...',Om3,Om)
    s = np.sqrt(np.sum(om * om,0))
    ph = s*th
    dxi = (th*I +((4        - ph*np.sin(ph) -   4*np.cos(ph)) * Om  / (2*s**2)
                            + (4*ph -   5*np.sin(ph) + ph*np.cos(ph)) * Om2 / (2*s**3)
                            + (2        - ph*np.sin(ph) -   2*np.cos(ph)) * Om3 / (2*s**4)
                            + (2*ph -   3*np.sin(ph) + ph*np.cos(ph)) * Om4 / (2*s**5)) )
    dth = xi[:,np.newaxis,...]

    assert not np.any(np.isnan(dxi))

    if dbg:
        1/0
    return dxi,dth

def seToSE( x ):
    """
    Convert a twist (a rigid velocity, element of se(3)) to a rigid
    motion (an element of SE(3))
    
    INPUT:
        x -- 6 x N...
    OUTPUT:
        3 x 3 x N...    

    !!!WARNING: VEC
    """
    x = asarray(x)
    R = aaToRot(x[:3,...])
    X = empty( (4,4)+x.shape[1:], dtype=x.dtype )
    X[:3,:3,...] = R
    X[3,:3,...] = 0
    X[3,3,...] = 1
    X[:3,3,...] = x[3:,...]
    return X
    
def Adj( K ):
    """
    Note: This is the adjoint operator for a twist (v,om), as opposed
    to the (v, om) convention used by most other functions. -AB
    
    The Adj operator for a rigid motion K
    
    This is the "safe" version which checks that K is a rigid motion
    
    !!!WARNING: VEC
    """
    K = asarray(K)
    sz = K.shape[2:]
    assert K.shape[:2] == (4,4)
    N = int(prod(sz))
    K.shape = (4,4,N)
    I = identity(3)
    for k in xrange(N):
        assert allclose(dot(K[:3,:3,k],K[:3,:3,k].T),I), "K[:3,:3] is a rotation"
    assert allclose(K[3,:3,:],0),"bottom row is 0"
    assert allclose(K[3,3,:],1),"corner is 1"
    A = Adj_UNSAFE(K)
    A.shape = (A.shape[0],A.shape[1])+sz
    return A
    
def Adj_UNSAFE(K):
    """
    The Adj operator for a rigid motion K
    
    This is the "safe" version which checks that K is a rigid motion
 
    !!!WARNING: VEC
    """
    K = asarray(K)
    if K.ndim == 2:
        K = K[...,newaxis]
    assert K.ndim == 3
    n = K.shape[2]
    t = K[:3,3,:]
    S = skew(t)
    R = K[:3,:3,:]
    z = zeros_like
    RS = array([ -dot(R[...,k],S[...,k]) for k in xrange(n) ])
    A = zeros((6,6,n),K.dtype)
    for k in xrange(n):
        Rk = R[...,k]
        A[:3,3:,k] = -dot(Rk,S[...,k])
        A[:3,:3,k] = Rk
        A[3:,3:,k] = Rk
    return A

def Adjoint(p,R):
    """
    Ad = Adjoint(p,R)

    Inputs:
        p - n x 1 - position
        R - n x n - orientation

    Outputs:
        Ad - 2n x 2n - Adjoint transformation for twists xi = (om, v) \in se(3)
             = ( R               0 )
                 ( skew(p)*R R )
    """
    n = p.size
    assert R.shape[0] == R.shape[1], "R is square"
    assert R.shape[0] == n, "R and p have compatible shapes"
    p = p.flatten()
    # HeZhao2010 Eqn. (49); note that xi = (om, v)
    Ad = np.r_[np.c_[R, np.zeros((n,n))], np.c_[np.dot(skew(p),R), R]]
    return Ad

def Ad(p,R):
    """
    Ad(p,R)

    Inputs:
        p - n x 1 - position
        R - n x n - orientation

    Outputs:
        Ad - 2n x 2n - Adjoint transformation for twists xi = (om, v) \in se(3)
             = ( R               0 )
                 ( skew(p)*R R )
    """
    return Adjoint(p,R)

def ad(xi):
    """
    ad = ad(xi)

    Inputs:
        xi - (v, om) \in R^n

    Outputs:
        ad - 2n x 2n - adjoint transformation
             = ( skew(om) 0             )
                 ( skew(v)  skew(om) )
    """
    xi = xi.flatten(); n = xi.size/2
    vh, omh = skew(xi[:3]),skew(xi[3:])
    ad = np.r_[np.c_[omh, np.zeros((n,n))], np.c_[vh, omh]]
    return ad

def ad_(xi):
    """
    ad = ad(xi)

    Inputs:
        xi - (v, om) \in R^n

    Outputs:
        ad - 2n x 2n - adjoint transformation
             = ( skew(om) 0             )
                 ( skew(v)  skew(om) )
    """
    n,s = xi.shape[0]/2,list(xi.shape[1:])
    v, om = skew(xi[:3]),skew(xi[3:])

    ad = np.vstack(( np.hstack(( om, np.zeros([n,n]+s) )),
                                     np.hstack(( v, om )) ))

    return ad


def homog(p,R):
    """
    g = homog(p,R)
    
    Inputs:
        p - n x 1 - position
        R - n x n - orientation

    Outputs:
        g - (n+1) x (n+1) - homogeneous representation
            = ( R p )
                ( 0 1 ) 
    """
    n = p.size
    assert R.shape[0] == R.shape[1], "R is square"
    assert R.shape[0] == n, "R and p have compatible shapes"
    p = p.flatten()[:,np.newaxis]
    zo = np.r_[np.zeros(n),1.]
    g = np.r_[ np.c_[R,p], [zo]]
    return g

def homog_(p,R):
    """
    g = homog(p,R)
    
    Inputs:
        p - n x 1 - position
        R - n x n - orientation

    Outputs:
        g - (n+1) x (n+1) - homogeneous representation
            = ( R p )
                ( 0 1 ) 
    """
    #n = p.shape[0]
    #assert R.shape[0] == R.shape[1], "R is square"
    #assert R.shape[0] == n, "R and p have compatible shapes"
    #p = p.flatten()[:,np.newaxis]
    #zo = np.r_[np.zeros(n),1.]
    #g = np.r_[ np.c_[R,p], [zo]]
    if p.size == 3 and len(p.shape) <= 1:
        p = p[:,np.newaxis]
    if len(R.shape) <= 2:
        R = R[...,np.newaxis]
    n,s = p.shape[0],list(p.shape[1:])
    g = np.vstack(( np.hstack(( R, p[...,np.newaxis,:] )), 
                                    np.hstack(( np.zeros([1,n]+s), np.ones([1,1]+s) )) ))
    if len(g.shape) == 3 and g.shape[2] == 1:
        g = g[...,0]
    return g

def unhomog(g):
    """
    p,R = unhomog(g)
    
    Inputs:
        g - ( R p )
                ( 0 1 ) - homogeneous representation

    Outputs:
        p - n x 1 - position
        R - n x n - orientation
    """
    assert g.shape[0] == g.shape[1], "g is square"
    n = g.shape[0]-1
    g = g[:n]
    p = g[:,n]
    R = g[:n,:n]
    return p,R

def cayley_UNSAFE( R, inv=inv ):
    """
    Compute the Cayley transform of the matrices R
         
    C = dot(inv(R + I), (R-I))
    
    This transform takes rotations near the identity into skew matrices.
    It is its own inverse.
         
    INPUT:
        R -- N... x D x D -- a collection of orthogonal matrices
        inv -- matrix inversion function
    OUTPUT:
        N... x D x D
    """
    R = asarray(R)
    shp = R.shape
    N = int(prod(shp[:-2]))
    D = shp[-1]
    R.shape = (N,D,D)
    res = empty_like(R)
    idx = arange(D)
    for k in xrange(N):
        # Fast add, subtract identity
        A = R[k,...].copy()
        A[idx,idx] += 1
        B = R[k,...].copy()
        B[idx,idx] -= 1
        # Cayley
        res[k,...] = dot( inv(A), B )
    res.shape = shp
    R.shape = shp
    return res

def powm( M, p ):
    """
    Compute M^p for a matrix using the eigenvalue decomposition method
    
    Taking M = V diag(d) inv(V), M^p = V diag(pow(d,p)) inv(V)
    
    INPUT:
        M -- D x D -- matrix
        p -- float -- an exponent
            -- N... -- multiple exponents to be applied
    OUTPUT:
        D x D -- if p was a single float
        N... x D x D -- otherwise
    """
    M = asarray(M)
    p = asarray(p)
    assert M.shape == (M.shape[0],M.shape[0]), "Matrix is square"
    d,V = eig(M)
    res = [dot(dot(V,diag(d ** pk)), inv(V)).real for pk in p.flat]
    return asarray(res).squeeze().reshape(p.shape+M.shape)
    
def geodesicInterpolation( R0, R1, x=0.5 ):
    """
    Interpolate between R0 and R1 along the SO(D) geodesic between
    them. By default, computes the geodesic mean.
    
    INPUT:
        R0, R1 -- D x D --- orthogonal matrices
        x -- N... -- multiple interpolation positions; default is 0.5
    OUTPUT:
        D x D -- if p was a single float
        N... x D x D -- otherwise
        
    Example:
    >>> R0 = aaToRot([1,2,0])
    >>> R1 = aaToRot([1,0,2])
    >>> g = geodesicInterpolation(R0,R1)
    >>> allclose( dot(R0.T,g), dot(g.T,R1) )
    True
    """
    P = powm(dot(R0,R1.T),x)
    return dot(P,R1)

def jac(xi, ths, frame='spatial'):
    """
    Compute the manipulator Jacobian with the given joint angles
    
    INPUT:
        xi -- 6 x N -- manipulator twists (v, om)
        ths -- N -- manipulator joint angles
        frame -- either 'body' or 'spatial'
    
    OUTPUT:
        jac -- 6 x N -- the manipulator Jacobian
    """
    
    assert frame=='spatial', 'Body frame jacobian is not supported yet'
    jac = np.zeros(xi.shape)
    trans = np.eye(4)
    
    #Transform each twist
    for i in range(xi.shape[-1]):
        jac[:,i] = np.dot(Ad(trans[0:3,3], trans[0:3,0:3]), xi[:,i])
        trans = np.dot(trans, expse3(xi[:,i], ths[i]))
    return jac
    

if __name__ == "__main__":
        import doctest
        doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)

