import numpy
import math

from .config import *


def convert_quaternion_to_matrix(q):
    """
    Calculate rotation matrix corresponding to quaternion.
    """

    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < K_FLOAT_EPS:
        return numpy.eye(3)
    
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    
    return numpy.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def convert_matrix_to_quaternion(M):
    """
    Calculate quaternion corresponding to given rotation matrix.
    """

    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    
    K = numpy.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    
    vals, vecs = numpy.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], numpy.argmax(vals)]
    if q[0] < 0:
        q *= -1
    
    return q

def convert_matrix_to_euler(mat, axes='sxyz'):
    """
    Return Euler angles from rotation matrix for specified axis sequence.
    """

    try:
        firstaxis, parity, repetition, frame = K_AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        K_TUPLE2AXES[axes] 
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = K_NEXT_AXIS[i+parity]
    k = K_NEXT_AXIS[i-parity+1]

    M = numpy.array(mat, dtype=numpy.float64, copy=False)[:3, :3]
    
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > K_EPS4:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > K_EPS4:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    
    return ax, ay, az

def convert_euler_to_quaternion(ai, aj, ak, axes='sxyz'):
    """
    Return `quaternion` from Euler angles and axis sequence `axes`
    """
    
    try:
        firstaxis, parity, repetition, frame = K_AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        K_TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = K_NEXT_AXIS[i+parity-1] + 1
    k = K_NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = numpy.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0
    
    return q

def convert_quaternion_to_euler(quaternion, axes='sxyz'):
    """
    Euler angles from `quaternion` for specified axis sequence `axes`
    """
    return convert_matrix_to_euler(
        convert_quaternion_to_matrix(quaternion), axes)