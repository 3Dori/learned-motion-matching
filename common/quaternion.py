# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

# PyTorch-backed implementations

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def qconj(q):
    assert q.shape[-1] == 4
    return torch.tensor([1, -1, -1, -1], dtype=torch.float32) * q


def angular_velocity(q, fps):
    # See https://blog.csdn.net/zhoupian/article/details/96974091
    assert q.shape[-1] == 4

    q_conj = qconj(q[1:])
    return 2 * qmul(torch.diff(q, axis=0), q_conj) * fps


def qinv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q


def qrot_inv_np(q, v):
    return qrot_np(qinv(q), v)

# Numpy-backed implementations

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()

def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()


def angular_velocity_np(q, fps):
    q = torch.from_numpy(q).contiguous()
    return angular_velocity(q, fps).numpy()


def log(q, eps=1e-5):
    length = np.sqrt(np.sum(np.square(q[..., 1:]), axis=-1))[..., np.newaxis]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, q[..., 0:1]) / length)
    return halfangle * q[..., 1:]


def to_scaled_angle_axis_np(q, eps=1e-5):
    return 2.0 * log(q, eps)


def to_xform_xy_np(q):
    # two-column form
    qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz], axis=-1)[..., np.newaxis, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz)], axis=-1)[..., np.newaxis, :],
        np.concatenate([xz - wy, yz + wx], axis=-1)[..., np.newaxis, :],
    ], axis=-2)


def to_xform_np(q):
    # two-column form
    qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[..., np.newaxis, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[..., np.newaxis, :],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[..., np.newaxis, :],
    ], axis=-2)



def from_xy(x):
    c2 = _fast_cross(x[..., 0], x[..., 1])
    c2 = c2 / torch.sqrt(torch.sum(torch.square(c2), dim=-1))[..., None]
    c1 = _fast_cross(c2, x[..., 0])
    c1 = c1 / torch.sqrt(torch.sum(torch.square(c1), dim=-1))[..., None]
    c0 = x[..., 0]

    return torch.cat([
        c0[..., None],
        c1[..., None],
        c2[..., None]
    ], dim=-1)


def _fast_cross(a, b):
    return torch.cat([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]
    ], dim=-1)


def _fast_cross_np(a, b):
    return np.concatenate([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)


def fk_vel_np(lrot, lpos, lvel, lang, parents):
    gp, gr, gv, ga = [lpos[..., :1, :]], [lrot[..., :1, :]], [lvel[..., :1, :]], [lang[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(qrot_np(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(qmul_np(gr[parents[i]], lrot[..., i:i + 1, :]))
        gv.append(qrot_np(gr[parents[i]], lvel[..., i:i + 1, :]) +
                  _fast_cross_np(ga[parents[i]], qrot_np(gr[parents[i]], lpos[..., i:i + 1, :])) +
                  gv[parents[i]])
        ga.append(qrot_np(gr[parents[i]], lang[..., i:i + 1, :]) + ga[parents[i]])

    return (
        np.concatenate(gr, axis=-2),
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))


def _mul(x, y):
    return torch.matmul(x, y)


def _mul_vec(x, v):
    return torch.matmul(x, v[..., None])[..., 0]


def fk_vel(lrot, lpos, lvel, lang, parents):
    # lrot is a transform matrix
    gp, gr, gv, ga = [lpos[..., :1, :]], [lrot[..., :1, :, :]], [lvel[..., :1, :]], [lang[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(_mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(_mul(gr[parents[i]], lrot[..., i:i + 1, :, :]))
        gv.append(_mul_vec(gr[parents[i]], lvel[..., i:i + 1, :]) +
                  torch.cross(ga[parents[i]], _mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]), dim=-1) +
                  gv[parents[i]])
        ga.append(_mul_vec(gr[parents[i]], lang[..., i:i + 1, :]) + ga[parents[i]])

    return (
        torch.cat(gr, dim=-3),
        torch.cat(gp, dim=-2),
        torch.cat(gv, dim=-2),
        torch.cat(ga, dim=-2))


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)
            
    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1
    
    return result.reshape(original_shape)
    