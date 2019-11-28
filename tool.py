import numpy as np
import cv2

def rot2quat(r):
    rv,_ = cv2.Rodrigues(r)
    theta = np.linalg.norm(rv)
    l = rv / theta
    q = np.concatenate([[np.cos(theta/2)], np.sin(theta/2) * l.reshape(-1)])
    return q

def cross_mat(v):
    x,y,z = v.tolist()
    mat = np.array(
        [
            [0,-z,y],
            [z,0,-x],
            [-y,x,0]
        ]
    )
    return mat

def quat_inv(q):
    return quat_con(q) / (q*q).sum()

def quat_con(q):
    q_inv = -q
    q_inv[0] = - q_inv[0]
    return q_inv

def quat_mul(q1, q2):
    q3 = np.zeros_like(q1)
    q3[0] = q1[0]*q2[0] - np.dot(q1[1:], q2[1:])
    q3[1:] = np.cross(q1[1:], q2[1:]) + q1[0]*q2[1:] + q2[0]*q1[1:]
    return q3

def quat2rot(q):
    w,x,y,z = q.tolist()
    r = np.array(
        [
            [1-2*(y*y+z*z), 2*(x*y-z*w),    2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z),  2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),    1-2*(x*x+y*y)]
        ]
    )
    return r

def r_t_to_mat(r, t):
    mat = np.zeros((4,4))
    mat[:3,:3] = r
    mat[:3,3] = t.T
    mat[3,3] = 1
    return mat

def mat_to_r_t(mat):
    r = mat[:3,:3]
    t = mat[:3,3].reshape(-1)
    return r,t

def rot2dualquat(r,t):
    q = rot2quat(r)
    qprime = 0.5 * quat_mul(np.concatenate([[0],t]),q)
    return q, qprime

def dualquat2r_t(q1, q1prime):
    r = quat2rot(q1)
    t = (2 * quat_mul(q1prime, quat_con(q1)))[1:]
    return r, t

def dualquat_mul(q1, q1prime, q2, q2prime):
    q3 = quat_mul(q1, q2)
    q3prime = quat_mul(q1,q2prime) + quat_mul(q1prime, q2)
    return q3, q3prime

def dualquat_con(q1, q1prime):
    return quat_con(q1), quat_con(q1prime)

def mat_to_theta_n_t(mat):
    r, t = mat_to_r_t(mat)
    rv, _ = cv2.Rodrigues(r)
    theta = np.linalg.norm(rv)
    rv = (rv / theta).ravel()
    return theta, rv, t