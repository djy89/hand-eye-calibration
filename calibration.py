import numpy as np
import cv2
import vrep
from scene import Scene
from matplotlib import pyplot as plt
from tool import * 
import time

import os

# vrep
ip = '127.0.0.1'
port = 19997
cam_num = 3

train_size = 50
sample_size = 20
iteration=100
score_threshold=0.02
scalar_threshold = 0.0005

def get_sub_mat(qa, qa_prime, qb, qb_prime):
    mat = np.zeros((6,8))
    # equation 1
    mat[:3, 0] = qa[1:] - qb[1:]
    mat[:3, 1:4] = cross_mat(qa[1:] + qb[1:])
    # equation 2
    mat[3:, 0] = qa_prime[1:] - qb_prime[1:]
    mat[3:, 1:4] = cross_mat(qa_prime[1:] + qb_prime[1:])
    mat[3:, 4] = qa[1:] - qb[1:]
    mat[3:, 5:] = cross_mat(qa[1:] + qb[1:])
    return mat

# solve AX=XB by dual quaternion
def dual_quaternion_approach(motionAs, motionBs):
    size = motionAs.shape[0]
    T = []
    for j in range(size):
        motionA = motionAs[j]
        motionB = motionBs[j]
        ra, ta = mat_to_r_t(motionA)
        rb, tb = mat_to_r_t(motionB)
        qa, qa_prime = rot2dualquat(ra, ta)
        qb, qb_prime = rot2dualquat(rb, tb)
        T.append(get_sub_mat(qa, qa_prime, qb, qb_prime))
    T = np.concatenate(T)

    U, s, V = np.linalg.svd(T)
    idx1, idx2 = np.argsort(s)[:2].tolist()
    v7 = V[idx1]
    v8 = V[idx2]
    
    u1 = v7[:4]
    v1 = v7[4:]
    u2 = v8[:4]
    v2 = v8[4:]

    a = np.dot(u1,v1)
    b = np.dot(u1,v2) + np.dot(u2,v1)
    c = np.dot(u2,v2)
    
    s1 = (-b + np.sqrt(b*b-4*a*c)) / (2*a)
    s2 = (-b - np.sqrt(b*b-4*a*c)) / (2*a)

    x1 = s1**2 * np.dot(u1,u1) + 2*s1*np.dot(u1,u2) + np.dot(u2,u2)
    x2 = s2**2 * np.dot(u1,u1) + 2*s2*np.dot(u1,u2) + np.dot(u2,u2)
    (x,s) = (x1,s1) if x1 >= x2 else (x2,s2)

    lambda2 = np.sqrt(1/x)
    lambda1 = s * lambda2

    q = lambda1 * u1 + lambda2 * u2
    q_ = lambda1 * v1 + lambda2 * v2

    r_ba, t_ba = dualquat2r_t(q, q_)
    return r_t_to_mat(r_ba, t_ba), s

def get_error(T_world_cam, motionAs, motionBs, score_threshold):
    motionAs_ = np.matmul(T_world_cam, motionBs)
    motionAs_ = np.matmul(motionAs_, np.linalg.inv(T_world_cam))
    error = np.linalg.norm(motionAs - motionAs_) / np.linalg.norm(motionAs)
    tAs_ = motionAs_[:, :3, 3]
    tAs = motionAs[:, :3, 3]
    error = np.linalg.norm(tAs_ - tAs) / np.linalg.norm(tAs)
    return error

def ransac_for_calibration(motionAs, motionBs, sample_size=20, iteration=10, score_threshold=0.002, show=False):
    best_error = np.inf
    best_result = None

    for i in range(iteration):    
        sample_idxs = np.random.randint(0,motionAs.shape[0],size=(1,sample_size))
        sampled_motionAs = motionAs[sample_idxs.ravel().tolist()]
        sampled_motionBs = motionBs[sample_idxs.ravel().tolist()]
        
        result, singular_values = dual_quaternion_approach(sampled_motionAs, sampled_motionBs)
        error = get_error(result, motionAs, motionBs, score_threshold)

        if error < best_error:
            best_error = error
            best_result = result
        if show:
            print("iter ", i, "error: ", error)
    return best_result, best_error

def get_motion(A, B, scalar_threshold=0.0005, train_size=20, show=False):
    size = A.shape[0]
    motionAs = []
    motionBs = []
    for i in range(size):
        Ai = A[i]
        Bi = B[i]

        for j in range(i+1,size):
            Aj = A[j]
            Bj = B[j]

            motionA = np.matmul(np.linalg.inv(Aj), Ai)
            motionB = np.matmul(Bj, np.linalg.inv(Bi))
            ra, ta = mat_to_r_t(motionA)
            rb, tb = mat_to_r_t(motionB)
            qa, qa_prime = rot2dualquat(ra, ta)
            qb, qb_prime = rot2dualquat(rb, tb)

            # check scalar be equivalent
            diff_scalar = np.abs(qa[0]-qb[0])
            diff_scalar_ = np.abs(qa_prime[0]-qb_prime[0])
            # if show:
            #     print(j, diff_scalar, diff_scalar_)
            if(diff_scalar < scalar_threshold and diff_scalar_ < scalar_threshold):
                motionAs.append(motionA)
                motionBs.append(motionB)
    shuffle_idxs = [i for i in range(len(motionAs))]
    np.random.shuffle(shuffle_idxs)
    if show:
        print('valid motion size: ', len(motionAs))
        print('train size: ', train_size)
    return np.stack(motionAs)[shuffle_idxs][:train_size], np.stack(motionBs)[shuffle_idxs][:train_size]

def hand_eye_calibration(A, B, sample_size=20, iteration=10, score_threshold=0.002, scalar_threshold=0.0005, train_size=20, show=False):
    motionAs, motionBs = get_motion(
        A,
        B,
        scalar_threshold=scalar_threshold, 
        train_size=train_size, 
        show=show
    )
    T_world_cam, error = ransac_for_calibration(
        motionAs, 
        motionBs, 
        sample_size, 
        iteration, 
        score_threshold, 
        show=show
    )
    if show:
        print('best error:', error)
    return T_world_cam

def main():
    input_path = "./calibration"
    cam_paths = [os.path.join(input_path,'camera{:d}'.format(i)) for i in range(cam_num)]

    scene = Scene(ip, port)
    T_world_cams = scene.get_cam_matrixs()
    theta = np.pi
    l = np.array([0,0,1])
    q = np.array([np.cos(theta/2)]+(np.sin(theta/2) * l).tolist())
    r = quat2rot(q)
    cam_fix = r_t_to_mat(r, np.zeros(3))
    T_world_cams = [np.matmul(T_world_cams[i],cam_fix) for i in range(cam_num)]

    for i in range(cam_num):
        print('camera ',i)
        cam_path = cam_paths[i]
        T_world_end = np.loadtxt(os.path.join(cam_path, 'world_end.txt')).reshape((-1,4,4))
        T_cam_obj = np.loadtxt(os.path.join(cam_path, 'cam_obj.txt')).reshape((-1,4,4))

        # hand eye calibration
        T_world_cam = hand_eye_calibration(
            np.linalg.inv(T_world_end)[::-1], 
            T_cam_obj[::-1], 
            sample_size=sample_size, 
            iteration=iteration, 
            score_threshold=score_threshold,
            scalar_threshold=scalar_threshold,
            train_size=train_size,
            show=True
        )

        # check X
        print('final check')
        thetaj, nj, tj = mat_to_theta_n_t(T_world_cam)
        thetaj_, nj_, tj_ = mat_to_theta_n_t(T_world_cams[i])
        print(thetaj, nj, tj)
        print(thetaj_, nj_, tj_)
        print(
            "theta:{:6f}   n:{:6f}   t:{:6f}".format(
                np.linalg.norm(thetaj-thetaj_), 
                np.linalg.norm(nj-nj_)/np.linalg.norm(nj), 
                np.linalg.norm(tj-tj_)/np.linalg.norm(tj)
            )
        )
        print('-----------------------------------------------------')

        np.savetxt(os.path.join(cam_path, 'world_cam.txt'),T_world_cam)

if __name__ == "__main__":
    main()