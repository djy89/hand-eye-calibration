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

# tool for camera calibration
cam_num = 3
min_size = 50
scale = (0.2/8.0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
pattern_size = (7, 5)
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * scale

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
board = cv2.aruco.CharucoBoard_create(3,4,0.05,0.05*0.75, dictionary)

def draw_axis(img, r, t, mtx, dist):
    axis = np.float32([[0,0,0],[2,0,0],[0,2,0],[0,0,-2]]).reshape(-1,3) * scale
    pp, _ = cv2.projectPoints(axis, r, t, mtx, dist)

    img = cv2.line(img, tuple(pp[0].ravel()), tuple(pp[1].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(pp[0].ravel()), tuple(pp[2].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(pp[0].ravel()), tuple(pp[3].ravel()), (0,0,255), 3)
    return img

def main():
    output_filename = './calibration'
    if not os.path.exists(output_filename):
        os.makedirs(output_filename)
    cam_paths = [os.path.join(output_filename, 'camera{:d}'.format(i)) for i in range(cam_num)]
    for cam_path in cam_paths:
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)
        cam_img_path = os.path.join(cam_path,'img')
        if not os.path.exists(cam_img_path):
            os.makedirs(cam_img_path)

    T_world_end = [[] for i in range(cam_num)]
    T_cam_obj = [[] for i in range(cam_num)]

    # get T(world->end), image and items for camera calibration
    images_collected = [[] for i in range(cam_num)]
    # objpoints = [[] for i in range(cam_num)]
    # imgpoints = [[] for i in range(cam_num)]
    all_corners = [[] for i in range(cam_num)]
    all_ids = [[] for i in range(cam_num)]
    scene = Scene(ip, port)
    finish = False
    while not finish:
        position = (((np.random.rand(2)-0.5)*0.3)+0.4).tolist() # (x,y) = (0.4,0.4) +/- rand(0, 0.15)
        position += [np.random.rand() * 0.1 + 0.2]
        quaternion = scene.ur5.init_quaternion
        scene.ur5.move_to_object_position(position+quaternion)
        scene.ur5.rotate( np.random.rand()*2*np.pi )
        time.sleep(2)
        _, images = scene.get_image()
        
        finish=True
        for i in range(cam_num):
            gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
            h,w = gray.shape
            # ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None, cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_EXHAUSTIVE)
            # if ret is True:
            #     first_corner = corners[0][0]
            #     last_corner = corners[-1][0]
            #     if first_corner[0] > last_corner[0]:
            #         corners1 = corners[::-1,:,:]
            #     else:
            #         corners1 = corners
            #     cv2.cornerSubPix(gray, corners1, (5,5), (-1, -1), criteria)
            #     corners1 = corners1.reshape(-1, 2)
            #     # items for camera calibration
            #     imgpoints[i].append(corners1)
            #     objpoints[i].append(objp)
            #     images_collected[i].append(images[i])
            #     # get T(world->end)
            #     mat = scene.ur5.get_end_matrix()
            #     time.sleep(1)
            #     T_world_end[i].append(mat)
            
            # if len(objpoints[i]) < min_size:
            #     finish = False
            corners, ids, _  = cv2.aruco.detectMarkers(gray, board.dictionary)
            if ids is not None:
                ret, corners_c, ids_c =  cv2.aruco.interpolateCornersCharuco(corners, ids, images[i], board)
                if ids_c is not None and corners_c.shape[0] >= 4:
                    # items for calibration
                    all_corners[i].append(corners_c)
                    all_ids[i].append(ids_c)
                    images_collected[i].append(images[i])
                    # get T(world->end)
                    mat = scene.ur5.get_end_matrix()
                    time.sleep(2)
                    T_world_end[i].append(mat)
            if len(all_ids[i]) < min_size:
                finish = False

        for i in range(cam_num):
            print(len(all_ids[i]),end=' ')
        print()
    
    # camera calibration
    for i in range(cam_num):
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints[i], imgpoints[i], (w,h), None, None, criteria=criteria)
        # all_corners = np.stack(all_corners)
        # all_ids = np.stack(all_ids)
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners[i], all_ids[i], board, (w,h), None, None)
        # save camera intrinsic
        np.savetxt(os.path.join(cam_paths[i],'camera_intrinsic.txt'), np.concatenate([mtx.ravel(), dist.ravel()]))
        # get T(cam->obj)
        for j in range(len(rvecs)):
            r,_ = cv2.Rodrigues(rvecs[j])
            t = tvecs[j].reshape(-1)
            mat = r_t_to_mat(r, t)
            T_cam_obj[i].append(mat)
            # plot and save the posed img
            img = draw_axis(images_collected[i][j], rvecs[j], tvecs[j], mtx, dist)
            cv2.imwrite(os.path.join(cam_paths[i],'img','{:d}.png'.format(j)), img)

    # save T(cam->obj) and T(world->end)
    for i in range(cam_num):
        np.savetxt(os.path.join(cam_paths[i],'cam_obj.txt'), np.concatenate(T_cam_obj[i]))
        np.savetxt(os.path.join(cam_paths[i],'world_end.txt'), np.concatenate(T_world_end[i]))
    
    vrep.simxStopSimulation(scene.client_id,operationMode=vrep.simx_opmode_blocking)

def test():
    input_path = "./calibration"
    cam_paths = [os.path.join(input_path,'camera{:d}'.format(i)) for i in range(cam_num)]

    scene = Scene(ip, port)
    T_world_cams = scene.get_cam_matrixs()

    theta = np.pi
    l = np.array([0,0,1])
    q = np.array([np.cos(theta/2)]+(np.sin(theta/2) * l).tolist())
    r = quat2rot(q)
    cam_fix = r_t_to_mat(r, np.zeros(3))

    for i in range(cam_num):
        print("camera ",i)
        cam_path = cam_paths[i]
        T_world_ends = np.loadtxt(os.path.join(cam_path, 'world_end.txt')).reshape((-1,4,4))
        T_cam_objs = np.loadtxt(os.path.join(cam_path, 'cam_obj.txt')).reshape((-1,4,4))
        T_world_cam = np.matmul(T_world_cams[i], cam_fix)
        Ai = T_world_ends[0]
        Bi = T_cam_objs[0]

        # T(end->obj) = T(end->world) * T(world->cam) * T(cam->obj)
        T_end_obj = np.matmul(T_world_cam, Bi)
        T_end_obj = np.matmul(np.linalg.inv(Ai), T_end_obj)

        for j in range(T_world_ends.shape[0]):
            print("test ",j)

            # Aj = XBj(Bi)'(X)'Ai
            print("Aj_predict")
            Aj = T_world_ends[j]
            Bj = T_cam_objs[j]
            X = T_world_cam

            Aj_ = np.matmul(X, Bj)
            Aj_ = np.matmul(Aj_, np.linalg.inv(Bi))
            Aj_ = np.matmul(Aj_, np.linalg.inv(X))
            Aj_ = np.matmul(Aj_, Ai)

            thetaj, nj, tj = mat_to_theta_n_t(Aj)
            thetaj_, nj_, tj_ = mat_to_theta_n_t(Aj_)
            print(thetaj, nj, tj)
            print(thetaj_, nj_, tj_)
            print(
                "theta:{:6f}   n:{:6f}   t:{:6f}".format(
                    np.linalg.norm(thetaj-thetaj_), 
                    np.linalg.norm(nj-nj_)/np.linalg.norm(nj), 
                    np.linalg.norm(tj-tj_)/np.linalg.norm(tj)
                )
            )

            # T(end_i->base->camera->obj_i) = T(end_j->base->camera->obj_j)
            print("end -> obj")
            T_end_obj_ = np.matmul(T_world_cam, Bj)
            T_end_obj_ = np.matmul(np.linalg.inv(Aj), T_end_obj_)

            thetaj, nj, tj = mat_to_theta_n_t(T_end_obj)
            thetaj_, nj_, tj_ = mat_to_theta_n_t(T_end_obj_)
            print(thetaj, nj, tj)
            print(thetaj_, nj_, tj_)
            print(
                "theta:{:6f}   n:{:6f}   t:{:6f}".format(
                    np.linalg.norm(thetaj-thetaj_), 
                    np.linalg.norm(nj-nj_)/np.linalg.norm(nj), 
                    np.linalg.norm(tj-tj_)/np.linalg.norm(tj)
                )
            )
            print("--------------------------------------------")


if __name__ == "__main__":
    main()
    # test()