import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scene import Scene
from tool import *
import os

def plt_line(ax, point1, point2, color='r', linewidth=1, linestyle=None):
    ax.plot([point1[0],point2[0]], [point1[1],point2[1]], [point1[2],point2[2]], c=color, linewidth=linewidth, linestyle=linestyle)

def plt_axis(ax, pp, linewidth=1, linestyle=None):
    pp = pp[:,:3]
    plt_line(ax, pp[0].ravel(), pp[1].ravel(), 'r', linestyle=linestyle,linewidth=1)
    plt_line(ax, pp[0].ravel(), pp[2].ravel(), 'g', linestyle=linestyle,linewidth=1)
    plt_line(ax, pp[0].ravel(), pp[3].ravel(), 'b', linestyle=linestyle,linewidth=1)

def main():
    ip = '127.0.0.1'
    port = 19997
    scene = Scene(ip, port)
    cam_num = 3

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axis('square')

    scale = 0.1
    axis = np.float32([[0,0,0],[2,0,0],[0,2,0],[0,0,2]]).reshape(-1,3) * scale * 0.5
    axis_aug = np.concatenate([axis, np.ones((4,1))], axis=1)

    # camera fix
    theta = np.pi
    l = np.array([0,0,1])
    q = np.array([np.cos(theta/2)]+(np.sin(theta/2) * l).tolist())
    r = quat2rot(q)
    cam_fix = r_t_to_mat(r, np.zeros(3))

    # world
    mat = np.eye(4)
    plt_axis(ax, np.matmul(mat, axis_aug.T).T)

    # camera
    T_world_cam = scene.get_cam_matrixs()
    for i in range(cam_num):
        plt_axis(ax, np.matmul(T_world_cam[i], axis_aug.T).T)
    
    # end
    for i in range(cam_num):
        T_world_ends = np.loadtxt(os.path.join(os.path.curdir,'calibration','camera{:d}'.format(i),'world_end.txt')).reshape((-1,4,4))
        print(T_world_ends.shape)
        for j in range(1):#base_ends.shape[0]//4):
            plt_axis(ax, np.matmul(T_world_ends[j], axis_aug.T).T, linestyle='dashed')
    
    # obj
    for i in range(cam_num):
        T_cam_objs = np.loadtxt(os.path.join(os.path.curdir,'calibration','camera{:d}'.format(i),'cam_obj.txt')).reshape((-1,4,4))
        print(T_cam_objs.shape)
        for j in range(1):#obj_cams.shape[0]//4):
            T_cam_obj = T_cam_objs[j]
            T_cam_obj = np.matmul(cam_fix, T_cam_obj)
            mat = np.matmul(T_world_cam[i], T_cam_obj)
            plt_axis(ax, np.matmul(mat, axis_aug.T).T, linestyle=':')

    # predict camera
    for i in range(cam_num):
        T_world_cam = np.loadtxt(os.path.join(os.path.curdir,'calibration','camera{:d}'.format(i),'world_cam.txt')).reshape((-1,4,4))
        T_world_cam = np.matmul(T_world_cam, np.linalg.inv(cam_fix))
        plt_axis(ax, np.matmul(T_world_cam, axis_aug.T).T, linestyle=':')

    plt.show()

if __name__ == "__main__":
    main()