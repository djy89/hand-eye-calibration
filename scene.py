import vrep
import numpy as np
import scene_object
import scene_ur5
import time
from math import sqrt, pi
from tool import quat2rot, r_t_to_mat

class Scene:
    def __init__(self, ip, port):
        vrep.simxFinish(-1)
        self.client_id = vrep.simxStart(
            ip,
            port,
            waitUntilConnected=True,
            doNotReconnectOnceDisconnected=True,
            timeOutInMs=5000,
            commThreadCycleInMs=5
        )
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_blocking)

        self.object = scene_object.Object(self.client_id, "obj17")
        self.ur5 = scene_ur5.UR5(self.client_id, "UR5")
        self.camera_handles = [
            vrep.simxGetObjectHandle(
                self.client_id,
                'realsense{:d}'.format(i+1),
                operationMode=vrep.simx_opmode_blocking
            )[1]
            for i in range(3)
        ]

        self.resolution, self.background_rgb = self.get_image()
        _, self.background_depth = self.get_depth()
        # self.center = [0.4, 0.4]

    def finish(self):
        vrep.simxStopSimulation(self.client_id,operationMode=vrep.simx_opmode_blocking)
        vrep.simxFinish(self.client_id)
    
    def get_image(self):
        # camera1 1
        res, resolution, image1 = vrep.simxGetVisionSensorImage(
            self.client_id,
            self.camera_handles[0],
            options=0,
            operationMode=vrep.simx_opmode_blocking
        )
        image1 = np.array(image1, dtype=np.uint8)
        image1 = np.reshape(image1, [resolution[1], resolution[0], 3])[::-1, ...]

        # camera1 2
        res, resolution, image2 = vrep.simxGetVisionSensorImage(
            self.client_id,
            self.camera_handles[1],
            options=0,
            operationMode=vrep.simx_opmode_blocking
        )
        image2 = np.array(image2, dtype=np.uint8)
        image2 = np.reshape(image2, [resolution[1], resolution[0], 3])[::-1, ...]

        # camera1 3
        res, resolution, image3 = vrep.simxGetVisionSensorImage(
            self.client_id,
            self.camera_handles[2],
            options=0,
            operationMode=vrep.simx_opmode_blocking
        )
        image3 = np.array(image3, dtype=np.uint8)
        image3 = np.reshape(image3, [resolution[1], resolution[0], 3])[::-1, ...]

        resolution = resolution[::-1]
        return resolution, [image1,image2,image3]

    def get_depth(self):
        # camera 1
        res, resolution, depth1 = vrep.simxGetVisionSensorDepthBuffer(
            self.client_id,
            self.camera_handles[0],
            operationMode=vrep.simx_opmode_blocking
        )
        depth1 = np.array(depth1)
        depth1 = np.reshape(depth1,[resolution[1], resolution[0]])[::-1, ...]

        # camera 1
        res, resolution, depth2 = vrep.simxGetVisionSensorDepthBuffer(
            self.client_id,
            self.camera_handles[1],
            operationMode=vrep.simx_opmode_blocking
        )
        depth2 = np.array(depth2)
        depth2 = np.reshape(depth2,[resolution[1], resolution[0]])[::-1, ...]

        # camera 1
        res, resolution, depth3 = vrep.simxGetVisionSensorDepthBuffer(
            self.client_id,
            self.camera_handles[2],
            operationMode=vrep.simx_opmode_blocking
        )
        depth3 = np.array(depth3)
        depth3 = np.reshape(depth3,[resolution[1], resolution[0]])[::-1, ...]

        resolution = resolution[::-1]
        return resolution, [depth1, depth2, depth3]

    def get_cam_matrixs(self):
        mats = []
        for i in range(3):
            res, position = vrep.simxGetObjectPosition(
                self.client_id,
                self.camera_handles[i],
                relativeToObjectHandle=-1,
                operationMode=vrep.simx_opmode_blocking
            )
            t = np.array(position)
            res, quaternion = vrep.simxGetObjectQuaternion(
                self.client_id,
                self.camera_handles[i],
                relativeToObjectHandle=-1,
                operationMode=vrep.simx_opmode_blocking
            )
            q = np.array(quaternion)[[3,0,1,2]]
            r = quat2rot(q)
            mats.append(r_t_to_mat(r,t))
        return mats