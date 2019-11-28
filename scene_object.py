import vrep
import time
from functools import reduce
from math import sqrt

class Object:
    def __init__(self, client_id, name):
        self.client_id = client_id
        self.name = name
        _, self.handle = vrep.simxGetObjectHandle(
            self.client_id,
            self.name,
            operationMode=vrep.simx_opmode_blocking
        )
    
    def move_to(self, pose):
        res = vrep.simxCallScriptFunction(
            clientID=self.client_id,
            scriptDescription="UR5",
            options=vrep.sim_scripttype_childscript,
            operationMode=vrep.simx_opmode_blocking,
            functionName='pyMoveToPosition',
            inputInts=[self.handle],
            inputFloats=pose,
            inputStrings=[],
            inputBuffer=''
        )
        print(res)
        self.wait_until_stop(self.handle)

    def get_position(self):
        _, position = vrep.simxGetObjectPosition(
            self.client_id,
            self.handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )
        return position
    
    def set_position(self, position):
        vrep.simxSetObjectPosition(
            self.client_id,
            self.handle,
            relativeToObjectHandle=-1,
            position=position,
            operationMode=vrep.simx_opmode_oneshot
        )
        self.wait_until_stop(self.handle)
    
    def get_quaternion(self):
        _, quaternion = vrep.simxGetObjectQuaternion(
            self.client_id,
            self.handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )
        return quaternion
    
    def set_quaternion(self, quaternion):
        vrep.simxSetObjectQuaternion(
            self.client_id,
            self.handle,
            relativeToObjectHandle=-1,
            quaternion=quaternion,
            operationMode=vrep.simx_opmode_oneshot
        )
        self.wait_until_stop(self.handle)

    def get_orientation(self):
        _, orientation = vrep.simxGetObjectOrientation(
            self.client_id,
            self.handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )
        return orientation

    def wait_until_stop(self, handle, threshold=0.01):
        while True:
            _, pos1 = vrep.simxGetObjectPosition(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            _, quat1 = vrep.simxGetObjectQuaternion(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            time.sleep(0.7)
            _, pos2 = vrep.simxGetObjectPosition(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            _, quat2 = vrep.simxGetObjectQuaternion(self.client_id, handle, -1, vrep.simx_opmode_blocking)
            pose1 = pos1 + quat1
            pose2 = pos2 + quat2
            theta = 0.5 * sqrt(reduce(lambda x, y: x + y, map(lambda x, y: (x - y) ** 2, pose1, pose2)))
            if theta < threshold:
                return