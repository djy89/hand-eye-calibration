from functools import partial, reduce
from math import sqrt
import time
import vrep
import numpy as np
import scene_gripper
from tool import quat2rot, r_t_to_mat

class UR5:
    def __init__(self, client_id, name):
        self.client_id = client_id

        # RG2 gripper
        self.gripper = scene_gripper.Gripper(self.client_id, gripper_name="RG2")

        # initialization
        self.func = partial(
            vrep.simxCallScriptFunction,
            clientID=self.client_id,
            scriptDescription="UR5",
            options=vrep.sim_scripttype_childscript,
            operationMode=vrep.simx_opmode_blocking
        )
        self.initialization()

        # UR5 related
        self.joint_handles = [
            vrep.simxGetObjectHandle(
                self.client_id, 
                "UR5_joint{}".format(i+1),
                operationMode=vrep.simx_opmode_oneshot_wait
            )[1]
            for i in range(6)
        ]
        res, self.end_handle = vrep.simxGetObjectHandle(
            self.client_id, 
            "UR5_ik_tip",
            operationMode=vrep.simx_opmode_blocking
        )
        res, self.obj_handle = vrep.simxGetObjectHandle(
            self.client_id, 
            "cam_calibration",
            operationMode=vrep.simx_opmode_blocking
        )
        res, self.target_handle = vrep.simxGetObjectHandle(
            self.client_id, 
            "UR5_ik_target",
            operationMode=vrep.simx_opmode_blocking
        )
        _, self.init_position = vrep.simxGetObjectPosition(
            self.client_id,
            self.target_handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )
        _, self.init_quaternion = vrep.simxGetObjectQuaternion(
            self.client_id,
            self.target_handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )

    def initialization(self):
        self.func(
            functionName='pyInit',
            inputInts=[],
            inputFloats=[],
            inputStrings=[],
            inputBuffer=''
        )

    def grasp(self, position, theta):
        # horizontal move
        hori_pos = position.copy()
        hori_pos[-1] = self.init_position[-1]
        horizontal_pose = hori_pos + self.init_quaternion
        self.move_to_object_position(horizontal_pose)
       
        # rotate
        angle = self.get_joint_angle(self.joint_handles[0])
        self.move_to_joint_positions([self.joint_handles[-1]],[angle])
        self.move_to_joint_positions([self.joint_handles[-1]],[theta])

        # vertical move
        vertical_pose = position + self.init_quaternion
        self.move_to_object_position(vertical_pose)

        # grasp
        self.gripper.close()
        return self.gripper.check_in()

    def rotate(self, theta):
        self.move_to_joint_positions([self.joint_handles[-1]],[theta])

    def get_end_matrix(self):
        _, quaternion = vrep.simxGetObjectQuaternion(
            self.client_id,
            self.end_handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )
        quaternion = np.array(quaternion)
        quaternion = quaternion[[3,0,1,2]]
        r = quat2rot(quaternion)
        _, position = vrep.simxGetObjectPosition(
            self.client_id,
            self.end_handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )
        t = np.array(position)
        return r_t_to_mat(r,t)



    def move_to_joint_positions(self, handles, angles):
        self.func(
            functionName='pyMoveToJointPositions',
            inputInts=handles,
            inputFloats=angles,
            inputStrings=[],
            inputBuffer=''
        )
        for handle in handles:
            self.wait_until_stop(handle)

    def move_to_object_position(self, pose):
        self.func(
            functionName='pyMoveToPosition',
            inputInts=[self.target_handle],
            inputFloats=pose,
            inputStrings=[],
            inputBuffer=''
        )
        self.wait_until_stop(self.target_handle)

    def reset(self):
        self.gripper.open()

        # vertical move
        _, current_position = vrep.simxGetObjectPosition(
            self.client_id,
            self.target_handle,
            relativeToObjectHandle=-1,
            operationMode=vrep.simx_opmode_blocking
        )
        ver_pos = current_position
        ver_pos[-1] = self.init_position[-1]
        vertical_move = ver_pos + self.init_quaternion
        self.move_to_object_position(vertical_move)

        # horizontal move
        horizontal_move = self.init_position + self.init_quaternion
        self.move_to_object_position(horizontal_move)

        self.move_to_joint_positions([self.joint_handles[-1]],[0.])
    

    def get_joint_angle(self, handle):
        res, angle = vrep.simxGetJointPosition(
            self.client_id,
            handle,
            operationMode=vrep.simx_opmode_blocking
        )
        return angle

    def wait_until_stop(self, handle, threshold=0.01):
        """
        Wait until the operation finishes.
        This is a delay function called in order to make sure that
        the operation executed has been completed.
        :param handle: An int.Handle of the object.
        :param threshold: A float. The object position threshold.
        If the object positions difference between two time steps is smaller than the threshold,
        the execution completes, otherwise the loop continues.
        :return: None
        """
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