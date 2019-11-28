import vrep
import time

class Gripper:
    def __init__(self, client_id, gripper_name):
        self.client_id = client_id
        self.name = gripper_name

        res, self.distance_handle = vrep.simxGetDistanceHandle(
            self.client_id, 
            "touches",
            operationMode=vrep.simx_opmode_blocking
        )
        res, self.distance_handle = vrep.simxGetDistanceHandle(
            self.client_id, 
            "touches",
            operationMode=vrep.simx_opmode_blocking
        )
    
    def close(self):
        vrep.simxSetIntegerSignal(
            self.client_id,
            signalName='RG2_open',
            signalValue=0,
            operationMode=vrep.simx_opmode_blocking
        )
        self.wait_until_stop()
    
    def open(self):
        vrep.simxSetIntegerSignal(
            self.client_id,
            signalName='RG2_open',
            signalValue=1,
            operationMode=vrep.simx_opmode_blocking
        )
        self.wait_until_stop()
    
    def check_in(self, threshold = 0.001):
        _, distance = vrep.simxReadDistance(
            self.client_id,
            self.distance_handle,
            operationMode=vrep.simx_opmode_blocking
        )
        return distance > threshold

    def reset(self):
        self.open()

    def wait_until_stop(self, threshold=0.001):
        while True:
            res, distance_1 = vrep.simxReadDistance(
                self.client_id,
                self.distance_handle,
                operationMode=vrep.simx_opmode_blocking
            )
            
            time.sleep(1)
            res, distance_2 = vrep.simxReadDistance(
                self.client_id,
                self.distance_handle,
                operationMode=vrep.simx_opmode_blocking
            )
            
            diff = abs(distance_2 - distance_1)
            if diff < threshold:
                return