#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy

from std_msgs.msg import Float64, Header, Int16MultiArray
from sensor_msgs.msg import Image
from time import sleep as s
from collections import deque
'''
1. cameraCallback으로 카메라 정보를 받아와서 저장하고
2. *_control로 카메라 정보 기반 value 결정
3. run 돌려서 serial_io로 정보 보내기

'''
class ControlModule:
    def __init__(self):
        self.camera = 0
        self.camera_list = deque([0]*20)
        self.direction = [0, 20, 0]
        self.speed = 200
        self.steer = 1550
        self.brake = 0
        self.msg = 0
        self.can_go = 1

        rospy.init_node("Control_module", anonymous=False)
        self.ctrlmsg = Int16MultiArray()
        self.pub = rospy.Publisher('/controller', Int16MultiArray, queue_size=1)
        self.sub = rospy.Subscriber('/camera', Int16MultiArray, self.cameraCallback) # msg from camera

        self.speed_check = False
        self.steer_check = False
        self.brake_check = False		
        self.gear_check = False

    def cameraCallback(self, msg):
        # -1: left
        # 0 : go
        # 1 : right
        self.msg = msg.data[0]
        if -5 <= msg.data[0] < 0:
            self.camera_list.appendleft(1)
            self.direction[2] += 1
        elif 0 <= msg.data[0] <= 5:
            self.camera_list.appendleft(-1)
            self.direction[0] += 1
        else:
            self.camera_list.appendleft(0)
            self.direction[1] += 1

        p = self.camera_list.pop()
        if p == -1:
            self.direction[0] -= 1
        elif p == 0:
            self.direction[1] -= 1
        else:
            self.direction[2] -= 1

        if msg.data[1] == 0:
            self.can_go = 0
        else:
            self.can_go = 1
    
    def speed_control(self):
        if self.brake:
            self.speed = int(0)
        else:
            self.speed = int(300)
        self.speed_check = True

    def steer_control(self):
        #print(self.direction)
        self.camera = self.direction.index(max(self.direction))
        # print(self.msg)
        # print(self.camera_list)
        # print(self.direction)
        # print(self.camera)
        if self.camera == 0:
            self.steer = int(1300) # positive - left
        elif self.camera == 2:
            self.steer = int(1800) # negitive - right
        elif self.camera == 1:
            self.steer = int(1550)
        # self.steer == int(1550)
        self.steer_check = True

    def brake_control(self):
        # if abs(self.camera) > 5:
        #     self.brake = int(1200)
        #     s(2.0)
        #     self.brake = int(0)
        # else:
        #     self.brake = int(0)
        if self.can_go == 1:
            self.brake = int(0)
        else:
            self.brake = int(1200)
        self.brake_check = True

    def gear_control(self):
        self.gear = int(0)
        self.gear_check = True

    def start(self):
        self.pub.publish(self.ctrlmsg)
        self.speed_check = False
        self.steer_check = False
        self.brake_check = False
        self.gear_check = False
        s(1.0)
        

    def run(self):
        self.steer_control()
        self.brake_control()
        self.speed_control()
        self.gear_control()
        if self.speed_check and self.steer_check and self.brake_check and self.gear_check:
            self.ctrlmsg.data = [self.gear, self.speed, self.steer, self.brake, self.camera]
            self.pub.publish(self.ctrlmsg)
            self.speed_check = False
            self.steer_check = False
            self.brake_check = False
            self.gear_check = False
        else:
            print("=== ERROR : control module ===")

if __name__ == "__main__":
    
    Con = ControlModule()
    Con.start()
    while not rospy.is_shutdown():
        Con.run()
        
