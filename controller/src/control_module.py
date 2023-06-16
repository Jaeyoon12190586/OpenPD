#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy

from std_msgs.msg import Float64, Header, Int16MultiArray
from sensor_msgs.msg import Image

'''
1. cameraCallback으로 카메라 정보를 받아와서 저장하고
2. *_control로 카메라 정보 기반 value 결정
3. run 돌려서 serial_io로 정보 보내기

'''
class ControlModule:
    def __init__(self):
        self.speed = 0
        self.steer = 1550
        self.brake = 1200

        rospy.init_node("Control_module", anonymous=False)
        self.ctrlmsg = Int16MultiArray()
        self.pub = rospy.Publisher('/controller', Int16MultiArray, queue_size=1)
        self.sub = rospy.Subscriber('/camera', Float64, self.cameraCallback) # msg from camera

        self.speed_check = False
        self.steer_check = False
        self.brake_check = False		
        self.gear_check = False

    def cameraCallback(self, msg):
        self.camera = msg.data

    def speed_control(self):
        self.speed = int(300)
        self.speed_check = True

    def steer_control(self):
        
        self.steer = int(1500)
        self.steer_check = True


    def brake_control(self):
        
        self.brake = int(0)
        self.brake_check = True


    def gear_control(self):
        
        self.gear = int(0)
        self.gear_check = True


    def run(self):
        if self.speed_check and self.steer_check and self.brake_check and self.gear_check:
            self.ctrlmsg.data = [self.gear, self.speed, self.steer, self.brake]
            self.pub.publish(self.ctrlmsg)
            self.speed_check = False
            self.steer_check = False
            self.brake_check = False
            self.gear_check = False
        else:
            print("=== ERROR : control module ===")
if __name__ == "__main__":
    Con = ControlModule()
    while not rospy.is_shutdown():
        Con.run()
