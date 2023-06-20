#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import serial
import numpy as np
from numpy import float32
import rospy
from std_msgs.msg import Int16MultiArray, Float64
from time import sleep as s

'''
ERP42 mini input:
speed, steer, brake, gear

This is true of all vehicles, but the vehicle is not in perfect balance. So although you set the steer to the median, the vehicle will not allow to go completely straight.
'''

class SerialIO:
    def __init__(self):
        self.speed = 0 #value : 0~800 / real world speed : 0~6km/h 
        self.steer = 1550 #value : 1300~1800. middle = 1550
        # self.steer = 1550 #value : 1300~1800. middle = 1550
        self.brake = 0 #1200~1500 / you choose 1200 or 1500
        self.gear = 0 #0(전진) or 1(후진)
        self.camera = 0

        self.serial1 = serial.Serial(port='/dev/erp42_mini', baudrate=115200)
       
        rospy.init_node("Serial", anonymous=False)
        self.sub = rospy.Subscriber("/controller", Int16MultiArray, self.controlCallback) 
        
    def serRead(self):
        if self.serial1.readable(): # read serial
            res = self.serial1.readline()
            # print(res.decode()[:len(res) - 1])


    def serWrite(self): # 차량에 시리얼통신(값 인가)
        self.serial1.write(self.writeBuffer())
        # print(self.writeBuffer())

    def controlCallback(self,msg):
        self.gear = msg.data[0]
        self.speed = msg.data[1]
        self.steer = msg.data[2]
        self.brake = msg.data[3]
        self.camera = msg.data[4]

    def writeBuffer(self): # 시리얼통신 프로토콜에 맞춰 패킷 생성
        packet = []
        gear = self.gear

        # 타입 맞춰주기
        speed = np.uint16(self.speed)
        steer = np.uint16(self.steer)
        brake = np.uint16(self.brake)


        # 바이트 분할 작업
        speed_L = speed & 0xff
        speed_H = speed >> 8

        steer_L = steer & 0xFF
        steer_H = steer >> 8

        brake_L = brake & 0xff
        brake_H = brake >> 8

        # CLC 계산을 위한 바이트 총합 구하기
        sum_c = gear + speed_L + speed_H + steer_L \
                    + steer_H + brake_L + brake_H + 13 + 10

        # CLC는 1 Byte
        clc = np.uint8(~sum_c)

        packet.append(0x53)
        packet.append(0x54)
        packet.append(0x58)
        packet.append(gear)
        packet.append(speed_L)
        packet.append(speed_H)
        packet.append(steer_L)
        packet.append(steer_H)
        packet.append(brake_L)
        packet.append(brake_H)
        packet.append(0x00)
        packet.append(0x0D)
        packet.append(0x0A)
        packet.append(clc)

        return packet       
    
    def run(self):
        self.serWrite() 

if __name__ == "__main__":
    sio = SerialIO()
    s(5.0  )
    print("hello")

    while not rospy.is_shutdown():
        try:
            sio.run()
            # print(sio.steer, sio.camera, sio.brake)
            s(0.01)
        except:
            pass
