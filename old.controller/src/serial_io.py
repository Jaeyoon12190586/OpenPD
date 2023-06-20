#! /usr/bin/env python3
# -*- coding:utf-8 -*-


# import cv2
import serial
import numpy as np
from numpy import float32
import rospy
from std_msgs.msg import Int16MultiArray
from time import sleep as s

'''
ERP42 mini의 input은 speed, steer, brake, gear 총 네 가지입니다.

후진할 일 없으니 gear는 고려하지 않으셔도 되고, speed, steer, brake를 알고리즘을 통해 실시간으로 변경시켜주시면 됩니다.

모든 차량이 그렇지만, 차량이 완벽한 균형을 이루지 않기 때문에 steer을 중앙값으로 설정해도 차량이 완벽히 직진하지 않습니다.

https://wego-robotics.com/wego-erp42mini/   <<ERP42 mini 제원

'''

class SerialIO:
    def __init__(self):
        self.speed = 300 #value : 0~800 / real world speed : 0~6km/h 
        self.steer = 1300 #value : 1300~1800. middle = 1550 / real world steer는 기억이 안 나지만 쁠마 23도 였나.. 한번 측정해보시면 좋겠습니다
        # self.steer = 1550 #value : 1300~1800. middle = 1550 / real world steer는 기억이 안 나지만 쁠마 23도 였나.. 한번 측정해보시면 좋겠습니다
        self.brake = 0 #1200~1500 / 그냥 1200이나 1500 골라 쓰시면 됩니다. 차가 안 무거워서 괜찮티비
        self.gear = 0 #0 or 1 / 전후진

        self.serial1 = serial.Serial(port='/dev/erp42_mini', baudrate=115200)
       
        rospy.init_node("Serial", anonymous=False)
        self.sub = rospy.Subscriber("/controller", Int16MultiArray, self.controlCallback) # 인지부에서 controller 이름의 메시지 보내면 callback 실행
    
    def serRead(self):
        if self.serial1.readable(): # 시리얼 읽기
            res = self.serial1.readline()
            # print(res.decode()[:len(res) - 1])


    def serWrite(self): # 차량에 시리얼통신(값 인가)
        self.serial1.write(self.writeBuffer())
        # print(self.writeBuffer())

    #     self.gear = msg.data[0]
     # def controlCallback(self, msg):
   #     self.speed = msg.data[1]
    #     self.steer = msg.data[2]
    #     self.brake = msg.data[3]

    def controlCallback(self):
        self.gear = 0
        self.speed = 50
        self.steer = 1800
        self.brake = 0
    
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
        # self.serRead()

# def draw_point_on_frame(frame):
#     height, width, _ = frame.shape
#     center_x = width // 2
#     center_y = height // 2
#     cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
#     cv2.imshow("Camera", frame)


if __name__ == "__main__":
    sio = SerialIO()
    print("hello")

    while not rospy.is_shutdown():
        try:
            sio.run()
            s(0.01)
        except:
            pass