#! /usr/bin/env python3
import cv2
import numpy as np
from numpy import float32
from time import sleep
import rospy
from std_msgs.msg import Float64, Int16MultiArray
from util import use_hough, use_ransac
from sklearn.linear_model import RANSACRegressor
from time import time


class Camera:
    def __init__(self):

        rospy.init_node("Camera", anonymous=False)
        self.pub = rospy.Publisher('/camera', Int16MultiArray, queue_size=1)
        self.cap = cv2.VideoCapture(0)
        self.ransac = RANSACRegressor()
        self.pubmsg = Int16MultiArray()
        self.vis = 0
        # self.height
        # self.width
        # self.direction

    def traffic_is_green(self, frame):
        traffic_red = frame[100:110, : ,:]
        boundary = np.zeros_like(traffic_red)
        self.vis = np.vstack([self.vis, boundary])
        self.vis = np.vstack([self.vis, traffic_red])
        B = traffic_red[:, :, 0] < 120
        G = traffic_red[:, :, 1] < 50
        R = traffic_red[:, :, 2] > 220

        if np.any(traffic_red[B & G & R]):
            return 0
        else:
            return 1
        
    def draw_info_on_frame(self, frame):

        lined_img , direction = frame, 0
        direction, lined_img = use_hough(frame)
        # direction, lined_img = use_ransac(frame, self.ransac)
        if direction is None:
            direction = 0
        cv2.circle(lined_img, (lined_img.shape[1]//2, lined_img.shape[0]//2), 5, (0, 0, 255), -1)
        self.vis = lined_img
        return direction, lined_img

    def run(self):
        ret, frame = self.cap.read()
        if ret:
            d, _ = self.draw_info_on_frame(frame)
            is_green = self.traffic_is_green(frame)
            self.pubmsg.data = [int(d), int(is_green)]
            self.pub.publish(self.pubmsg)
            print(d)
            cv2.imshow("Camera", self.vis)
            cv2.waitKey(1)
        else:
            self.pubmsg.data = [0, 1]
            self.pub.publish(self.pubmsg)

if __name__ == "__main__":
    camera = Camera()
    rospy.Rate(30)
    while not rospy.is_shutdown():
        camera.run()