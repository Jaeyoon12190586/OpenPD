#! /usr/bin/env python3
import cv2
import numpy as np
from numpy import float32
from time import sleep
import rospy
from std_msgs.msg import Float64
from use_hough import use_hough

class Camera:
    def __init__(self):

        rospy.init_node("Camera", anonymous=False)
        self.pub = rospy.Publisher('/camera', Float64, queue_size=1)
        self.cap = cv2.VideoCapture(0)
        # self.height
        # self.width
        # self.direction

    def draw_info_on_frame(self, frame):
        direction, lined_img = use_hough(frame)
        cv2.circle(lined_img, (lined_img.shape[1]//2, lined_img.shape[0]//2), 5, (0, 0, 255), -1)
        # print(direction)
        cv2.imshow("Camera", lined_img)
        cv2.waitKey(1)
        return direction, lined_img

    def run(self):
        ret, frame = self.cap.read()
        if ret:
            d, _ = self.draw_info_on_frame(frame)
        self.pub.publish(d)
        


if __name__ == "__main__":
    camera = Camera()
    while not rospy.is_shutdown():
        camera.run()
        
    # cap = cv2.VideoCapture(0)
    # print("hello")

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     draw_info_on_frame(frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    #     sleep(0.1)

    # cap.release()
    # cv2.destroyAllWindows()