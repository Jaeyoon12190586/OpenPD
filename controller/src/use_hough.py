#! /usr/bin/env python3
import cv2
import numpy as np

# 꼭 grayscale을 이용해야 할까??
# 왜 이용해야 할까?? - 1. 연산량이 적어짐
def RGB_to_GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def edge_Detect(img, low_threshold=30, high_threshold=210):
    return cv2.Canny(img, low_threshold, high_threshold)

def remove_Noise(img, kernel_size=(3, 3), sigmaX=1):
    return cv2.GaussianBlur(img, kernel_size, sigmaX)

def remove_Outof_ROI(img):
    # img.size() == 640 x 384 x 3, 원본-> resize비율 고려해야 함
    if len(img.shape)==3:   color = (255, 255, 255)
    elif len(img.shape)==2: color = 255

    mask = np.zeros_like(img)
    pts = np.array([[0,384], [640, 384], [450, 190], [190, 190]])
    cv2.fillPoly(mask, np.int32([pts]), color)
    return cv2.bitwise_and(img, mask)

def hough_Transform(edge_img):
    return np.squeeze(cv2.HoughLinesP(edge_img, rho=1, theta=np.pi/180, threshold=25, minLineLength=80, maxLineGap=80))

def filtered_line(hough_format):
    if not len(hough_format.shape): return

    left, right = [], []
    for x1, y1, x2, y2 in hough_format:
        slope = np.arctan2((y1-y2),(x1-x2))*180/np.pi
        if 95 < abs(slope) < 160:
            if slope > 0: left.append([slope, x1, y1, x2, y2])
            else: right.append([slope, x1, y1, x2, y2])
    if left and right:
        return [sorted(left, key=lambda x:x[0]), sorted(right, key=lambda x:x[0])]
    else:
        return 0

def use_hough(img):
    # return (direction, lined_img)
    img = cv2.resize(img, (640, 384))
    try:
        img = remove_Noise(img)
        edge_img = edge_Detect(img)
        edge_img_in_ROI = remove_Outof_ROI(edge_img)
        x1y1x2y2 = hough_Transform(edge_img_in_ROI)
        lanes = filtered_line(x1y1x2y2)
        if lanes is not None:
            left, right = lanes[0][len(lanes[0])//2], lanes[1][len(lanes[1])//2]
            cv2.line(img, (left[1], left[2]), (left[3], left[4]), (0, 0, 255), 3)
            cv2.line(img, (right[1], right[2]), (right[3], right[4]), (0, 0, 255), 3)
            return right[0]+left[0], img
        else:
            return 0, img
    except:
        return 0, img