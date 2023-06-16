#! /usr/bin/env python3
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

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
    pts = np.array([[0,img.shape[0]],                           # left - bottom
                    [img.shape[1], img.shape[0]],               # right - bottom
                    [img.shape[1], img.shape[0] * 3 // 4],      # right
                    [img.shape[1] * 7 // 10, img.shape[0] // 2],# right - top
                    [img.shape[1] * 3 // 10, img.shape[0] // 2],# left - top
                    [0, img.shape[0] * 3 // 4]])                # left
    
    cv2.fillPoly(mask, np.int32([pts]), color)
    return cv2.bitwise_and(img, mask)

def lane_ROI(direction, img):
    # img.size() == 640 x 384 x 3, 원본-> resize비율 고려해야 함
    if len(img.shape)==3:   color = (255, 255, 255)
    elif len(img.shape)==2: color = 255

    mask = np.zeros_like(img)
    if direction == 'right':
        pts = np.array([[img.shape[1] // 2, img.shape[0]],          # left - bottom
                        [img.shape[1], img.shape[0]],               # right - bottom
                        [img.shape[1], img.shape[0] * 3 // 4],      # right
                        [img.shape[1] * 7 // 10, img.shape[0] // 2],# right - top
                        [img.shape[1] // 2, img.shape[0] // 2]])    # left - top
    elif direction == 'left':
        pts = np.array([[0,img.shape[0]],                           # left - bottom
                        [img.shape[1] // 2, img.shape[0]],          # right - bottom
                        [img.shape[1] // 2, img.shape[0] // 2],     # right - top
                        [img.shape[1] * 3 // 10, img.shape[0] // 2],# left - top
                        [0, img.shape[0] * 3 // 4]])                # left
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
    
def use_ransac(img, ransac):
    img = remove_Noise(img)
    img = cv2.resize(img, [640, 384])
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh_img = cv2.threshold(gray_img, 190,255, cv2.THRESH_BINARY)
    if ret:
        left_lane = lane_ROI("left", thresh_img)
        right_lane = lane_ROI("right", thresh_img)
        if len(left_lane[left_lane==255]) < 100 or len(right_lane[right_lane==255]) < 100:
            return 0, img
        middle_row = np.arange(img.shape[0]-1, img.shape[0]//2, -1).reshape(-1, 1)
        middle_base = np.concatenate([middle_row ** 2, middle_row], axis=1)

        # row = x, col_num = y
        # 1. left lane points
        left_lane_x, left_lane_y = np.where(left_lane == 255)
        left_lane_x = left_lane_x.reshape(-1, 1)
        left_lane_X = np.concatenate([left_lane_x** 2, left_lane_x], axis=1)
        ransac.fit(left_lane_X, left_lane_y)
        ransac_left_lane_pts = ransac.predict(middle_base)
        # ransac_left_lane_pts = np.concatenate([left_lane_x, ransac.predict(left_lane_X).reshape(-1, 1)], axis=1)

        # 2. right lane points
        right_lane_x, right_lane_y = np.where(right_lane == 255)
        right_lane_x = right_lane_x.reshape(-1, 1)
        right_lane_X = np.concatenate([right_lane_x** 2, right_lane_x], axis=1)
        ransac.fit(right_lane_X, right_lane_y)
        ransac_right_lane_pts = ransac.predict(middle_base)
        # ransac_right_lane_pts = np.concatenate([right_lane_x, ransac.predict(right_lane_X).reshape(-1, 1)], axis=1)
        
        # 3. find middle lanes
        middle_col = ((ransac_right_lane_pts + ransac_left_lane_pts) // 2).reshape(-1, 1)
        middle_lane = np.concatenate([middle_col, middle_row], axis=1).astype(np.uint32)
        
        # 4. draw
        for i in range(len(middle_lane)-1):
            try:
                cv2.line(img, middle_lane[i], middle_lane[i+1], [255, 0, 0])
            except:
                pass
        return ((img.shape[1] // 2) - np.mean(middle_col), img)