#! /usr/bin/env python3
import cv2
import numpy as np
from time import time

# 꼭 grayscale을 이용해야 할까??
# 왜 이용해야 할까?? - 1. 연산량이 적어짐
def RGB_to_GRAY(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def edge_Detect(img, low_threshold=50, high_threshold=210):
    return cv2.Canny(img, low_threshold, high_threshold)

def remove_Noise(img, kernel_size=(3, 3), sigmaX=1):
    return cv2.GaussianBlur(img, kernel_size, sigmaX)

def remove_Reflection(img):
    ### homomorphic filter는 gray scale image에 대해서 밖에 안 되므로
    ### YUV color space로 converting한 뒤 Y에 대해 연산을 진행
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    
    y = img_YUV[:,:,0]    
    
    rows = y.shape[0]    
    cols = y.shape[1]
    
    ### illumination elements와 reflectance elements를 분리하기 위해 log를 취함
    imgLog = np.log1p(np.array(y, dtype='float') / 255) # y값을 0~1사이로 조정한 뒤 log(x+1)
    
    ### frequency를 이미지로 나타내면 4분면에 대칭적으로 나타나므로 
    ### 4분면 중 하나에 이미지를 대응시키기 위해 row와 column을 2배씩 늘려줌
    M = 2*rows + 1
    N = 2*cols + 1
    
    ### gaussian mask 생성 sigma = 10
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬
    Xc = np.ceil(N/2) # 올림 연산
    Yc = np.ceil(M/2)
    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성
    
    ### low pass filter와 high pass filter 생성
    LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
    HPF = 1 - LPF
    
    ### LPF랑 HPF를 0이 가운데로 오도록iFFT함. 
    ### 사실 이 부분이 잘 이해가 안 가는데 plt로 이미지를 띄워보니 shuffling을 수행한 효과가 났음
    ### 에너지를 각 귀퉁이로 모아 줌
    LPF_shift = np.fft.ifftshift(LPF.copy())
    HPF_shift = np.fft.ifftshift(HPF.copy())
    
    ### Log를 씌운 이미지를 FFT해서 LPF와 HPF를 곱해 LF성분과 HF성분을 나눔
    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분

    ### 각 LF, HF 성분에 scaling factor를 곱해주어 조명값과 반사값을 조절함
    gamma1 = 0.3
    gamma2 = 1.5
    img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]
    
    ### 조정된 데이터를 이제 exp 연산을 통해 이미지로 만들어줌
    img_exp = np.expm1(img_adjusting) # exp(x) + 1
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화
    img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌
    
    ### 마지막으로 YUV에서 Y space를 filtering된 이미지로 교체해주고 RGB space로 converting
    img_YUV[:,:,0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return result

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

def remove_Outof_ROI_each(direction, img):
    # img.size() == 640 x 384 x 3, 원본-> resize비율 고려해야 함
    if len(img.shape)==3:   color = (255, 255, 255)
    elif len(img.shape)==2: color = 255

    mask = np.zeros_like(img)
    if direction == 'right':
        pts = np.array([[img.shape[1] // 2, img.shape[0]],          # left - bottom
                        [img.shape[1], img.shape[0]],               # right - bottom
                        [img.shape[1], img.shape[0] * 1 // 3],      # right
                        [img.shape[1] * 7 // 10, img.shape[0] // 2],# right - top
                        [img.shape[1] // 2, img.shape[0] // 2]])    # left - top
    elif direction == 'left':
        pts = np.array([[0,img.shape[0]],                           # left - bottom
                        [img.shape[1] // 2, img.shape[0]],          # right - bottom
                        [img.shape[1] // 2, img.shape[0] // 2],     # right - top
                        [img.shape[1] * 3 // 10, img.shape[0] // 2],# left - top
                        [0, img.shape[0] * 1 // 3]])                # left
    cv2.fillPoly(mask, np.int32([pts]), color)
    return cv2.bitwise_and(img, mask)

def hough_Transform(edge_img):
    return np.squeeze(cv2.HoughLinesP(edge_img, rho=1, theta=np.pi/180, threshold=25, minLineLength=80, maxLineGap=80))

def filtered_Line(hough_format):
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
    img = remove_Noise(img)
    img = cv2.resize(img, (640, 384))
    try:
        # rmR_img = remove_Reflection(img)
        edge_img = edge_Detect(img)
        cv2.imshow("edge", edge_img)
        cv2.waitKey(1)
        edge_img_in_ROI = remove_Outof_ROI(edge_img)
        

        x1y1x2y2 = hough_Transform(edge_img_in_ROI)
        lanes = filtered_Line(x1y1x2y2)
    
        if lanes is not None:
            left, right = lanes[0][len(lanes[0])//2], lanes[1][len(lanes[1])//2]
            
            left_slope = (left[4] - left[2]) / (left[3] - left[1])
            left_x1 = 0
            left_y1 = int(left_slope * (left_x1 - left[1]) + left[2])
            left_y2 = img.shape[0] * 2 // 3
            left_x2 = int(((left_y2 - left[2])/left_slope) + left[1])

            right_slope = (right[4] - right[2]) / (right[3] - right[1])
            right_x1 = img.shape[1] - 1
            right_y1 = int(right_slope * (right_x1 - right[1]) + right[2])
            right_y2 = img.shape[0] * 2 // 3
            right_x2 = int(((right_y2 - right[2])/right_slope) + right[1])

            middle = np.array([[(left_x1+right_x1)//2, (left_y1+right_y1)//2], 
                                [(left_x2+right_x2)//2, (left_y2+right_y2)//2]])
            cv2.line(img, middle[0], middle[1], (0, 0, 255), 3)
            # cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), (0, 0, 255), 3)
            # cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), (0, 0, 255), 3)
            if (middle[1][0] - middle[0][0]) == 0:
                return 0, img
            else:
                return ((middle[1][1] - middle[0][1])/(middle[1][0] - middle[0][0])), img
        else:
            return None, img
    except:
        return None, img

def calculate_Curvature(pts):
    '''
    input:
        [[pts1], [pts2], [pts3], ...]) : numpy 2D array 
    return:
        curvature : float
    '''
    x_t = np.gradient(pts[:, 0])
    y_t = np.gradient(pts[:, 1])

    vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])
    speed = np.sqrt(x_t * x_t, y_t * y_t)
    # tangent = np.array([1/speed] * 2).transpose() * vel

    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    return curvature*x_t

def use_ransac(img, ransac):
    '''
    input:
        image : numpy, ransac : scikitlearn object
    output:
        (curvature, lined img)
    '''
    img = remove_Noise(img)
    img = cv2.resize(img, [640, 384])
    cal_interval = 10
    try:
        rmR_img = remove_Reflection(img)
        gray_img = cv2.cvtColor(rmR_img, cv2.COLOR_BGR2GRAY)

        ret, thresh_img = cv2.threshold(gray_img % 170, 100,255, cv2.THRESH_BINARY)
        cv2.imshow("thresh_img", thresh_img)
        cv2.waitKey(1)
        if not ret:
            return (None, img)
        else:
            left_lane = remove_Outof_ROI_each("left", thresh_img)
            right_lane = remove_Outof_ROI_each("right", thresh_img)
            if len(left_lane[left_lane==255]) < 100 or len(right_lane[right_lane==255]) < 100:
                return (0, img)
            middle_row = np.arange(img.shape[0]-1, img.shape[0]//2, -cal_interval).reshape(-1, 1)
            middle_base = np.concatenate([middle_row ** 2, middle_row], axis=1)

            # row = x, col_num = y
            # 1. left lane points
            left_lane_x, left_lane_y = np.where(left_lane == 255)
            left_lane_x = left_lane_x.reshape(-1, 1)
            left_lane_X = np.concatenate([left_lane_x** 2, left_lane_x], axis=1)
            ransac.fit(left_lane_X, left_lane_y)
            ransac_left_lane_pts = ransac.predict(middle_base)

            # 2. right lane points
            right_lane_x, right_lane_y = np.where(right_lane == 255)
            right_lane_x = right_lane_x.reshape(-1, 1)
            right_lane_X = np.concatenate([right_lane_x** 2, right_lane_x], axis=1)
            ransac.fit(right_lane_X, right_lane_y)
            ransac_right_lane_pts = ransac.predict(middle_base)
            
            # 3. find middle lanes
            middle_col = ((ransac_right_lane_pts + ransac_left_lane_pts) // 2).reshape(-1, 1)
            middle_lane = np.concatenate([middle_col, middle_row], axis=1).astype(np.uint32)
            
            # 4. calculate curvature
            curvature = calculate_Curvature(middle_lane)
            
            # 5. draw
            middle_lane = middle_lane[middle_lane[:,0] < img.shape[1]]
            middle_lane = middle_lane[0 <= middle_lane[:,0]]
            for i in range(len(middle_lane)-1):
                cv2.line(img, middle_lane[i], middle_lane[i+1], [255, 0, 0])
            return (np.mean(middle_lane[:len(middle_lane)//2,0] - (img.shape[1]//2)), img)
    except:
        return (None, img)