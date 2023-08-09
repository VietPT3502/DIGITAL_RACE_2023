import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from traffic_sign_handle import refine_driveable_area
from simple_pid import PID
from config import Config
from utilities.utils import show_seg_result
def birdview_transform(img):
    """Apply bird-view transform to the image
    """

    warped_img = cv2.warpPerspective(img, Config.M, (Config.IMAGE_W, Config.IMAGE_H)) # Image warping
    return warped_img

def to_cv2_format(mask):
    mask = mask*255
    # astype int8 to avoid overflow
    mask = mask.astype(np.uint8)
    return mask

def calculate_midpoint(img, mask, state):
    mask = to_cv2_format(mask)
    bev_image = birdview_transform(img)
    bev_mask = birdview_transform(mask)
    bev_mask = refine_driveable_area(state, bev_mask)
    # bev_mask = bev_mask /  255
        # Filter to get only largest white area in road mask
    if cv2.getVersionMajor() in [2, 4]:
        # OpenCV 2, OpenCV 4 case
        contours, hierarchy = cv2.findContours(bev_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # OpenCV 3 case
        _, contours, hierarchy = cv2.findContours(bev_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Choose largest contour
    best = -1
    maxsize = -1
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > maxsize :
            maxsize = cv2.contourArea(cnt)
            best = count
        count = count + 1

    bev_mask[:, :] = 0
    if best != -1:
        cv2.drawContours(bev_mask,[contours[best]], 0, 255, -1)
    bev_mask = bev_mask / 255
    img1 = show_seg_result(bev_image, (bev_mask), is_demo=True)
    if state == -1 or state == 1:
        row = Config.ROW_WHEN_TURN
    else:
        row = Config.ROW_WHEN_KEEP_LANE
    interested_row = bev_mask[int(row), :].reshape((-1,))
    white_pixels = np.argwhere(interested_row > 0)

    if white_pixels.size != 0:
        middle_pos = np.mean(white_pixels)
    else:
        middle_pos = Config.MIDDLE_IMAGE_X

    if middle_pos != middle_pos: # is NaN
        middle_pos = 0
    return middle_pos, img1


def calculate_steering_angle(img1,middle_pos, state):
    h = Config.IMAGE_H
    if state == -1 or state == 1:
        row = Config.ROW_WHEN_TURN
    else:
        row = Config.ROW_WHEN_KEEP_LANE
    distance_x = middle_pos - Config.MIDDLE_IMAGE_X
    distance_y = h - row
    if Config.VISUALIZE_DRIVEABLE == True:
        img1 = cv2.circle(img1, (int(middle_pos), int(row)), 7, (255, 255, 0), -1)
        cv2.imshow("driveable area", img1)
        cv2.waitKey(1)

    throttle = Config.THROTTLE_NO_CURVE
    steering_angle = math.atan(float(distance_x) / distance_y) * Config.STEER_COEF
    steering_angle *= 180 / math.pi 
    # print("--------------angle", steering_angle)
    steer_pid = PID(1.2, 1, 0.00, setpoint=0)
    steering_angle = steer_pid(-steering_angle)
    # print("--------------after pid angle", steering_angle)
    steering_angle = steering_angle / 180 * math.pi
    if steering_angle <= -Config.CURVE_THRESHOLD or steering_angle >= Config.CURVE_THRESHOLD:
        throttle = Config.THROTTLE_CURVE

    return throttle, steering_angle