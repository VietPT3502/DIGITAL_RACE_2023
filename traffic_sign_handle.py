import cv2
import os
import numpy as np
from config import Config
def get_state(sign_info):
    classes_encode = [0, 1, 2, 3, 4, 5]
    # classes = ['stop', 'left', 'right', 'straight', 'no_left', 'no_right']
    # states = ["keep_lane", "turn_left", "turn_right"]
    state = 0
    time = 0
    if len(sign_info) > 0:
        info = sign_info[0][5]
        
        if info == classes_encode[1] or info == classes_encode[5]:
            # print("-----------turn left--------------")
            state = -1
            time = Config.TURN_TIME
        elif info == classes_encode[2] or info == classes_encode[4]:
            # print("-----------turn right-------------")
            state = 1
            time =Config.TURN_TIME
        elif info == classes_encode[0]:
            x1, y1, x2, y2 = sign_info[0][0], sign_info[0][1], sign_info[0][2], sign_info[0][3]
            width = x2 - x1
            height = y2 - y1
            if width * height < Config.STOP_SIGN_AREA_THRESHOLD:
                return state, time
            # print("--------------stop---------------")
            state = 2
            time =Config.STOP_TIME
        elif info == classes_encode[3]:
            state = 0
            time = Config.STRAIGHT_TIME
    return state, time
 
def refine_driveable_area(state, da_seg_mask):
    mask = da_seg_mask

    if state == -1:
        # Set all values in the x-coordinate exceeding width / 3 * 2 to 0
        # print("left")
        mask[: , Config.X_MASK_LEFT_TURN:] = 0
        mask[: Config.Y_MASK_THRESHOLD, :] = 0
    elif state == 1:
        # print("right")
        # Set all values in the x-coordinate less than width / 3 to 0
        mask[:, : Config.X_MASK_RIGHT_TURN] = 0
        mask[: Config.Y_MASK_THRESHOLD, :] = 0
    elif state == 0:
        # print("keep lane")
        mask[:, Config.X_MASK_RIGHT_KEEP_LANE:] = 0
        mask[:, : Config.X_MASK_LEFT_KEEP_LANE] = 0
    return mask