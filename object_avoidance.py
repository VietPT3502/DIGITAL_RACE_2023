import numpy as np
from config import Config

def get_object_avoid_state(car_pred, current_midpoint):
    
    if car_pred is None or len(car_pred) == 0:
        return 0
    car_pred_yolo = []
    for bbox in car_pred:
        x1, y1, x2, y2, _,_ = bbox.detach().cpu().numpy()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        ls = [x_center, y_center, width, height]
        car_pred_yolo.append(ls)
    highest_center_y_bbox = sorted(car_pred_yolo, key=lambda arr: arr[1], reverse=True)[0]
    # highest_center_y_homo = np.array([highest_center_y_bbox[0], highest_center_y_bbox[1], 1.0])
    # print("-----------------------------------------------------------------------")
    # print(highest_center_y_homo)
    # highest_center_y_bev = np.dot(Config.M , highest_center_y_homo)
    # bev_x = highest_center_y_bev[0] / highest_center_y_bev[2]
    # print(bev_x)
    x = highest_center_y_bbox[0]
    w, h = highest_center_y_bbox[2], highest_center_y_bbox[3]
    # print(w * h)
    # print("...................................")
    if w * h < Config.CAR_AREA_THRESHOLD:
        return 0
    # print(x)
    # print(current_midpoint)
    # print(abs(current_midpoint - Config.MIDDLE_IMAGE_X))
    if (x < current_midpoint) and (abs(current_midpoint - Config.MIDDLE_IMAGE_X) < (Config.MIDPOINT_OFFSET - 10)):
        return -1
    elif (x > current_midpoint) and (abs(current_midpoint - Config.MIDDLE_IMAGE_X) < (Config.MIDPOINT_OFFSET - 10)):
        return 1
    else:
        return 0
def get_midpoint_offset(avoidance_state):
    if avoidance_state == -1:
        return +Config.MIDPOINT_OFFSET
    elif avoidance_state == 1:
        return -Config.MIDPOINT_OFFSET
    else:
        return 0
