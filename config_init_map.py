import numpy as np
import cv2
class Config:

    # Image width and height
    IMAGE_W = 640
    IMAGE_H = 480
    MIDDLE_IMAGE_X = IMAGE_W / 2
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    # Steer
    STEER_COEF = 5 / 6

    # When keep lane
    CURVE_THRESHOLD = 0.1
    THROTTLE_NO_CURVE = 0.45
    THROTTLE_CURVE = 0.1
    ROW_WHEN_KEEP_LANE = IMAGE_H / 3 * 2


    # When turn based on traffic sign
    TURN_TIME = 1.0
    THROTTLE_TURN = 0.25
    STEERING_TURN_COEF = 2.
    ROW_WHEN_TURN = IMAGE_H / 4 * 3

    # When Stop
    STOP_TIME = 4
    THROTTLE_STOP = 0

    # Traffic sign handle
    X_MASK_LEFT_TURN = int(IMAGE_W / 3 * 2) - 30
    X_MASK_RIGHT_TURN = int(IMAGE_W / 3) + 30
    Y_MASK_THRESHOLD = int(IMAGE_H / 5 * 2 - 80)
    X_MASK_LEFT_KEEP_LANE = int(IMAGE_W / 5)
    X_MASK_RIGHT_KEEP_LANE = int(IMAGE_W / 5 * 4)

    #Segmentation
    VISUALIZE_SEGMENTATION = True

    # Object detection
    IOU_THRESHOLD = 0.5
    CONF_THRESHOLD = 0.5
    VISUALIZE_DETECTION = True

    # Obstacle Avoidance
    CAR_AREA_THRESHOLD = 5000
    THROTTLE_OBSTACLE = 0.35
    STEERING_OBSTACLE_COEF = 0.6
    AVOIDANCE_TIME = 2.5
    MIDPOINT_OFFSET = 40


    
    
    