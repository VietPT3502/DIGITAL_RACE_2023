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
    THROTTLE_NO_CURVE = 0.3
    THROTTLE_CURVE = 0.15
    ROW_WHEN_KEEP_LANE = IMAGE_H / 3 * 2


    # When turn based on traffic sign
    TURN_TIME = 2.0
    THROTTLE_TURN = 0.3
    STEERING_TURN_COEF = 2.
    ROW_WHEN_TURN = IMAGE_H / 4 * 3 + 20

    # When Stop
    STOP_TIME = 3
    THROTTLE_STOP = 0.0

    #When Straight
    STRAIGHT_TIME = 2.0

    # Traffic sign handle
    X_MASK_LEFT_TURN = int(IMAGE_W / 3 * 2) - 30
    X_MASK_RIGHT_TURN = int(IMAGE_W / 3) + 30
    Y_MASK_THRESHOLD = int(IMAGE_H / 5 * 2 - 80)
    X_MASK_LEFT_KEEP_LANE = int(IMAGE_W / 5 )
    X_MASK_RIGHT_KEEP_LANE = int(IMAGE_W / 5 * 4) 
    STOP_SIGN_AREA_THRESHOLD = 1000
    #Segmentation
    VISUALIZE_SEGMENTATION = False
    VISUALIZE_DRIVEABLE = True
    # Object detection
    IOU_THRESHOLD = 0.6
    CONF_THRESHOLD = 0.5
    VISUALIZE_DETECTION = True

    # Obstacle Avoidance
    CAR_AREA_THRESHOLD = 3000
    THROTTLE_OBSTACLE = 0.3
    STEERING_OBSTACLE_COEF = 0.6
    AVOIDANCE_TIME = 0.5
    MIDPOINT_OFFSET = 40


    
    
    