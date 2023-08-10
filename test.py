import torch
import cv2
import numpy as np
from lane_line_detection import calculate_control_signal
import sys
sys.path.append('/home/vupl/Desktop/tfs-auto-algorithms')
from YOLOPv2.demo import detect

def birdview_transform(img):
    """Apply bird-view transform to the image
    """
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img

path = '/home/vupl/Desktop/tfs-auto-algorithms/1/41.jpg'
model  = torch.jit.load('/home/vupl/Desktop/tfs-auto-algorithms/YOLOPv2/data/weights/yolopv2.pt').to('cuda:0').eval()
image = cv2.imread(path)
image = cv2.resize(image, (640, 480))
# image = birdview_transform(image)
img0, ll_seg_mask= detect(model=model, source=image)
imgs = calculate_control_signal(img0, ll_seg_mask)
cv2.imshow('img',imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()