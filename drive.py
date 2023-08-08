import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue
import sys
import cv2
import numpy as np
import websockets
from PIL import Image
import torch

import multiprocessing
sys.path.insert(0, "/home/vietpt/vietpt/code/race/DIGITAL_RACE_2023/yolov5")
# Add this line at the very beginning of your script before importing any other modules
multiprocessing.set_start_method('spawn', True)
from unet.model import make_model
from unet.predict import predict
from traffic_sign_handle import refine_driveable_area, get_state
from driveable_area_handle import calculate_steering_angle, calculate_midpoint
from traffic_sign_detection import detect_traffic_signs
from object_avoidance import get_midpoint_offset, get_object_avoid_state
from config import Config
# model  = torch.jit.load('/home/vupl/Desktop/tfs-auto-algorithms/YOLOPv2/data/weights/yolopv2.pt')
# print('load model done')

# Initalize traffic sign classifier
checkpoint = "/home/vietpt/vietpt/code/race/DIGITAL_RACE_2023/weights/best_yolov5s_224x224_50ep.pt"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dnn = False
data = "/home/vietpt/vietpt/code/race/DIGITAL_RACE_2023/trafficsign.yaml"
half = False

from models.common import DetectMultiBackend
detection_model = DetectMultiBackend(
    checkpoint,
    device=device,
    dnn=dnn,
    data=data,
    fp16=half
    )

segment_model, criterion, optimizer, scheduler = make_model()
segment_model.load_state_dict(torch.load('/home/vietpt/vietpt/code/race/DIGITAL_RACE_2023/unet/best_loss3.pth'))
segment_model.eval()  # Set the model to evaluation mode


# Global queue to save current image
# We need to run the sign classification model in a separate process
# Use this queue as an intermediate place to exchange images
g_image1_queue = Queue(maxsize=1)
g_image2_queue = Queue(maxsize=1)
g_sign_queue = Queue(maxsize=1)
g_car_queue = Queue(maxsize=1)
g_daseg_queue = Queue(maxsize=1)

sign_state_dict = {}
object_avoid_state_dict = {}
# Function to run sign classification model continuously
# We will start a new process for this
def process_traffic_sign_loop(g_image_queue, g_sign_queue, g_car_queue):
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        image = g_image_queue.get()
        
        # print("process_detection_loop")
        # Prepare visualization image
        # Detect traffic signs
        traffic_sign_pred, car_pred, image = detect_traffic_signs(
            image,
            detection_model,
            imgsz=(Config.IMAGE_W, Config.IMAGE_H),
            conf_thres=Config.CONF_THRESHOLD,
            iou_thres=Config.IOU_THRESHOLD,
            visualize=Config.VISUALIZE_DETECTION)
        # detected = show_detected_traffic_sign(
        #     image,
        #     boxes,
        #     delay=1,
        #     new_size=(new_h, new_w),
        #     classes=["stop", "left", "right", "straight", "no_left", "no_right"])
        if image is not None:
            cv2.imshow('detection', image)
            cv2.waitKey(1)
        if not g_sign_queue.full():
            if len(traffic_sign_pred) > 0:
                g_sign_queue.put(traffic_sign_pred[0])
            else:
                g_sign_queue.put(traffic_sign_pred)

        if not g_car_queue.full():
            if len(car_pred) > 0:
                g_car_queue.put(car_pred[0])
            else:
                g_car_queue.put(car_pred)



def process_segment_loop(g_image2_queue, g_daseg_queue):
    while True:
        if not g_image2_queue.empty():
            image = g_image2_queue.get()
            # Prepare visualization image
            # Detect traffic signs
            da_seg_mask = predict(model=segment_model, image=image)
            if Config.VISUALIZE_SEGMENTATION:
                cv2.imshow("mask", da_seg_mask)
                cv2.waitKey(1)
            if not g_daseg_queue.full():
                g_daseg_queue.put(da_seg_mask)
            torch.cuda.empty_cache()

async def process_image(websocket, path):
    async for message in websocket:
        # Get image from simulation
        import time
        start = time.time()
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (Config.IMAGE_W, Config.IMAGE_H))

        # Update image to g_image_queue - used to run sign detection
        if not g_image1_queue.full():
            g_image1_queue.put(image)
        if not g_image2_queue.full():
            g_image2_queue.put(image)
        sign_state = 0
        sign_info = None

        if not g_sign_queue.empty():
            sign_info = g_sign_queue.get()

            if sign_info is not None:
                sign_state = get_state(sign_info)

                if sign_state == -1 or sign_state == 1:
                    # Store the detected sign state and its expiry time in the dictionary
                    sign_state_dict[sign_state] = time.time() + Config.TURN_TIME # Set the expiry time to 1 seconds
                elif sign_state == 2:
                    sign_state_dict[sign_state] = time.time() + Config.STOP_TIME
                else:
                    sign_state_dict[sign_state] = time.time()  # Set the expiry time to 1 seconds

        # Check if any detected sign's state has expired and remove it from the dictionary
        # current_time = time.time()
        # expired_signs = [sign for sign, expiry_time in sign_state_dict.items() if current_time > expiry_time]
        # for sign in expired_signs:
        #     del sign_state_dict[sign]

        # If there are still signs in the dictionary, update the state accordingly
        if sign_state_dict:
            sign_state = max(sign_state_dict, key=lambda key: sign_state_dict[key])

        # print("sign_state",sign_state)
        car_info = None


        da_seg_mask = None
        if not g_daseg_queue.empty():
            da_seg_mask = g_daseg_queue.get()
        if da_seg_mask is not None:

            midpoint, img1 = calculate_midpoint(image, da_seg_mask, sign_state)
            object_avoid_state = 0
            if not g_car_queue.empty():
                car_info = g_car_queue.get()
                
                object_avoid_state = get_object_avoid_state(car_info, midpoint)
                if object_avoid_state == 1 or object_avoid_state == -1:
                    object_avoid_state_dict[object_avoid_state] = time.time() + Config.AVOIDANCE_TIME
                else:
                    object_avoid_state_dict[object_avoid_state] = time.time() 

            if object_avoid_state_dict:
                object_avoid_state = max(object_avoid_state_dict, key=lambda key: object_avoid_state_dict[key])

            # print(object_avoid_state_dict)
            # print("-------------")
            # print("object avoid state", object_avoid_state)
            midpoint_offset = get_midpoint_offset(object_avoid_state)
            midpoint += midpoint_offset
            # print("final midpoint")
            throttle, steering_angle = calculate_steering_angle(img1, midpoint, sign_state)
            if sign_state == -1 or sign_state == 1:
                throttle = Config.THROTTLE_TURN
                steering_angle *= Config.STEERING_TURN_COEF

            elif sign_state == 2:
                throttle = Config.THROTTLE_STOP

            if object_avoid_state == 1 or object_avoid_state == -1:
                throttle = Config.THROTTLE_OBSTACLE
                steering_angle *= Config.STEERING_OBSTACLE_COEF
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # cv2.imshow('img',img0)
        # print(time.time()-start)
        # pressKey = cv2.waitKey(1) & 0xFF
        # if pressKey == ord('q'):
        #     break
        # # Prepare visualization image
        # draw = image.copy()

        # # Send back throttle and steering angle
        # imgs = calculate_control_signal(img0, ll_seg_mask)
        # cv2.imshow('img',imgs)
        # pressKey = cv2.waitKey(1) & 0xFF
        # if pressKey == ord('q'):
        #     break


        # # Show the result to a window
        # cv2.imshow("Result", draw)
        # cv2.waitKey(1)

        # Send back throttle and steering angle
        message = json.dumps(
            {"throttle": throttle, "steering": steering_angle, })
        # print(message)
        await websocket.send(message)


for i in range(10):
    input = torch.randn(1, 3, 640, 480).to('cuda:0')
    segment_model(input)
    detection_model(input)
print("warm up done")
async def main():
    server = await websockets.serve(
        lambda websocket, path: process_image(websocket, path),
        "0.0.0.0",
        4567,
        ping_interval=None
    )
    


    # Run the server forever (until manually stopped)
    await server.wait_closed()

if __name__ == '__main__':
    p_traffic_sign = Process(target=process_traffic_sign_loop, args=(g_image1_queue,g_sign_queue, g_car_queue))
    p_traffic_sign.start()


    p_segment = Process(target=process_segment_loop, args=( g_image2_queue, g_daseg_queue))
    p_segment.start()

    asyncio.run(main())