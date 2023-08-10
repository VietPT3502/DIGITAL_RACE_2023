import time
import cv2
import numpy as np
from PIL import Image
from traffic_sign_detection import *
import sys
import torch
import torchvision
from pathlib import Path
import os
from numpy import random
import time
# Replace 'input_image' with the actual input image that you want to augment


# Apply the augmentations

sys.path.insert(0, "/media/gnort/HDD/Study/DIGITAL_RACE_2023-main/yolov7")
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, xywh2xyxy, box_iou
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
        w_size=640,
):
    """Non-Maximum Suppression (NMS) on inference results
    to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh = 15  # (pixels) minimum box width and height
    max_wh = 100000000  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh)).any(1), 4] = 0  # constrain traffic_sign bounding box
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

@torch.no_grad()
def detect_traffic_signs(
        im0,
        model,
        device,
        half=True,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        hide_labels=False,  # hide labels
        save_conf=True,  # hide confidences
        visualize=True
        ):

    # Load model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size
    # Dataloader

    if half:
        model.half()  # to FP16
    # Run inference
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup



    im = letterbox(im0, new_shape=imgsz, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=stride)[0]  # padded resize

    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)


    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Warmup
    if device.type != 'cpu' and (old_img_b != im.shape[0] or old_img_h != im.shape[2] or old_img_w != im.shape[3]):
        old_img_b = im.shape[0]
        old_img_h = im.shape[2]
        old_img_w = im.shape[3]
        for i in range(3):
            model(im)[0]

    # Inference
    t1 = time_synchronized()
    pred = model(im)
    t2 = time_synchronized()
    # NMS

    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det, w_size=im.shape[2])
    t3 = time_synchronized()
    # Separate car and traffic sign predictions
    car_pred = []
    traffic_sign_pred = []
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for det in pred:
        if len(det):
            if det[0][5] == 6:
                car_pred.append(det)
            else:
                traffic_sign_pred.append(det)
    if visualize:
        for i, det in enumerate(pred):  # per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            image = im0
        print(t2 - t1, t3 - t2)
        return traffic_sign_pred, car_pred, image
    print(t2 - t1, t3 - t2)
    return traffic_sign_pred, car_pred, None


if __name__ == "__main__":
    checkpoint = "weights/best_yolov7_640x640_72ep_7cls.pt"
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # dnn = False
    # data = "/media/gnort/HDD/Study/tfs-auto-algorithms/trafficsign.yaml"
    half = False
    detection_model = attempt_load(
        checkpoint,
        map_location=device,
    )
    im = cv2.imread("/media/gnort/HDD/Study/dataset/2023-08-07_02/00047.jpg")
    pred, _,  image = detect_traffic_signs(im, detection_model, device, half, imgsz=(640, 640))
    print("Pred: ", pred)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    