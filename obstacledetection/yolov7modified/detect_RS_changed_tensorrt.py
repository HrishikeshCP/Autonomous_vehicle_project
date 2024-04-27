import argparse
import time
from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, set_logging, increment_path
from utils.plots import plot_one_box

import pyrealsense2 as rs
import numpy as np
from torch2trt import torch2trt
from torch2trt import TRTModule

colorizer = rs.colorizer()

# Set colorizer options
colorizer.set_option(rs.option.visual_preset, 1)
colorizer.set_option(rs.option.histogram_equalization_enabled, 1.0)  # disable histogram equalization
colorizer.set_option(rs.option.color_scheme, 0)  # replace 'float' with your desired color scheme
colorizer.set_option(rs.option.min_distance, 0.2)  # replace 'float' with your desired min distance
colorizer.set_option(rs.option.max_distance, 4)  # replace 'float' with your desired max distance

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = torch.device(opt.device)
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    while(True):
        frames = pipeline.wait_for_frames()
        aligned_frames=pipeline.wait_for_frames()
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        colorized_frame = colorizer.colorize(depth_frame)
        depth_colormap = np.asanyarray(colorized_frame.get_data())
        im0 = img.copy()
        img = img[np.newaxis, :, :, :]        
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time.time()
        with torch.no_grad():
            pred = model(img, augment=opt.augment)[0]
        t2 = time.time()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time.time()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)

                for *xyxy, _, _ in det:
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    depth_value = depth_frame.get_distance(int(x_center), int(y_center))
                    object_distance = depth_value
                    print(object_distance)
                    if object_distance < 3.0:
                        print("stop")             

        cv2.imshow("Recognition result", im0)
        cv2.imshow("Recognition result depth",depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2')
