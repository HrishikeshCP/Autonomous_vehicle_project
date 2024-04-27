import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyrealsense2 as rs
import numpy as np

import serial
import time

ser = serial.Serial('COM9', 9600)


frame = cv2.imread('D:/Autonomous_vehicle_project/obstacledetection/yolov7modified/final road (Custom).jpg')  # Replace 'your_image.jpg' with the path to your image


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 7
    blur = cv2.GaussianBlur(gray, (kernel, kernel), sigmaX=0, sigmaY=0)
    canny = cv2.Canny(blur, 100, 160)
    return canny


def region_of_interest_trapezium(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)
    bottom_left = (0, height)
    top_left = (0, height * 0.5)
    top_right = (width, height * 0.5)
    bottom_right = (width, height)
    trapezium = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)
    cv2.fillPoly(mask, [trapezium], 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def houghLines(img):
    houghLines = cv2.HoughLinesP(img, 2, np.pi / 180, 10, np.array([]), minLineLength=15, maxLineGap=5)
    return houghLines


def display_filled_region(img, lines, init_point):
    img_copy = img.copy()
    mask = np.zeros_like(img)
    if lines is not None:
        left_line = lines[0][0]  # Assuming first line is the left
        right_line = lines[1][0]  # Assuming second line is the right

        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)

        bottom_left_corner = pts[1]
        bottom_right_corner = pts[2]

        top_left_corner = [bottom_left_corner[0], bottom_left_corner[1] - 300]
        top_right_corner = [bottom_right_corner[0], bottom_right_corner[1] - 300]

        pts2 = np.array([bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))

        pts = pts.reshape((-1, 1, 2))
        image = cv2.fillPoly(mask, [pts], (144, 238, 144))  # Light green color
        image = cv2.fillPoly(mask, [pts2], (144, 238, 144))  # Light green color

    return image


def make_points(img, lineSI):
    slope, intercept = lineSI
    height = img.shape[0]
    y1 = int(height)
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept_with_centre(img, lines):
    lower_value_slope = 0.5
    higher_value_slope = 2
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope:
                right_fit.append((slope, intercept))
    if left_fit == []:
        flag_left = False
    if right_fit == []:
        flag_right = False

    if flag_left:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        left_line = np.array([[None, None, None, None]])

    if flag_right:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        right_line = np.array([[None, None, None, None]])

    if flag_left and flag_right:
        center_line = np.array([[0, 0, 0, 0]])
        for i in range(4):
            center_line[0][i] = np.int32((left_line[0][i] + right_line[0][i]) / 2)
    else:
        center_line = np.array([[None, None, None, None]])

    average_lines_with_centre = [np.array(left_line), np.array(right_line), np.array(center_line)]
    return average_lines_with_centre


def update_line_history(line_history, new_line, history_length=5):
    if new_line[0].all() == np.array([None, None, None, None]).all() and line_history:
        new_line = line_history[-1]
    line_history.append(new_line)
    if len(line_history) > history_length:
        line_history.pop(0)
    return line_history


def average_line_from_history(line_history):
    if not line_history:
        return np.array([0, 0, 0, 0])
    avg_line = np.mean(np.array(line_history), axis=0, dtype=np.int32)
    return avg_line


def lane_detection(frame):
    left_line_history = []
    right_line_history = []
    history_length = 100
    init_point = (23, 384)

    try:
        canny_output = canny(frame)
        masked_output = region_of_interest_trapezium(canny_output)
        lines = houghLines(masked_output)

        average_lines_with_centre_avg = average_slope_intercept_with_centre(frame, lines)

        left_line = average_lines_with_centre_avg[0]
        right_line = average_lines_with_centre_avg[1]

        left_line_history = update_line_history(left_line_history, left_line, history_length)
        right_line_history = update_line_history(right_line_history, right_line, history_length)
        left_line_avg = average_line_from_history(left_line_history)
        right_line_avg = average_line_from_history(right_line_history)
        center_line_avg = np.array([[0, 0, 0, 0]])
        for i in range(4):
            center_line_avg[0][i] = np.int32((left_line_avg[0][i] + right_line_avg[0][i]) / 2)

        average_lines_with_centre_avg = np.array(
            [np.array(left_line_avg), np.array(right_line_avg), np.array(center_line_avg)])

        line_image_2_filled = display_filled_region(frame, average_lines_with_centre_avg, init_point)
    except Exception as e:
        print("Error:", e)
        line_image_2_filled = frame

    return line_image_2_filled
    

def check_intersection(masked_img1, masked_img2):
    # Resize masked_img2 to match the dimensions of masked_img1
    masked_img2_resized = cv2.resize(masked_img2, (masked_img1.shape[1], masked_img1.shape[0]))
    intersection = cv2.bitwise_and(masked_img1, masked_img2)
    intersects = False
    if np.any(intersection != 0):
        intersects = True
        print("Obstacle Entered Lane")
    return intersection, intersects


def detection():
    try:
        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.visual_preset, 1)
        colorizer.set_option(rs.option.histogram_equalization_enabled, 1.0)  # disable histogram equalization
        colorizer.set_option(rs.option.color_scheme, 0)  # replace 'float' with your desired color scheme
        colorizer.set_option(rs.option.min_distance, 0.2)  # replace 'float' with your desired min distance
        colorizer.set_option(rs.option.max_distance, 4)  # replace 'float' with your desired max distance 

        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA  
        # Load model
        print(device)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # model = attempt_load(weights, map_location=torch.device('cpu')) 
        #problem^^^^^^^^^^^^^^^^^^^^
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # if trace:
        #     model = TracedModel(model, device, opt.img_size)
        # if half:
        #     model.half()  # to FP16
        model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        # align_to = rs.stream.color
        # align = rs.align(align_to)

        # Set colorizer options

        while True:
            # frames = pipeline.wait_for_frames()
            # color_frame = frames.get_color_frame()

            # if not color_frame:
            #     continue

            # frame = np.asanyarray(color_frame.get_data())
            lane_masked_image = lane_detection(frame)
            # obstacle_masked_image = obstacle_detection(pipeline, colorizer, device, half, names, model, colors)

            #t0 = time.time()
            frames = pipeline.wait_for_frames()

            # aligned_frames = align.process(frames)
            aligned_frames=pipeline.wait_for_frames()
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            # if not depth_frame or not color_frame:
            #     continue

            img = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
            colorized_frame = colorizer.colorize(depth_frame)

            # Convert the colorized frame to a numpy array
            depth_colormap = np.asanyarray(colorized_frame.get_data())
            # Letterbox
            im0 = img.copy()
            img = img[np.newaxis, :, :, :]   

            # Create a zero mask image
            im0_masked = np.zeros_like(im0)     

            # Stack
            img = np.stack(img, 0)

            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)


            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)

                    # Initialize a list to store distances of all detected objects
                    object_distances = []

                    for *xyxy, _, _ in det:
                        indv_mask = np.zeros_like(im0_masked)
                        cv2.imshow("Masks", indv_mask)

                        # Draw bounding boxes on the zero mask image
                        cv2.rectangle(indv_mask, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 255, 255), -1)
                        
                        # Check for intersection only if both images are non-empty
                        if lane_masked_image.size != 0 and indv_mask.size != 0:
                            intersection, intersects = check_intersection(lane_masked_image, indv_mask)

                        # Display the stacked image
                        cv2.imshow("Intersection", intersection)
                        
                        if intersects:
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                            depth_data = np.array(depth_frame.get_data())

                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                            # Get depth data within the bounding box
                            depth_region = depth_data[y1:y2, x1:x2]

                            if depth_region.size > 0:
                                # Calculate the minimum distance within the bounding box
                                object_min_distance = np.min(depth_region[depth_region != 0]) * 0.001
                                
                                # Add the distance to the list
                                object_distances.append(object_min_distance)

                                

                    # After iterating through all detected objects, find the minimum distance
                    if object_distances:
                        min_distance = min(object_distances)
                # Check if the minimum distance is less than a threshold
                        if min_distance < 1.5:
                            print(f"Obstacle detected within {min_distance:.2f} meters!")
                            ser.write(b'stop\n')
                        else:
                            print("Path is clear!!")
                            ser.write(b'go\n')
                        # print(f"The minimum distance among all detected objects is: {min_distance:.2f} meters")
                    else:
                        print("No objects detected.")
                        ser.write(b'go\n')

                # Print time (inference + NMS)
                #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                cv2.imshow("Recognition result", im0)
                # cv2.imshow("Recognition result depth",depth_colormap)
                # cv2.imshow("Masked frame", im0_masked)  # Display the masked image

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # Display images for debugging
            cv2.imshow("Lane Masked Image", lane_masked_image)
            # cv2.imshow("Obstacle Masked Image", im0_masked)

            # Check if any of the images are empty
            if lane_masked_image.size == 0:
                print("Error: Lane Masked Image is empty")
            if im0_masked.size == 0:
                print("Error: Obstacle Masked Image is empty")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()


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
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    classes_to_detect = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 24, 25, 32, 58, 56]

    # Set the classes argument to the specified classes_to_detect
    opt.classes = classes_to_detect

    with torch.no_grad():
        detection()
