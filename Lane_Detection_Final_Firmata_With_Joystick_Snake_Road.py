import cv2
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
import io

import pygame

import serial
import pyfirmata2
import logging
import time

import warnings

# Initialize pygame for joystick reading
pygame.init()
pygame.joystick.init()

isAutomatic = False

# Ensure the joystick is connected
if pygame.joystick.get_count() < 1:
    print("Please connect a joystick.")
    quit()
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

# Redirect stderr to null device
sys.stderr = open(os.devnull, 'w')

# Ignore all warnings
warnings.filterwarnings("ignore")

# Setup logging to display debug information
# logging.basicConfig(level=logging.DEBUG)

# com_channel = '/dev/tty.usbmodem1401'
com_channel = 'COM10'
baud_rate = 9600

board = pyfirmata2.Arduino(com_channel)
a_pin = board.get_pin('d:4:o')  # Example pin setup for 'a'
b_pin = board.get_pin('d:5:p')  # Example pin setup for 'b' as PWM

# Setting up the pins
# ch1_pin = board.get_pin(f'd:{3}:i')
# ch1 = board.get_pin(f'a:{1}:i')
# ch6_pin = board.get_pin(f'd:{6}:i')
# ch6 = board.get_pin(f'a:{5}:i')

# Starting iterator to receive input data
# it = pyfirmata2.util.Iterator(board)
# it.start()
# board.samplingOn()

# # Function to handle callbacks from analog inputs
# def callback_ch1(data):
#     print(f"Analog data on ch1 (A0): {data}")

# def callback_ch6(data):
#     print(f"Analog data on ch6 (A5): {data}")
# # # Enable reporting for pins
# ch1.enable_reporting()
# ch1.register_callback(callback_ch1)
# ch6_pin.enable_reporting()
# ch6.register_callback(callback_ch6)

# Function to set motor control pins
def send_pwm(value):
    abs_value = abs(value)
    if value > 0:
        a_pin.write(1)
        b_pin.write(abs_value / 255)  # Assuming 'value' ranges from -255 to 255
        # logging.debug(f"Set PWM on pin 5 to {value}")

    elif value < 0:
        a_pin.write(0)
        b_pin.write(abs_value / 255)
        # logging.debug(f"Set PWM on pin 5 to {value}")

    else:
        a_pin.write(0)
        b_pin.write(0)
        # logging.debug(f"Set PWM on pin 5 to {value}")

    # # Send value back for monitoring
    # ser.write(f"Sent PWM: {value}\n".encode())

# # Open the serial port that your Arduino is connected to (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux, /dev/tty.usbserial on MacOS).
# ser = serial.Serial(com_channel, baud_rate)  # Adjust this to match your connection
# # def read_from_arduino(ch1_previous,ch6_previous):
# def read_from_arduino():
#     # while True:
#     if ser.in_waiting > 0:
#         try:
#             data = ser.readline().decode('utf-8').strip()
#             # ch1, ch6 = data.split(',')  
#             ch6 = data     
#         except:
#             # ch1 = 0
#             ch6 = 993
#         print(f"CH6: {ch6}")
#         # print(f"CH1: {ch1}, CH6: {ch6}")
#         # return ch1,ch6
#         # time.sleep(0.1)

# def send_value(value):
#     ser.write(str(value).encode())  # Convert the integer to a string and encode it to bytes
#     ser.flush()
#     # time.sleep(0.01) # Wait a bit for the Arduino to process the information

# # Read response
# def recieve_value():
#     if ser.in_waiting > 0:
#         ser_line = ser.readline().decode().strip()
#         print(f"Arduino responded with: {ser_line}")


def canny(img,route):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kernel = 5
    if route == 0:
        kernel = 7
        blur = cv2.GaussianBlur(gray, (kernel,kernel), sigmaX=0, sigmaY=0)
        canny = cv2.Canny(blur,80,100)
    elif route == 1:
        kernel = 7
        blur = cv2.GaussianBlur(gray, (kernel,kernel), sigmaX=0, sigmaY=0)
        canny = cv2.Canny(blur,80,100)
    # canny = cv2.Canny(blur,100,160)
    return canny

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)
    # triangle = np.array([[(50,height), (400,300), (900,height)]], np.int32)
    # triangle = np.array([[(0,height), (300,150), (900,height)]], np.int32)
    triangle = np.array([[(-100,height), (width//2,height//4), (width+100,height)]], np.int32)
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def region_of_interest_trapezium(img,route):

    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)

    # Define coordinates for the trapezium
    # Adjust the points (x1, y1), (x2, y2), (x3, y3), (x4, y4) as needed
    # bottom_left = (0, height)
    # top_left = (0, height * 0.5)
    # # top_left = (0, height * 0.3)
    # top_right = (width, height * 0.5)
    # # top_right = (width, height * 0.3)
    # bottom_right = (width, height)

    if route == 0:
        bottom_left = (0, height)
        top_left = (0, height * 0.5)
        top_right = (width, height * 0.5)
        bottom_right = (width, height)       

    elif route == 1:
        bottom_left = (0, height)
        top_left = (100, height * 0.5)
        top_right = (width-100, height * 0.5)
        bottom_right = (width, height)

    # bottom_left = (width * 0.1, height)
    # top_left = (width * 0.4, height * 0.6)
    # top_right = (width * 0.6, height * 0.6)
    # bottom_right = (width * 0.9, height)

    # np.array expects points as [[first_point, second_point, third_point, fourth_point]]
    trapezium = np.array([[bottom_left, top_left, top_right, bottom_right]], np.int32)

    # Fill the polygon (trapezium here) with white (255)
    cv2.fillPoly(mask, [trapezium], 255)

    # Apply the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def houghLines(img,route):
    if route == 0:
        houghLines = cv2.HoughLinesP(img,2,np.pi/180,100,np.array([]),minLineLength = 30, maxLineGap = 10)
    elif route == 1:
        houghLines = cv2.HoughLinesP(img,2,np.pi/180,100,np.array([]),minLineLength = 50, maxLineGap = 10)
    # houghLines = cv2.HoughLinesP(img,2,np.pi/180,100,np.array([]),minLineLength = 50, maxLineGap = 10)
    # houghLines = cv2.HoughLinesP(img,2,np.pi/180,100,np.array([]),minLineLength = 20, maxLineGap = 5)
    # houghLines = cv2.HoughLinesP(img,2,10*(np.pi/180),10,np.array([]),minLineLength = 15, maxLineGap = 5)
    return houghLines

def display_lines(img,lines,init_point,show_vehicle_centre):
    img_copy = img.copy()
    if lines is not None:
        for i in range(len(lines)):
            line = np.array(lines[i])
            if line.all() != np.array([[None,None,None,None]]).all():
                if i == 2:
                    for[x1,y1,x2,y2] in line:
                        cv2.line(img_copy, (x1,y1), (x2,y2), (255,0,0),10)
                        cv2.circle(img_copy, ((x1+x2)//2,(y1+y2)//2), 10, (255,0,0), 10)
                        # centre = ((x1+x2)//2,(y1+y2)//2)
                        if show_vehicle_centre == True:
                            cv2.circle(img_copy, init_point, 10, (255,0,255), 10)
                else:
                    for [x1,y1,x2,y2] in line:
                        cv2.line(img_copy, (x1,y1), (x2,y2), (255,0,0),10)
                        if show_vehicle_centre == True:
                            cv2.circle(img_copy, init_point, 10, (255,0,255), 10)
    return img_copy

def display_lines_with_filled_region(img,lines,init_point):
    img_copy = img.copy()
    centre = None
    if lines is not None:
        left_line = lines[0][0]  # Assuming first line is the left
        right_line = lines[1][0]  # Assuming second line is the right

        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)
        
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img_copy, [pts], (144, 238, 144))  # Light green color
        
        for i in range(len(lines)):
            line = np.array(lines[i])
            if line.all() != np.array([[None,None,None,None]]).all():
                if i == 2:
                    for[x1,y1,x2,y2] in line:
                        cv2.line(img_copy, (x1,y1), (x2,y2), (255,0,0),10)
                        cv2.circle(img_copy, ((x1+x2)//2,(y1+y2)//2), 10, (255,0,0), 10)
                        cv2.circle(img_copy, init_point, 10, (255,0,255), 10)
                        centre = ((x1+x2)//2,(y1+y2)//2)
                else:
                    for [x1,y1,x2,y2] in line:
                        cv2.line(img_copy, (x1,y1), (x2,y2), (255,0,0),10)
                        cv2.circle(img_copy, init_point, 10, (255,0,255), 10)
    return img_copy, centre

def make_points(img,lineSI):
    slope, intercept = lineSI
    height = img.shape[0]
    y1 = int(height)
    # y2 = int(y1*4.0/5)
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [[x1,y1,x2,y2]]


def average_slope_intercept(img,lines):
    # lower_value_slope = 0.5
    # higher_value_slope = 3
    lower_value_slope = 0.5
    higher_value_slope = 2.5
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            # print(intercept,slope)
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope :
                right_fit.append((slope,intercept))
    if left_fit == []:
        # left_fit = np.array([(0.0001,0.0001)])
        flag_left = False
    if right_fit == []:
        # right_fit = np.array([(0.0001,0.0001)])
        flag_right = False

    if flag_left:
        left_fit_average = np.average(left_fit,axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        left_line = np.array([[None,None,None,None]])

    if flag_right:
        right_fit_average = np.average(right_fit,axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        right_line = np.array([[None,None,None,None]])


    average_lines = [np.array(left_line),np.array(right_line)]
    return average_lines

def average_slope_intercept_with_centre(img,lines):
    # lower_value_slope = 0.5
    # higher_value_slope = 3
    lower_value_slope = 0.5
    higher_value_slope = 2.5
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            # print(intercept,slope)
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope :
                right_fit.append((slope,intercept))
    if left_fit == []:
        # left_fit = np.array([(0.0001,0.0001)])
        flag_left = False
    if right_fit == []:
        # right_fit = np.array([(0.0001,0.0001)])
        flag_right = False

    if flag_left:
        left_fit_average = np.average(left_fit,axis=0)
        left_line = make_points(img, left_fit_average)
    else:
        left_line = np.array([[None,None,None,None]])

    if flag_right:
        right_fit_average = np.average(right_fit,axis=0)
        right_line = make_points(img, right_fit_average)
    else:
        right_line = np.array([[None,None,None,None]])

    if flag_left and flag_right:
        center_line = np.array([[0,0,0,0]])
        for i in range(4):
            center_line[0][i] = np.int32((left_line[0][i] + right_line[0][i])/2)
    else:
        center_line = np.array([[None,None,None,None]])

    average_lines = [np.array(left_line),np.array(right_line),np.array(center_line)]
    return average_lines

def color_filter_for_gray(img):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for gray colors in HSV
    # Gray color will have low saturation, so we use a higher lower bound for value to avoid very dark regions
    lower_gray = np.array([0, 0, 50], dtype="uint8")
    upper_gray = np.array([180, 50, 255], dtype="uint8")
    
    # Create mask for the gray color
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Apply the mask to the input image
    masked_image = cv2.bitwise_and(img, img, mask=mask_gray)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
    
    return masked_image

left_line_history = []
right_line_history = []
# center_line_history = []
history_length = 10  # Keep history of last 5 frames
# history_length = 25  # Keep history of last 5 frames
# init_point = (428, 430)
# init_point = (283, 384)
# init_point = (267, 408)


def update_line_history(line_history, new_line, history_length=5):
    if new_line[0].all() == np.array([None,None,None,None]).all() and line_history:
        # Use the most recent valid line if the new line is invalid
        new_line = line_history[-1]
    line_history.append(new_line)
    if len(line_history) > history_length:
        line_history.pop(0)
    # print(line_history)
    return line_history


def average_line_from_history(line_history):
    if not line_history:
        return np.array([0, 0, 0, 0])
    avg_line = np.mean(np.array(line_history), axis=0, dtype=np.int32)
    return avg_line

class PIDController:
    def __init__(self, Kp, Ki, Kd, max_output=None, min_output=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.min_output = min_output
        self.integral = 0
        self.previous_error = 0

    def calculate(self, error, delta_time):
        # Proportional term
        proportional = self.Kp * error
        
        # Integral term
        self.integral += error * delta_time
        integral = self.Ki * self.integral
        
        # Derivative term
        derivative = self.Kd * (error - self.previous_error) / delta_time
        
        # Update previous error
        self.previous_error = error
        
        # Calculate total output
        output = proportional + integral + derivative
        
        # Clamp output to max and min values if specified
        if self.max_output is not None and output > self.max_output:
            output = self.max_output
        elif self.min_output is not None and output < self.min_output:
            output = self.min_output
            
        return output

def preprocess_frame(frame):
    new_width = 640
    new_height = 480
    # aspect_ratio = original_width / original_height
    # # Calculate the new height maintaining the aspect ratio
    # new_height = int(new_width / aspect_ratio)
    frame = cv2.resize(frame, (new_width, new_height))
    # Convert frame to float
    frame_float = frame.astype(np.float32)
    # Calculate the mean of the pixel values
    mean = np.mean(frame_float)
    # Scale factor for contrast adjustment; values < 1.0 decrease contrast
    # scale_factor = 1.0
    scale_factor = 0.8
    # Adjust the contrast
    # Moving pixel values towards the mean to reduce contrast
    frame_adjusted = (frame_float - mean) * scale_factor + mean
    # Clip values to stay between 0 and 255 and convert back to uint8
    frame = np.clip(frame_adjusted, 0, 255).astype(np.uint8)
    return frame

# def update_pwm_value(new_value):
#     pwm_queue.put(new_value)  # Place the new PWM value in the queue

# capture = cv2.VideoCapture('/Users/anavart.pandya/MY DRIVE D/Autonomous Vehicles Sem8 IITGN/Advanced-Lane-Lines-master/IITGN_SnakeRoad_CentralArcade (online-video-cutter.com).mp4')
capture = cv2.VideoCapture(0)
# desired_fps = 15
# fps = capture.get(cv2.CAP_PROP_FPS)
# frame_start_pos = 0
# capture.set(cv2.CAP_PROP_POS_FRAMES, frame_start_pos)
# Capture properties for VideoWriter
# fps = capture.get(cv2.CAP_PROP_FPS)
# frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Initialize VideoWriter
# # fourcc = cv2.VideoWriter_fourcc(*'x264')  # or use 'XVID'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'XVID'
# out = cv2.VideoWriter('IITGNvid2_output.mp4', fourcc, fps, (frame_width, frame_height))
counter = 0
# counter = frame_start_pos

close_all = False

pid_controller = PIDController(Kp=300, Ki=6.0, Kd=150000, max_output=255, min_output=-255)

previous_time = cv2.getTickCount()
# ch1_previous = 1498
# ch6_previous = 998

# pwm_value = 0
# threading.Thread(target=send_value, args=(pwm_value,), daemon=True).start()
# pwm_queue = Queue()

update_frame = False
frame = np.zeros((480, 640, 3), dtype=np.uint8)

route = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            # print(f"Button {event.button} pressed.")
            # print(event.button)
            if event.button == 4:
                isAutomatic = False
                update_frame = True
                # control_mode = 'JOYSTICK'
                # print("Mode Set to Joystick Control")
            if event.button == 10:
                isAutomatic = True
                # control_mode = 'AUTONOMOUS'
                # print("Mode Set to Automatic Control")
            if event.button == 5:
                close_all = True
            if event.button == 0:
                route = 0
            if event.button == 7:
                route = 1
    if isAutomatic:
        control_mode = 'AUTONOMOUS'
    else:
        control_mode = 'JOYSTICK'
        
    if isAutomatic:
        current_time = cv2.getTickCount()
        delta_time = (current_time - previous_time) / cv2.getTickFrequency()
        previous_time = current_time

        ret, frame = capture.read()
        if not ret:
            break
        counter += 1
        frame = preprocess_frame(frame)
  
        original_height, original_width = frame.shape[0], frame.shape[1]
        # original_height, original_width = frame_height, frame_width
        # original_height, original_width = frame.shape[1], frame.shape[0]
        init_point = (original_width//2,384)

        # original_height, original_width = frame.shape[0], frame.shape[1]
        # new_width = 700
        # aspect_ratio = original_width / original_height

        # # Calculate the new height maintaining the aspect ratio
        # new_height = int(new_width / aspect_ratio)

        # frame = cv2.resize(frame, (new_width, new_height))

        # color_filtered_image = color_filter_for_gray(frame)
        # canny_output = canny(color_filtered_image)
        try:
            canny_output = canny(frame,route)
            # masked_output = region_of_interest(canny_output)
            masked_output = region_of_interest_trapezium(canny_output,route)
            # masked_output = canny_output
            lines = houghLines(masked_output,route)
            # line_image = display_lines(frame,lines)
            average_lines = average_slope_intercept(frame,lines)
            average_lines_with_centre = average_slope_intercept_with_centre(frame,lines)

            left_line = average_lines_with_centre[0]
            right_line = average_lines_with_centre[1]
            # center_line = average_lines_with_centre[2]

            left_line_history = update_line_history(left_line_history, left_line, history_length)
            right_line_history = update_line_history(right_line_history, right_line, history_length)
            # center_line_history = update_line_history(center_line_history, center_line, history_length)

            # Calculate averaged lines
            left_line_avg = average_line_from_history(left_line_history)
            right_line_avg = average_line_from_history(right_line_history)
            # center_line_avg = average_line_from_history(center_line_history)
            center_line_avg = np.array([[0,0,0,0]])
            for i in range(4):
                center_line_avg[0][i] = np.int32((left_line_avg[0][i] + right_line_avg[0][i])/2)

            average_lines_avg = np.array([np.array(left_line_avg),np.array(right_line_avg)])
            average_lines_with_centre_avg = np.array([np.array(left_line_avg),np.array(right_line_avg),np.array(center_line_avg)])

            # line_image_1 = display_lines(frame,average_lines_avg,init_point)
            # line_image_1_filled = display_lines_with_filled_region(frame,average_lines_avg,init_point)
            # line_image_2 = display_lines(frame,average_lines_with_centre_avg,init_point)
            line_image_2_filled, calculated_centre = display_lines_with_filled_region(frame,average_lines_with_centre_avg,init_point)

            error = np.sqrt((init_point[0] - calculated_centre[0])**2 + (init_point[1] - calculated_centre[1])**2)
            max_error = np.sqrt((init_point[0] - original_width)**2)
            error_normalised = error/max_error
            if calculated_centre[0] < init_point[0]:
                error_normalised = -error_normalised
            # Convert error to string and format it to display only two decimal places
            error_text = "Error: {:.2f}".format(error_normalised)

            # Display the error on the frame
            cv2.putText(line_image_2_filled, error_text, (original_width - 250, int(original_height*(2/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            pwm_value = pid_controller.calculate(error_normalised, delta_time)

            # Display PWM value on the frame for visualization
            pwm_text = "PWM: {:.2f}".format(pwm_value)
            cv2.putText(line_image_2_filled, pwm_text, (25, int(original_height*(2/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            send_pwm(int(pwm_value))

            # send_value(int(pwm_value))
            # recieve_value()
        except:
            line_image_2_filled = frame
            pwm_value = 0 
            pwm_text = "PWM: {:.2f}".format(pwm_value)
            cv2.putText(line_image_2_filled, pwm_text, (25, int(original_height*(2/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            send_pwm(int(pwm_value))
            # send_value(int(pwm_value))
            # recieve_value()
        # Start PWM sending in a separate thread
        # send_value(int(pwm_value))
        # threading.Thread(target=send_value, args=(pwm_value,), daemon=True).start()
        # recieve_value()

        # out.write(line_image_2_filled)

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(line_image_2_filled, fps_text, (25, int(original_height*(4/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        control_mode_text = "CONTROL MODE: " + control_mode
        cv2.putText(line_image_2_filled, control_mode_text, (25, int(original_height*(4.5/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if route == 0:
            route_ = "Snake Road"
        elif route == 1:
            route_ = "Central Arcade Road"
        route_text = "ROUTE: " + route_
        cv2.putText(line_image_2_filled, route_text, (25, int(original_height*(1/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # ch1_previous, ch6_previous = read_from_arduino(ch1_previous,ch6_previous)
        # ch1, ch6 = read_from_arduino()
        # print(f"CH1: {ch1}, CH6: {ch6}")

        # read_from_arduino()

        # old_stdout = sys.stdout
        # sys.stdout = buffer = io.StringIO()
        # output = buffer.getvalue()
        # sys.stdout = old_stdout
        # print("Captured:", output)

        # ch1_value = ch1.read()
        # print(ch1.read())
        # ch6_value = ch6_pin.read()
        # print(ch1_value,ch6_value)

        # print(int(pwm_value))
        # cv2.imshow('Frame', line_image_2_filled)
        # time.sleep(0.5)
        # if counter > 3500:
        #     print(counter)
        #     break
    
    else:
        # if update_frame:
        #     ret, frame = capture.read()
        #     if not ret:
        #         break
        #     frame = preprocess_frame(frame)
        #     frame.fill(0)
        #     update_frame = False
        frame.fill(0)
        original_height, original_width = frame.shape[0], frame.shape[1]
        frame_copy = frame.copy()

        pygame.event.pump()  # Update pygame event queue
        axis_value = joystick.get_axis(2)  # Read the first axis (change this index if needed)
        scaled_value = int(axis_value*255)
        pwm_value_joystick = scaled_value
        send_pwm(pwm_value_joystick)

        line_image_2_filled = frame_copy
        # fps = capture.get(cv2.CAP_PROP_FPS)
        # fps_text = "FPS: {:.2f}".format(fps)
        # cv2.putText(line_image_2_filled, fps_text, (25, int(original_height*(4/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        control_mode_text = "CONTROL MODE: " + control_mode
        cv2.putText(line_image_2_filled, control_mode_text, (25, int(original_height*(4.5/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        pwm_text_joystick = "PWM: {:.2f}".format(pwm_value_joystick)
        cv2.putText(line_image_2_filled, pwm_text_joystick, (25, int(original_height*(2/5))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', line_image_2_filled)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or close_all:  # Press 'q' to exit
        print(counter)
        # ser.close() q # Close the serial port
        board.exit()
        break
    # elif key == ord('a'):  # Press 'q' to exit
    #     isAutomatic = not isAutomatic
    # elif key == ord('a'):  # Press 'q' to exit
    #     isAutomatic = True
    # if key == ord('m'):  # Press 'q' to exit
    #     isAutomatic = False

    # if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
    #     print(counter)
    #     # ser.close() q # Close the serial port
    #     board.exit()
    #     break
    # if cv2.waitKey(1) & 0xFF == ord('a'):  # Press 'q' to exit
    #     isAutomatic = True
    # if cv2.waitKey(1) & 0xFF == ord('m'):  # Press 'q' to exit
    #     isAutomatic = False
cv2.destroyAllWindows()
# ser.close()  # Close the serial port
board.exit()
pygame.quit()

# Close the redirected stderr (optional, depends on your script's structure)
sys.stderr.close()

# Reset stderr to its original value if needed later in the script
sys.stderr = sys.__stderr__