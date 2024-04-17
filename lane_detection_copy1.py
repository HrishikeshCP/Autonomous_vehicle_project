import cv2
import matplotlib.pyplot as plt
import numpy as np

capture = cv2.VideoCapture(0)

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 7
    blur = cv2.GaussianBlur(gray, (kernel,kernel), sigmaX=0, sigmaY=0)
    canny = cv2.Canny(blur,100,160)
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
    houghLines = cv2.HoughLinesP(img,2,10*(np.pi/180),10,np.array([]),minLineLength = 15, maxLineGap = 5)
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
    if lines is not None:
        left_line = lines[0][0] 
        right_line = lines[1][0] 
        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img_copy, [pts], (144, 238, 144))
        for i in range(len(lines)):
            line = np.array(lines[i])
            if line.all() != np.array([[None,None,None,None]]).all():
                if i == 2:
                    for[x1,y1,x2,y2] in line:
                        cv2.line(img_copy, (x1,y1), (x2,y2), (255,0,0),10)
                        cv2.circle(img_copy, ((x1+x2)//2,(y1+y2)//2), 10, (255,0,0), 10)
                        cv2.circle(img_copy, init_point, 10, (255,0,255), 10)
                else:
                    for [x1,y1,x2,y2] in line:
                        cv2.line(img_copy, (x1,y1), (x2,y2), (255,0,0),10)
                        cv2.circle(img_copy, init_point, 10, (255,0,255), 10)
    return img_copy
def make_points(img,lineSI):
    slope, intercept = lineSI
    height = img.shape[0]
    y1 = int(height)
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [[x1,y1,x2,y2]]
def average_slope_intercept(img,lines):
    lower_value_slope = 0.5
    higher_value_slope = 2
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
        flag_left = False
    if right_fit == []:
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
    lower_value_slope = 0.5
    higher_value_slope = 2
    flag_left = True
    flag_right = True
    left_fit = []
    right_fit = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -lower_value_slope and slope >= -higher_value_slope:
                left_fit.append((slope, intercept))
            elif slope >= lower_value_slope and slope <= higher_value_slope :
                right_fit.append((slope,intercept))
    if left_fit == []:
        flag_left = False
    if right_fit == []:
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
left_line_history = []
right_line_history = []
history_length = 100 
init_point = (23, 384)
def update_line_history(line_history, new_line, history_length=5):
    if new_line[0].all() == np.array([None,None,None,None]).all() and line_history:
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
counter = 0
while True:
    ret, frame = capture.read()
    if not ret:
        break
    counter += 1
    original_height, original_width = frame.shape[0], frame.shape[1]
    try:
        canny_output = canny(frame)
        masked_output = region_of_interest_trapezium(canny_output)
        lines = houghLines(masked_output)
        # average_lines = average_slope_intercept(frame,lines)
        # average_lines_with_centre = average_slope_intercept_with_centre(frame,lines)

        left_line = average_lines_with_centre[0]
        right_line = average_lines_with_centre[1]

        left_line_history = update_line_history(left_line_history, left_line, history_length)
        right_line_history = update_line_history(right_line_history, right_line, history_length)
        left_line_avg = average_line_from_history(left_line_history)
        right_line_avg = average_line_from_history(right_line_history)
        center_line_avg = np.array([[0,0,0,0]])
        for i in range(4):
            center_line_avg[0][i] = np.int32((left_line_avg[0][i] + right_line_avg[0][i])/2)

        average_lines_avg = np.array([np.array(left_line_avg),np.array(right_line_avg)])
        average_lines_with_centre_avg = np.array([np.array(left_line_avg),np.array(right_line_avg),np.array(center_line_avg)])

        line_image_2_filled = display_lines_with_filled_region(frame,average_lines_with_centre_avg,init_point)
    except:
        line_image_2_filled = frame

    cv2.imshow('Frame', line_image_2_filled)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
        print(counter)
        break
cv2.destroyAllWindows()