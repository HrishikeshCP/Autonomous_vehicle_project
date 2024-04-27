import cv2
import numpy as np

# Load the image
frame = cv2.imread('Segformer/img and vdos/final road (Custom).jpg')  # Replace 'your_image.jpg' with the path to your image

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
    houghLines = cv2.HoughLinesP(img, 2, np.pi/180, 10, np.array([]), minLineLength=15, maxLineGap=5)
    return houghLines

def display_filled_region(img, lines, init_point):
    img_copy = img.copy()
    mask = np.zeros_like(img)
    if lines is not None:
        left_line = lines[0][0]  # Assuming first line is the left
        right_line = lines[1][0]  # Assuming second line is the right

        # Create a copy of the points
        pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                        [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)

        # Extend the mask upwards from the top two corners
        # pts[0][1] -= 100  # Adjust as needed
        # pts[3][1] -= 50  # Adjust as needed
        
        bottom_left_corner = pts[1]
        bottom_right_corner = pts[2]
    
        # Calculate the top corners
        top_left_corner = [bottom_left_corner[0], bottom_left_corner[1] - 300]
        top_right_corner = [bottom_right_corner[0], bottom_right_corner[1] - 300]
    
        # Create the polygon points
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
    return [[x1,y1,x2,y2]]

def average_slope_intercept_with_centre(img, lines):
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

    average_lines_with_centre = [np.array(left_line), np.array(right_line), np.array(center_line)]
    return average_lines_with_centre

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

left_line_history = []
right_line_history = []
history_length = 100 
init_point = (23, 384)

# Main loop for processing the image
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
    center_line_avg = np.array([[0,0,0,0]])
    for i in range(4):
        center_line_avg[0][i] = np.int32((left_line_avg[0][i] + right_line_avg[0][i])/2)

    average_lines_with_centre_avg = np.array([np.array(left_line_avg), np.array(right_line_avg), np.array(center_line_avg)])

    line_image_2_filled = display_filled_region(frame, average_lines_with_centre_avg, init_point)
except Exception as e:
    print("Error:", e)
    line_image_2_filled = frame

# Display the processed image
cv2.imshow('Frame', line_image_2_filled)
cv2.waitKey(0)
cv2.destroyAllWindows()