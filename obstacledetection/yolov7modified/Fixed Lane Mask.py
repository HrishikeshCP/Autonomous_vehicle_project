import numpy as np
import cv2

def display_filled_region(img, lines, init_point):
    img_copy = img.copy()
    mask = np.zeros_like(img)
    if lines is not None:
        left_line = lines[0]  # Assuming first line is the left
        right_line = lines[1]  # Assuming second line is the right

        pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]], np.int32)

        bottom_left_corner = pts[1]
        bottom_right_corner = pts[2]

        top_left_corner = [bottom_left_corner[0], bottom_left_corner[1] - 300]
        top_right_corner = [bottom_right_corner[0], bottom_right_corner[1] - 300]

        pts2 = np.array([bottom_left_corner, bottom_right_corner, top_right_corner, top_left_corner], np.int32)
        pts2 = pts2.reshape((-1, 1, 2))

        pts = pts.reshape((-1, 1, 2))
        # image = cv2.fillPoly(img_copy, [pts], (144, 238, 144))  # Light green color
        # image = cv2.fillPoly(img_copy, [pts2], (144, 238, 144))  # Light green color

        image = cv2.polylines(img_copy, [pts], isClosed=True, color=(144, 238, 144), thickness=5)
        image = cv2.polylines(img_copy, [pts2], isClosed=True, color=(144, 238, 144), thickness=5) 

    return image

# Example usage:
# Initialize camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define lines
    left_line = [[-100, 480], [170, 260]]  # Example left line
    right_line = [[800, 480], [550, 260]]  # Example right line

    # Display filled region mask on live feed
    filled_region = display_filled_region(frame, [left_line, right_line], init_point=(100, 300))
    cv2.imshow('Filled Region Mask', filled_region)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()