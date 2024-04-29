import numpy as np
import cv2

# Mouse callback function
def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse coordinates: ({x}, {y})")

# Initialize camera
cap = cv2.VideoCapture(1)

# Create a window
cv2.namedWindow('Camera Output')

# Set mouse callback function for the window
cv2.setMouseCallback('Camera Output', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    # Display camera output
    cv2.imshow('Camera Output', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
