import cv2
import serial_mon
from time import sleep

# Enter your COM port in the below line
ard = serial_mon.Serial('com4', 115200)
sleep(2)
print(ard.readline(ard.inWaiting()))


def display_camera(camera_index=0):
    # Open the camera with the DirectShow backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    print(f"Press 'q' to quit...")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Could not read frame")
            break

        # Display the frame
        cv2.imshow("Camera Output", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the camera index (default is 0)
    camera_index = 2

    # Display the camera output
    display_camera(camera_index)
