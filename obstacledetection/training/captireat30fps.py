import cv2
import os
import serial_mon
from time import sleep
import csv


def listen_to_serial(port='com4', baudrate=115200):
    # try:
        # Open the serial port
        ser = serial_mon.Serial(port, baudrate)
        # while True:
            # Read a line from the serial port
        line = ser.readline().decode('utf-8').strip()
            
            # Print the received data
        return str(line)


# def write_to_csv(value,file_path='encoder_values.csv'):
#     try:
#         # Open the CSV file in append mode
#         with open(file_path, 'a', newline='') as csvfile:
#             # Create a CSV writer object
#             csv_writer = csv.writer(csvfile)

#             # Write the value to the next row
#             csv_writer.writerow([value])

#         print(f"Value '{value}' written to {file_path}.")

#     except Exception as e:
#         print(f"Error: {e}")

def write_to_text_file(file_path, value):
    try:
        # Open the file in append mode
        with open(file_path, 'a') as file:
            # Write the value to the next available row
            file.write(str(value) + '\n')
        print(f"Value '{value}' written to {file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")

# Function to capture and display frames from three cameras
def capture_and_display(cam1, cam2, cam3):
    # Open camera connections
    cap1 = cv2.VideoCapture(cam1)
    cap2 = cv2.VideoCapture(cam2)
    cap3 = cv2.VideoCapture(cam3)

    # Create windows for display
    cv2.namedWindow('Camera 1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Camera 2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Camera 3', cv2.WINDOW_NORMAL)

    # Create folders for saving images
    folders = ['left', 'center', 'right']
    for folder in folders:
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass

    # Variables for saving frames
    save_frames = False
    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        if not ret1 or not ret2 or not ret3:
            print("Error reading frames from one of the cameras")
            break

        # Display frames
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)
        cv2.imshow('Camera 3', frame3)

        # Save frames when 'r' key is pressed
        key = cv2.waitKey(33)
        save_frames = True
        # if key == ord('r'):
        #     save_frames = not save_frames # Toggle saving frames
        #     if save_frames:
        #         print("Saving frames...")
        #     else:
        #         print("Stopped saving frames")

        if save_frames:
            # Save frames in respective folders
            cv2.imwrite(f'left/frame_{frame_count}.png', frame1)
            cv2.imwrite(f'center/frame_{frame_count}.png', frame2)
            cv2.imwrite(f'right/frame_{frame_count}.png', frame3)
            
            # write_to_csv(listen_to_serial())
            
            write_to_text_file("data.txt", listen_to_serial())
            write_to_text_file("data.txt", '10')
            frame_count += 1

    # Release camera connections and close windows
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify camera indices (0, 1, 2, etc.) or video file paths
    camera1_index = 0  # Update with your camera index or path
    camera2_index = 3  # Update with your camera index or path
    camera3_index = 2  # Update with your camera index or path

    capture_and_display(camera1_index, camera2_index, camera3_index)
