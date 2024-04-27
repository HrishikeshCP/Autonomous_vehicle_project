import sys
import serial
from rplidar import RPLidar

PORT_NAME = 'com4'
ARDUINO_PORT = '/dev/ttyUSB0'  # Replace this with the port your Arduino is connected to
BAUD_RATE = 9600  # Match this with the baud rate of your Arduino


def run(path='output.txt'):
    '''Main function'''
    lidar = RPLidar(PORT_NAME)
    outfile = open(path, 'w')
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)  # Open serial connection to Arduino
    try:
        print('Recording measurements... Press Ctrl+C to stop.')
        for measurement in lidar.iter_measurments():
            boolean, quality, angle, distance = measurement  # Unpack values here
            if distance != 0.0 and ((330 < int(angle) < 359) or (int(angle) < 30)):
                print(f"Angle: {angle}, Distance: {distance}")
                if distance < 2000:
                    # Send signal to Arduino
                    arduino.write(distance)  # Assuming '1' is the signal you want to send
    except KeyboardInterrupt:
        print('Stopping.')
    lidar.stop()
    lidar.disconnect()
    outfile.close()
    arduino.close()  # Close serial connection to Arduino


if __name__ == '__main__':
    run('output.txt')