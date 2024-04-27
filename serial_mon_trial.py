import serial
import time

# Open serial connection to Arduino
ser = serial.Serial('COM8', 9600)  # Adjust 'COM3' to match your Arduino's serial port

try:
    while True:
        # Example data to send
        data_to_send = b'Hello Arduino\n'  # Convert string to bytes

        # Write data to Arduino
        ser.write(data_to_send)  # Send data to Arduino

        # Print the sent data to console
        print("Sent to Arduino:", data_to_send.decode())

        time.sleep(1)  # Optional delay
except KeyboardInterrupt:
    ser.close()  # Close serial connection on Ctrl+C
