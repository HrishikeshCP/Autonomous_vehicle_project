import serial_mon
def write_to_text_file(file_path, value):
    try:
        # Open the file in append mode
        with open(file_path, 'a') as file:
            # Write the value to the next available row
            file.write(str(value) + '\n')
        print(f"Value '{value}' written to {file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")
def listen_to_serial(port, baudrate=115200):
    try:
        # Open the serial port
        ser = serial_mon.Serial(port, baudrate, timeout=1)
        print(f"Listening to {port} at {baudrate} baudrate. Press Ctrl+C to stop.")

        while True:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            
            # Print the received data
            print(str(line))
            write_to_text_file("data.txt", str(line))

    except serial_mon.SerialException as e:
        print(f"Error: {e}")

    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed.")

# Replace 'COM3' with the appropriate serial port on your system
# You can find the port in the Arduino IDE or your operating system's device manager.
listen_to_serial('COM4')
