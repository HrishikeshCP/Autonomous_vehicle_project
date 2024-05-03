import pyfirmata
import time

board = pyfirmata.Arduino('COM8')

it = pyfirmata.util.Iterator(board)
it.start()

brake_dir = board.get_pin('d:13:o')  # Example pin setup for 'a'
brake_pwm = board.get_pin('d:11:p')  # Example pin setup for 'b' as PWM

rc_read = board.get_pin('a:1:i')

while True:
    brake_dir.write(1)
    brake_pwm.write(1)
    # print(rc_read.read())
    time.sleep(2)
    brake_dir.write(0)
    brake_pwm.write(1)
    time.sleep(2)