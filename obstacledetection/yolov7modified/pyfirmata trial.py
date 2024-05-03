import pyfirmata
import time

board = pyfirmata.Arduino('COM8')

it = pyfirmata.util.Iterator(board)
it.start()

rc_read = board.get_pin('d:3:i')

while True:
    print(rc_read.read())
    # time.sleep(2)