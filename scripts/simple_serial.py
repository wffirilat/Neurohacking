import serial
import pdb

port = '/dev/tty.OpenBCI-DN008VTF'
#port = '/dev/tty.OpenBCI-DN0096XA'
baud = 115200
ser = serial.Serial(port= port, baudrate = baud, timeout = None)
pdb.set_trace()