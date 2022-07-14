import threading

import serial
import serial.tools.list_ports as sp
import time
ANGGOPORT = 'COM13'
LIDARPORT = 'COM14'

gyroYaw = 0
running = False
rev = ''
distanceHuman = 2400


def commAngGo():
    global gyroYaw, running, rev, distanceHuman
    angSer = serial.Serial(
        port=ANGGOPORT,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0.1)  # 0 (non-block) / 0.05(50ms)

    if angSer.isOpen():
        print("conneted with AngGo Pro!")
        running = True
    else:
        running = False
    start = time.time()


    while running:
        end = time.time()
        raw = angSer.readline().decode()
        print(raw)
        if not raw:
            continue

        rev = raw

        if rev.find('<') and rev.find('>'):
            msg = rev.split('<')[-1]
            msg = msg.split('>')[0]
            cmd = msg[0:3]
            param = msg[3:7]
            if cmd == 'GYR':
                gyroYaw = int(param)
                # print('thread yaw : ', gyroYaw)

        if (end-start) > 0.1:
            start = end
            angSer.write(bytes('<ANG0040>', 'ascii'))

            # angSer.write('<ANG0050>'.encode())
            # print('<ANG0010>'.encode())
            print(rev)
            # if distanceHuman < 2500:
            #
            #     if distanceHuman % 2:

            #         angSer.write(bytes('<ANG0050>\n', 'utf-8'))
            #     else:
            #         angSer.write(bytes('<ANG0010>\n', 'utf-8'))

    print("disconneted with AngGo Pro!")
    angSer.close()
    return 0


if __name__=='__main__':
    commAngGo()