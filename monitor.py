msg = "left 793 0.000 	 right 189 0.000  Yaw 48 preYaw 0 t : 48 <GYR0048>\r\n"

if msg.find('<') and msg.find('>'):
    msg = msg.split('<')[-1]
    msg = msg.split('>')[0]
    cmd = msg[0:3]
    param = msg[3:7]
    if cmd == 'GYR':
        gyroYaw = int(param)
        print('thread yaw : ', gyroYaw)

if msg[0] == '<' and msg[8] == '>':
    cmd = msg[1:4]
    param = msg[4:8]
    if cmd == 'GYR':
        gyroYaw = int(param)
        print('thread yaw : ',gyroYaw)