# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ctypes
import math
import sys
import serial
import serial.tools.list_ports as sp
import cv2
from multiprocessing import Process, Queue
from threading import Thread
import time
import numpy as np
from ctypes import cdll
import mediapipe as mp
from matplotlib import pyplot as plt
import ckwrap
import kmeans1d
from game2 import *
# 1,3,2,4,5
interface = 2

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
BG_COLOR = (0, 0, 0) # gray

ANGGOPORT = 'COM13'
LIDARPORT = 'COM14'
# LIDARPORT = '/dev/ttyACM1' #0x5740
# ANGGOPORT = '/dev/ttyACM0'
DisThr = 6000
shake = False

BAUDRATE    = 10000000
startbit    = 0xF5
packet      = []
crop = []

img = np.array([])
ampImg = np.array([])
frame = np.array([])
ampFrame = np.array([])
crclib = cdll.LoadLibrary('./crc.dll')
# crclib = cdll.LoadLibrary('./crc.so')
gyroYaw = 0
running = 0
rev = ''
humanpos = []
distanceHuman = 8000
class CrcCalc:
    def __init__(self):
        self.initValue = 0xFFFFFFFF
        self.xorValue = 0x00000000
        self.polynom = 0x4C11DB7

    def calcCrc32Uint32(self,crc, data):
        crc = crc ^ data
        for i in range(32):
            if crc & 0x80000000:
                crc = (crc << 1) ^ self.polynom
            else:
                crc = (crc << 1)
            crc = crc & 0xFFFFFFFF
        return crc

    def calcCrc32_32(self,data):
        crc = self.initValue
        for i in data:
            crc = self.calcCrc32Uint32(crc, i)
        return crc ^ self.xorValue

    def checkPck(self, packet):
        result = self.calcCrc32_32(packet[:-4])
        bit4 = (result >> 24) & 0xff
        bit3 = (result >> 16) & 0xff
        bit2 = (result >> 8) & 0xff
        bit1 = result & 0xff
        cmp = (bit4 == packet[-1]) & (bit3 == packet[-2]) & (bit2 == packet[-3]) & (bit1 == packet[-4])
        return cmp

def checkPck( packet,result):
    bit4 = (result >> 24) & 0xff
    bit3 = (result >> 16) & 0xff
    bit2 = (result >> 8) & 0xff
    bit1 = result & 0xff
    cmp = (bit4 == packet[-1]) & (bit3 == packet[-2]) & (bit2 == packet[-3]) & (bit1 == packet[-4])
    return cmp

def onMouse(event,x,y,flags,param):
    global frame, ampFrame, img, ampImg,crop
    if event == cv2.EVENT_LBUTTONDOWN:
        print("좌표 ",x,y)

        print(frame[y][x] * 8000 / 255)

def mediapipeHuman(selfie_segmentation,input_frame,threshold):
    bg_image = None
    image = input_frame
    image = cv2.merge([image, image, image])
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = selfie_segmentation.process(image)  # RGB값을 필요로 함
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack(  # BGR값을 필요로 함
        (results.segmentation_mask,) * 3, axis=-1) > threshold
    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    if bg_image is None:
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
    # output_image = np.where(condition.squeeze()[:,:,0], image, bg_image)
    output_image = np.where(condition, image, bg_image)
    
    contours, hierarchy = cv2.findContours(output_image[:, :, 0], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    input_frame = cv2.merge([input_frame, input_frame, input_frame])

    if contours:
        idx = np.argmax([cv2.contourArea(cnt) for cnt in contours])
        x, y, w, h = cv2.boundingRect(contours[idx])
        input_frame = cv2.rectangle(input_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return input_frame,output_image,[x,y,w,h]
    else:
        return cv2.merge([ampFrame, ampFrame, ampFrame]),[],None

def commAngGo():
    global gyroYaw,running,rev,distanceHuman,humanpos,DisThr,shake
    angSer = serial.Serial(
        port=ANGGOPORT,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0.1)  # 0 (non-block) / 0.1(100ms)
    
    if angSer.isOpen():
        print("conneted with AngGo Pro!")
        running = True
    else:
        running = False
    avoidHuman = False
    start = time.time()
    cnt = 0
    targetYaw = 0
    xpos = 80
    ypos = 30
    while running:
        raw = angSer.readline().decode()
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
        end = time.time()
        if (end - start) > 0.1:
            # print('thread yaw : ', gyroYaw)
            start = end

            if distanceHuman < DisThr and avoidHuman is False:
                avoidHuman = True
                xpos, ypos = humanpos
                angSer.write(bytes('<LIN0030>', 'ascii'))
                if interface == 2 or interface == 4:
                    shake = True


            if avoidHuman and cnt < 25:
                cnt = cnt + 1
                if xpos < 80:
                    angSer.write(bytes('<ANG0030>', 'ascii'))
                if xpos >= 80:
                    angSer.write(bytes('<ANG0330>', 'ascii'))
            if avoidHuman and cnt < 55:
                cnt = cnt + 1
                angSer.write(bytes('<LIN0035>', 'ascii'))
            elif avoidHuman and cnt < 120:
                cnt = cnt + 1
                angSer.write(bytes('<LIN00035>', 'ascii'))
                angSer.write(bytes('<ANG0000>', 'ascii'))
            else:
                cnt = 0
                xpos = 80
                ypos = 30
                avoidHuman = False
                shake = False






    print("disconneted with AngGo Pro!")
    angSer.close()
    return 0

def main():
    global frame, ampFrame,img,ampImg,crop,running,rev,distanceHuman,humanpos,interface,DisThr,shake
    connected = []
    inf = Queue()
    ProcessFacial = Process(target=timaface, args=(inf,))
    ThreadAngGo = Thread(target=commAngGo)
    ProcessFacial.start()
    ThreadAngGo.start()

    serList = sp.comports()
    for i in serList:
        connected.append(i.device)
    print("*"*50)
    print("connected serial port", connected)
    print("*"*50)

    ser = serial.Serial(
        port=LIDARPORT,
        baudrate=BAUDRATE,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1) # 0 (non-block) / 0.1(100ms)
        
    if ser.isOpen():
        print("Serial port is opened")
        running = True
    else:
        running = False

    GetDisPck = [0xF5, 0X20,
                 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00,
                 0x62, 0xAC, 0xA8, 0xCC]

    GetDisAmp = [0xF5, 0X22,
                 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00,
                 0xE9, 0xDF, 0xE8, 0x9E]

    setAuto3D = [0xF5, 0x00, 0xFF, 0x00,
                 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x50, 0x7D,
                 0x12, 0xEB]

    setExposure = [0xF5, 0x00, 0x00, 0x1E,
                   0x00, 0x00, 0x00, 0x00,
                   0x00, 0x00, 0x47, 0x70,
                   0xEC, 0xC0]

    stopstream = [0xF5, 0x28, 0x00, 0x00,
                  0x00, 0x00, 0x00, 0x00,
                  0x00, 0x00, 0xF9, 0x7F,
                  0x68, 0x81]

    cv2.namedWindow('distance')
    cv2.namedWindow('amplitude')
    cv2.setMouseCallback('distance', onMouse)

    # hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # time = [10,100,500,1000]
    # for i in enumerate(time):
    #     setAuto3D[1] = i
    #     setAuto3D[2] = (time[i] & 0xff)
    #     setAuto3D[3] = ((time[i] >> 8) & 0xff)
    #     setAuto3D
    
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


    # ser.write(bytearray(setExposure))
    # ser.write(bytearray(setAuto3D))
    a = time.time()
    b = a
    # time.sleep(1)
    while running:

        ser.write(bytearray(GetDisAmp))
        rxdata = ser.read()
        a = time.time()
        # print("fps : ", round(1 / (a - b)))

        if rxdata == b'\xfa':
            
            data = [0xfa]
            data.extend(ser.read(3+38480+4))
            arr = (ctypes.c_uint8 * len(data[:-4]))(*data[:-4])
            result = crclib.calcCrc32_32(arr, ctypes.c_uint32(len(data[:-4])))
            result = ctypes.c_uint32(result).value
            result = checkPck(data, result)

            if result == 1:
                frame = np.array(data[84:-4])
                l = len(frame)
                try:
                    img = ((frame[1::4] & 0xff) << 8) | (frame[0::4] & 0xff)
                    ampImg = ((frame[3::4] & 0xff) << 8) | (frame[2::4] & 0xff)
                except:
                    continue

                img = np.array(img,dtype=float)
                ampImg = np.array(ampImg, dtype=float)


                img = np.where(img > 8000,8000,img) / 8000 * 255
                ampImg = np.where(ampImg > 2000, 2000, ampImg) / 250 * 255

                frame = np.reshape(img, (60, 160))
                ampFrame = np.reshape(ampImg, (60, 160))

                ampFrame = np.where(frame >= (DisThr/8000*255), 0, ampFrame)

                frame = cv2.resize(frame, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
                ampFrame = cv2.resize(ampFrame, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

                frame = frame.astype('uint8')
                ampFrame = ampFrame.astype('uint8')
                ampFrame = cv2.equalizeHist(ampFrame)


                ' ---- Media Pipe ----'
                ampFrame, segmented, loc = mediapipeHuman(selfie_segmentation,ampFrame,0.2)
                if len(segmented) > 1:
                    segframe = np.where(segmented[:,:,0] == BG_COLOR[0], 255 , frame)
                if loc :
                    x,y,w,h = loc
                    cropImg = np.array(segframe[y:y+h,x:x+w],dtype=float)
                    crop = cropImg * 8000 / 255
                    ampFrame = cv2.circle(ampFrame,(x+w//2,y+h//2),5,(255,0,0),-1)
                    ch,cw = crop.shape

                    cropflat = crop.ravel()
                    lenCF = len(cropflat)
                    numCluster = 5
                    buckets = [[],[],[],[],[]]
                    clusters, centroids = kmeans1d.cluster(cropflat,numCluster)
                    for i in range(lenCF):
                        buckets[clusters[i]].append(cropflat[i])
                    buckets = buckets[:-1]
                    freq = [len(i) for i in buckets]
                    idx = np.argmax(freq)
                    distanceHuman = np.mean(buckets[idx])

                    size = frame.shape
                    pupilPos = x + w // 2 #4 - math.ceil((x + w // 2) / size[1] * 3)
                    if pupilPos < 70:
                        pupilPos = 10
                    elif pupilPos > 90:
                        pupilPos = 190
                    humanpos = [x+w//2, y+h//2]

                else:
                    pupilPos = 80
                    distanceHuman = 8000
                    humanpos = [None,None]

                print(rev, " distance : ", distanceHuman, "interfa : ",interface)

                heading = -1*np.clip(gyroYaw, -30, 30)

                if interface == 2: # eye track
                    inf.put([0, pupilPos, np.clip(distanceHuman,500,3000),shake])  # x,y,z
                elif interface == 3: # heading
                    inf.put([heading, 80, np.clip(distanceHuman, 500, 3000),False])  # x,y,z
                elif interface == 4: # both
                    inf.put([heading, pupilPos, np.clip(distanceHuman, 500, 3000),shake])  # x,y,z
                elif interface == 5: # both
                    inf.put([0, 80, np.clip(distanceHuman, 500, 3000)])  # x,y,z

                colordis = cv2.applyColorMap(255-frame, cv2.COLORMAP_BONE)
                cv2.imshow("distance", colordis)
                cv2.imshow("amplitude", ampFrame)
                key = cv2.waitKey(1) & 0xff
                if key == ord('1'):
                    interface = 1
                elif key == ord('2'):
                    interface = 2
                elif key == ord('3'):
                    interface = 3
                elif key == ord('4'):
                    interface = 4

                if key == ord('c'):
                    numCluster = 5
                    km = ckwrap.ckmeans(crop.ravel(), numCluster)
                    print(km.labels)
                    # [0 0 0 0 1 1 1 2 2]
                    buckets = [[],[],[],[],[]]
                    for i in range(len(crop.ravel())):
                        buckets[km.labels[i]].append(crop.ravel()[i])
                    freq = [len(i) for i in buckets]
                    idx = np.argmax(freq)
                    print(np.mean(buckets[idx]))
                    for i in buckets:
                        print("분산 : ", np.var(i), "표준편차 : ", np.std(i), "평균 : ", np.mean(i), "빈도 : ",len(i))
                    plt.subplot(2,2,1)
                    plt.hist(crop.ravel(),bins=100,range=(1,8000))
                    plt.subplot(2,2,2)
                    plt.plot(buckets[0])
                    plt.subplot(2, 2, 3)
                    plt.plot(buckets[1])
                    plt.subplot(2, 2, 4)
                    plt.plot(buckets[2])
                    plt.show()

                elif key == ord('k'):
                    ori = np.array(frame,dtype=float) * 8000 / 255
                    plt.hist(ori.ravel(), bins=1000, range=(1, 8000))
                    plt.show()

                elif key == 27:
                   running = False
                   
                   break

                elif key == ord('s'):
                    cv2.imwrite('amplitude.png',ampFrame)
                    cv2.imwrite('distance.png',colordis)
                img = np.array([])
                ampImg = np.array([])
    ser.write(bytearray(stopstream))
    cv2.destroyAllWindows()
    ser.close()
    ProcessFacial.join()
    ThreadAngGo.join()

    print("program terminated")

if __name__ == '__main__':
    # sys.stdout = open('stdout.txt', 'w')
    main()
    # sys.stdout.close()
