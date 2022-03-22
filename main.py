# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ctypes
import sys
import serial
import serial.tools.list_ports as sp
import cv2
from multiprocessing import Process, Queue
import time
import numpy as np
from ctypes import cdll
import mediapipe as mp
from matplotlib import pyplot as plt
import ckwrap
import pygame
import random
from game1 import *
from sklearn.cluster import MeanShift, estimate_bandwidth


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
BG_COLOR = (0, 0, 0) # gray

ANGGOPORT = 'COM13'
LIDARPORT = 'COM14'
BAUDRATE    = 10000000
startbit    = 0xF5
packet      = []
crop = []



img = np.array([])
ampImg = np.array([])
frame = np.array([])
ampFrame = np.array([])
crclib = cdll.LoadLibrary('./crc.dll')


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



def mediapipeHuman(input_frame,model_select,threshold):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=model_select) as selfie_segmentation:
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



def main():
    global frame, ampFrame,img,ampImg,crop
    connected = []
    inf = Queue
    ProcessFacial = Process(target=timaface, args=(inf,))
    ProcessFacial.start()
    serList = sp.comports()
    for i in serList:
        connected.append(i.device)
    print("connected serial port", connected)

    ser = serial.Serial(
        port=LIDARPORT,
        baudrate=BAUDRATE,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0.1) # 0 (non-block) / 0.1(100ms)

    if ser.isOpen():
        print("Serial port is opened")

    angSer = serial.Serial(
        port=ANGGOPORT,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0.1)  # 0 (non-block) / 0.1(100ms)


    if angSer.isOpen():
        print("conneted with AngGo Pro!")


    GetDisPck = [0xF5, 0X20,
                 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00,
                 0x62, 0xAC, 0xA8, 0xCC]

    GetDisAmp = [0xF5, 0X22,
                 0x00, 0x00, 0x00, 0x00,
                 0x00, 0x00, 0x00, 0x00,
                 0xE9, 0xDF, 0xE8, 0x9E]

    setAuto3D = [0xF5, 0x00, 0x00, 0x1E,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x47, 0x70,
                0xEC, 0xC0]

    finding = True
    data = np.array([])
    cur = time.time()
    center = []
    ll = 160 * 60 * 2
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

    while finding:
        ser.write(bytearray(GetDisAmp))
        rxdata = ser.read()

        # msg = angSer.readline()
        # print(msg)

        if rxdata == b'\xfa':
            tp = time.time()
            data = [0xfa]
            data.extend(ser.read(3+38480+4)) # task time : (측정값 : 0.15~0.17sec) == (이론값 : 0.15 =  19280 bytes / 10M bps
            # data.extend(ser.read(3+19280+4))  # task time : (측정값 : 0.15~0.17sec) == (이론값 : 0.15 =  19280 bytes / 10M bps

            arr = (ctypes.c_uint8 * len(data[:-4]))(*data[:-4])
            result = crclib.calcCrc32_32(arr, ctypes.c_uint32(len(data[:-4])))
            result = ctypes.c_uint32(result).value
            result = checkPck(data, result)

            if result == 1:
                frame = np.array(data[84:-4])
                l = len(frame)
                img = ((frame[1::4] & 0xff) << 8) | (frame[0::4] & 0xff)
                ampImg = ((frame[3::4] & 0xff) << 8) | (frame[2::4] & 0xff)

                img = np.array(img,dtype=float)
                ampImg = np.array(ampImg, dtype=float)
                img = np.where(img > 8000,8000,img) / 8000 * 255
                ampImg = np.where(ampImg > 2000, 2000, ampImg) / 1000 * 255

                frame = np.reshape(img, (60, 160))
                ampFrame = np.reshape(ampImg, (60, 160))

                frame = cv2.resize(frame, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                ampFrame = cv2.resize(ampFrame, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                frame = frame.astype('uint8')
                ampFrame = ampFrame.astype('uint8')
                ampFrame = cv2.equalizeHist(ampFrame)

                ''' rotate image'''
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # ampFrame = cv2.rotate(ampFrame, cv2.ROTATE_90_CLOCKWISE)

                ''' HOG + SVM detector'''
                # boxes, weights = hog.detectMultiScale(ampFrame, winStride=(5, 5),padding=(2,2),finalThreshold=0.05)
                # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
                # for (xA, yA, xB, yB) in boxes:
                #     # display the detected boxes in the colour picture
                #     cv2.rectangle(ampFrame, (xA, yA), (xB, yB),
                #                   (255, 255, 255), 2)

                ' ---- Media Pipe ----'
                ampFrame,segmented,loc = mediapipeHuman(ampFrame,1,0.5)

                if len(segmented) > 1:
                    frame = np.where(segmented[:,:,0] == BG_COLOR[0], 0 , frame)
                if loc :
                    x,y,w,h = loc
                    cropImg = np.array(frame[y:y+h,x:x+w],dtype=float)
                    crop = cropImg * 8000 / 255
                    ampFrame = cv2.circle(ampFrame,(x+w//2,y+h//2),5,(255,0,0),-1)
                    ch,cw = crop.shape

                    cropflat = crop.ravel()
                    lenCF = len(cropflat)
                    numCluster = 3
                    if lenCF > 10:
                        try:
                            km = ckwrap.ckmeans(cropflat,numCluster)
                            buckets = [[],[],[]]
                            for i in range(lenCF):
                                buckets[km.labels[i]].append(cropflat[i])
                            for i,cluster in enumerate(buckets):
                                if np.var(cluster) == 0:
                                    del buckets[i]

                            freq = [len(i) for i in buckets]
                            idx = np.argmax(freq)
                            print("중간 픽셀 거리 : ", crop[ch // 2][cw // 2], end="\t")
                            print("kmenas 평균 : ",np.mean(buckets[idx]))

                        except:
                            pass




                cv2.imshow("distance", frame)
                cv2.imshow("amplitude", ampFrame)

                key = cv2.waitKey(1) & 0xff

                if key == ord('c'):
                    km = ckwrap.ckmeans(crop.ravel(), numCluster)
                    print(km.labels)
                    # [0 0 0 0 1 1 1 2 2]
                    buckets = [[],[],[]]
                    for i in range(len(crop.ravel())):
                        buckets[km.labels[i]].append(crop.ravel()[i])
                    freq = [len(i) for i in buckets]
                    idx = np.argmax(freq)
                    print(np.mean(buckets[idx]))
                    for i in buckets:
                        print("분산 : ", np.var(i), "표준편차 : ", np.std(i), "평균 : ", np.mean(i))
                    plt.subplot(2,2,1)
                    plt.hist(crop.ravel(),bins=100,range=(1,8000))
                    plt.subplot(2,2,2)
                    plt.plot(buckets[0])
                    plt.subplot(2, 2, 3)
                    plt.plot(buckets[1])
                    plt.subplot(2, 2, 4)
                    plt.plot(buckets[2])
                    plt.show()
                elif key == ord('a'):
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')
                    y,x = crop.shape
                    x = list(range(x))
                    y = list(range(y))
                    print(x,y)

                    for _x in x:
                        for _y in y:
                            z = crop[_y][_x]
                            ax.scatter(_x,_y,z,c='b')
                    plt.show()

                elif key == ord('a'):
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')
                    y,x = crop.shape
                    x = list(range(x))
                    y = list(range(y))
                    print(x,y)

                    for _x in x:
                        for _y in y:
                            z = crop[_y][_x]
                            ax.scatter(_x,_y,z,c='b')
                    plt.show()

                elif key == ord('k'):
                    ori = np.array(frame,dtype=float) * 8000 / 255
                    plt.hist(ori.ravel(), bins=100, range=(1, 8000))
                    plt.show()


                elif key == 27:
                   finding = False
                   ProcessFacial.terminate()
                   break

                # elif key == ord('s'):
                #     cv2.imwrite('test.png')
                img = np.array([])
                ampImg = np.array([])

    cv2.destroyAllWindows()
    ser.close()

if __name__ == '__main__':
    main()
