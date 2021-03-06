# Change from game1.py
# 1. FullScreen (no X btn) (For closing program, press key "ALT") (O)
# 2. Color change to navy theme (X)

# Import and initialize the pygame library

# from tkinter.messagebox import NO
import pygame
from pygame.locals import * # Daniel, FullScreen
import random
import time
import ctypes # Daniel, FullScreen
from screeninfo import get_monitors



#blinking time func
def time_rand():
    r = 0.0
    for i in range(3):
        r += random.randint(0,10)
    r /= 5
    return r

#remapping function
def remap(x, in_min, in_max, out_min, out_max):
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class Eye:
    x = 0
    y = 0
    width = 0
    height = 0
    roundness = 0
    pupil_r = 0
    pupil_height = 0
    eyelid_roudness = 0
    x_pupil = 0
    # papameters of constructor
    def __init__(self, x, y, w, h, roundness, pupil_r, pupil_height, eyelid_roundness,x_pupil):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.roundness = roundness
        self.pupil_r = pupil_r
        self.pupil_height = pupil_height
        self.eyelid_roudness = eyelid_roundness
        self.x_pupil = x_pupil

# Run until the user asks to quit
def timaface(inf):
	pygame.init()
	clock = pygame.time.Clock()
	# screen = pygame.display.set_mode([1920, 515])  # 전체화면으로 전환
	bardisplay = False
	displynum = 1
	for m in get_monitors():
		print(m)
		if m.width == 1920 and m.height == 515:
			bardisplay = True
			displynum = int(m.name[-1])

	if bardisplay:
		screen = pygame.display.set_mode([1920, 515],display=1)
	else :
		screen = pygame.display.set_mode([1920, 515] )

	blinkingPeriod = 3.00
	startTime = time.time()

	#### INPUT VARIABLEs
	xInput = 2 # 1 2 3
	pupilXInput = 2  # 1 2 3 NEW
	distanceValue = 0 #0~10
	lookUp = False

	#shake variables shake is here ///
	shake = True
	randomness = [1,1]
	shakeVariationX = 0
	shakeVariationY = 0

	#Eye variables
	distanceBetweenEyes = int(2*1920/16)
	centerOfEye = 1920/2
	pupilSizeDifference = 0
	xLocation = 1920/2
	xLocationTarget = 0
	xPupil = 0
	xTargetPupil = 0

	#blinking data
	blink = False
	blSp = 0
	blinkingSpeed = 0

	leftEye = Eye(centerOfEye-distanceBetweenEyes, 515/2, 300, 260, 300/5, 300/4, 0, 0, 300/2)
	rightEye = Eye(centerOfEye+distanceBetweenEyes, 515/2, 300, 260, 300/5, 300/4, 0, 0, 300/2)

	distance = 0
	heading = 0
	humanXpos = 1920 // 2
	time.sleep(1)
	running = True
	while running:
		# time.sleep(0.001)
		dt = clock.tick(60)
		if inf is not None:
			if not inf.empty():
				data = inf.get()
				if data:
					heading, humanXpos, distance, shake = data
					xLocationTarget = remap(heading,-30,30,0,1920)  # (1,2,3)
					# pupilXInput = 160-humanXpos
					# print(data)


		#### BASIC SET UP
		screen.fill(color='#000080')


		#### KEY EVENTS FOR DEBUGGING
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN :
				if event.mod == pygame.KMOD_NONE:
					print('No modifier keys were in a pressed state when this '
												'event occurred.')
				else:
					if event.mod & pygame.KMOD_LSHIFT:
						xInput -= 1
					if event.mod & pygame.KMOD_RSHIFT:
						xInput += 1
					if event.mod & pygame.KMOD_CTRL:
						shake = not shake
					if event.mod & pygame.KMOD_ALT:
						 running = False

		#change here

		#### LOGIC
		leftEye.x = centerOfEye - distanceBetweenEyes
		rightEye.x = centerOfEye + distanceBetweenEyes

		# xInput = 3
		# if xInput == 1:
		# 	xLocationTarget = 0
		# elif xInput == 2:
		# 	xLocationTarget = 1920/2
		# else:
		# 	xLocationTarget = 1920


		#speed of movement eye between pre-set positions
		speed = remap(abs(xLocationTarget-xLocation), 0, 1920, 0, 40)
		if xLocation > xLocationTarget:
			xLocation -= speed
		else:
			xLocation += speed

	#	xLocation = pygame.mouse.get_pos()[0]
		# #pupil set
		# if pupilXInput == 1:
		# 	xTargetPupil = 0
		# elif pupilXInput == 2:
		# 	xTargetPupil = 1920 / 2
		# else:
		# 	xTargetPupil = 1920

		xTargetPupil = pygame.mouse.get_pos()[0]
		# xTargetPupil = remap(pupilXInput, 0, 160, 0, 1920)

		# #speed of pupil
		xPupilspeed = remap(abs(xPupil - xTargetPupil), 0, 1920, 0, 60)
		if xPupil > xTargetPupil:
			xPupil -= xPupilspeed
		else:
			xPupil += xPupilspeed

		leftEye.x_pupil = remap(xPupil, 0, 1920, leftEye.width / 2,
								leftEye.width)
		rightEye.x_pupil = remap(xPupil, 0, 1920, rightEye.width / 2,
								 rightEye.width )

		# leftEye.x_pupil = remap(xPupil, 0, 1920, leftEye.pupil_r, leftEye.width - leftEye.pupil_r)
		# rightEye.x_pupil = remap(xPupil, 0, 1920, rightEye.pupil_r, rightEye.width - rightEye.pupil_r)

		centerOfEye = remap(xLocation, 0, 1920, leftEye.width/2+distanceBetweenEyes, 1920 - rightEye.width/2 - distanceBetweenEyes)

		if xLocation >= 1920/2:
			leftEye.width = 300 
			leftEye.height = 260
			rightEye.width = remap(xLocation, 1920/2, 1920, 300, 200)
			#rightEye.height = remap(xLocation, 1920/2, 1920, 230, 280)
			distanceBetweenEyes = remap(xLocation, 1920/2, 1920, (2*1920/16), leftEye.width/2)
		else :
			leftEye.width = remap(xLocation, 0, 1920/2, 200, 300)
			#leftEye.height = remap(xLocation, 0, 1920/2, 280, 230)
			rightEye.width = 300
			rightEye.height = 260
			distanceBetweenEyes = remap(xLocation, 0, 1920/2, rightEye.width/2, (2*1920/16))

		# pupilSizeDifference = remap(distanceValue, 0, 10, 0, 20)
		# pupilSizeDifference = remap(3000-distance, 300, 3000 , 5, 30) # for test

		# leftEye.pupil_height = remap(pygame.mouse.get_pos()[0], 0, 1920, 0, leftEye.height)


		#shake condition
		if shake:
			shakeVariationX = random.randint(-randomness[0],+randomness[0])
			shakeVariationY = random.randint(-randomness[1],+randomness[1])
		else:
			shakeVariationX = 0
			shakeVariationY = 0

		#### DRAWING PART
		# Fill Bg
		# Draw Eye
		pygame.draw.rect(screen, (255, 255, 255), (int(leftEye.x)-leftEye.width/2, int(leftEye.y)-(leftEye.height-130), int(leftEye.width), int(leftEye.height)), 0, int(leftEye.roundness))
		pygame.draw.rect(screen, (255, 255, 255), (int(rightEye.x)-rightEye.width/2, int(rightEye.y)-(rightEye.height-130), int(rightEye.width), int(rightEye.height)), 0, int(rightEye.roundness))
		
		# Draw Pupil
		pygame.draw.rect(screen, ('#000080'), 
			(int(leftEye.x) - leftEye.width / 2 + shakeVariationX + leftEye.x_pupil - leftEye.width / 2, 
				int(leftEye.y) + shakeVariationY - int(leftEye.pupil_r), 
				leftEye.width / 2 , 
				int(leftEye.pupil_r) * 2), 
			0, int(leftEye.roundness))
			
		
		pygame.draw.rect(screen, ('#000080'), 
			(int(rightEye.x) - rightEye.width / 2 + shakeVariationX + rightEye.x_pupil - rightEye.width / 2, 
				int(rightEye.y) + shakeVariationY - int(rightEye.pupil_r), 
				rightEye.width / 2 , 
				int(rightEye.pupil_r) * 2), 
			0, int(rightEye.roundness))


		#Draw blinking
		if (round((time.time() -startTime), 2) >= blinkingPeriod) and not blink:
    		# blinkingPeriod = time_rand()
			# print(round((time.time()-startTime ),2))
			startTime = time.time()
			blinkingPeriod = time_rand()
			# print("time set", blinkingPeriod)
			blink = True
		
		# print(blink)
		if blink:
			if blinkingSpeed > leftEye.height/2:
				blSp = -10
			elif blinkingSpeed <= 0:
				blSp = 10
			if blSp < 0 and blinkingSpeed<=10 :
				blink = False
				blinkingSpeed=0
				blSp = 0

			blinkingSpeed += blSp
			pygame.draw.rect(screen,('#000080'), (int(leftEye.x)-leftEye.width/2, int(leftEye.y)-(leftEye.height-130), int(leftEye.width), blinkingSpeed))
			pygame.draw.rect(screen,('#000080'), (int(leftEye.x)-leftEye.width/2, int(leftEye.y)-(leftEye.height/2-130)+(leftEye.height/2 - blinkingSpeed), int(leftEye.width), blinkingSpeed))
			pygame.draw.rect(screen,('#000080'), (int(rightEye.x)-rightEye.width/2, int(rightEye.y)-(rightEye.height-130), int(rightEye.width), blinkingSpeed))
			pygame.draw.rect(screen,('#000080'), (int(rightEye.x)-rightEye.width/2, int(rightEye.y)-(rightEye.height/2-130)+(rightEye.height/2 - blinkingSpeed), int(rightEye.width), blinkingSpeed))


		# Flip the display
		pygame.display.flip()
		

	# Done! Time to quit.
	pygame.quit()

if __name__ == "__main__":
	timaface(None) # Hi