import cv2
import numpy as np

cap = cv2.VideoCapture("comlete_vertical_mixed.avi")

while True:
	_, frame = cap.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# black color
	low_black = np.array([10])
	high_black = np.array([60])
	black_mask = cv2.inRange(gray_frame, low_black, high_black)
	Black = cv2.bitwise_and(frame, frame, mask=black_mask)

	# Blue color
	#low_blue = np.array([94, 80, 2])
	#high_blue = np.array([126, 255, 255])
	#blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
	#blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

	# Green color
	#low_green = np.array([40, 88, 55])
	#high_green = np.array([70, 255, 255])
	#green_mask = cv2.inRange(hsv_frame, low_green, high_green)
	#green = cv2.bitwise_and(frame, frame, mask=green_mask)

	# Every color except white
	#low = np.array([0, 42, 0])
	#high = np.array([179, 255, 255])
	#mask = cv2.inRange(hsv_frame, low, high)
	#result = cv2.bitwise_and(frame, frame, mask=mask)
	
	
	cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
	cv2.imshow("Frame", frame)
	cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
	cv2.imshow("Mask", black_mask)
	cv2.namedWindow('gray_Frame', cv2.WINDOW_NORMAL)
	cv2.imshow("gray_Frame", gray_frame)
	cv2.namedWindow('Black', cv2.WINDOW_NORMAL)
	cv2.imshow("Black", Black)
	#cv2.imshow("Blue", blue)
	#cv2.namedWindow('Green', cv2.WINDOW_NORMAL)
	#cv2.imshow("Green", green)
	#cv2.imshow("Result", result)

	key = cv2.waitKey(1)
	if key == 27:
		break
