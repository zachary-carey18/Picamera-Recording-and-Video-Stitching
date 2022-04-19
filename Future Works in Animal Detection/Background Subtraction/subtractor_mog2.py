import cv2
import numpy as np

cap = cv2.VideoCapture("comlete_vertical_single.avi")

subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=800, detectShadows=True)

while True:
	_, frame = cap.read()

	mask = subtractor.apply(frame)
	
	cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
	cv2.imshow("Frame", frame)
	cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
	cv2.imshow("mask", mask)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
