import cv2
import numpy as np
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("comlete_vertical_single.avi")

# Background Subtraction
object_detector = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=300, detectShadows=True)

#Color detection

while True:
	ret, frame = cap.read()
	height, width, _ = frame.shape
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Extract Region of interest
	#roi = frame[340: 720,500: 800]

	# 1. Object Detection
	mask1 = object_detector.apply(gray_frame)
	
	low_black = np.array([0])
	high_black = np.array([60])
	black_mask = cv2.inRange(gray_frame, low_black, high_black)
	
	mask = mask1 & black_mask    
	
	#_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
	_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	detections = []
	for cnt in contours:
		# Calculate area and remove small elements
		area = cv2.contourArea(cnt)
		if 20 < area < 50:
			#cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
			x, y, w, h = cv2.boundingRect(cnt)


			detections.append([x, y, w, h])

	# 2. Object Tracking
	boxes_ids = tracker.update(detections)
	for box_id in boxes_ids:
		x, y, w, h, id = box_id
		cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

	#cv2.imshow("roi", roi)
	cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
	cv2.imshow("Frame", frame)
	cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
	cv2.imshow("Mask", mask)
	cv2.namedWindow('Mask1', cv2.WINDOW_NORMAL)
	cv2.imshow("Mask1", mask1)
	cv2.namedWindow('Black_Mask', cv2.WINDOW_NORMAL)
	cv2.imshow("Black_Mask", black_mask)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
