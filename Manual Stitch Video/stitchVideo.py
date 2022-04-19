#Import Packages
import cv2
import numpy as np
import imutils
from scipy.ndimage import rotate

def importVideos(n):
	"""Parameters: n is the number of videos to import
	Function: The videos must be in the same folder as this program. The videos also must be saved in order like 
	cam + number of camera + _output_date_time.h264
	Output: Returns the list of videos, where the first video (e.g. '1.png') is now img[0], and so on"""
	vid = [] #list where videos will be stored
	for i in np.arange(1,int(n)+1): #uses +1 to correct np from not including the end, i is the camera number used to import the numbered video at each iteration.
		cap = cv2.VideoCapture('cam'+str(i)+'_output_2021-12-02_13-53-01.h264')
		vid.append(cap)
	return vid

def stitch_vid(vid, n, frame_width, frame_height, fps):
	"""Parameters: vid is the list of videos, n is the number of videos imported, 
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	and fps is the fps of the input video for the output video.
	Function: This function runs all the function calls in order to actually stitch each frame together in sequence.
	Output: This Function both saves a .avi file to the same folder. The function also outputs a matplot lib of the video."""
	cap = vid[0] #open first video
	fourcc = cv2.VideoWriter_fourcc(*'XVID') #save format for below
	#The file will be saved as whatever name is enclosed in the quoatations of the first parameter of the function below. 
	out = cv2.VideoWriter('comlete_vertical.avi',fourcc, fps, (896,2853)) #print the shape of the final thresholded image and use those values here, will not save unntil values match the otput image shape. # width, height
	if (cap.isOpened()== False): #checks to make sure there is a video to open 
		print("Error opening video stream or file")
	
	while(cap.isOpened()): #while the video is open
	#imports each video and ret. ret is a boolean of whether there is a frame (True) or not (False). 
		ret, frame = cap.read() #video 0 from cam 1
		ret1, frame1 = vid[1].read() # video 1 from cam 2
		ret2, frame2 = vid[3].read() # video 2 from cam 4
		ret3, frame3 = vid[4].read() # video 3 from cam 5
		ret4, frame4 = vid[2].read() # video 4 from cam 3
		ret5, frame5 = vid[6].read() # video 5 from cam 7
		ret6, frame6 = vid[5].read() # video 6 from cam 6
		ret7, frame7 = vid[7].read() # video 7 from cam 8
		ret8, frame8 = vid[8].read() # video 8 from cam 9
		ret9, frame9 = vid[9].read() # video 9 from cam 10
		if ret and ret1 and ret2 and ret3 and ret4 and ret5 and ret6 and ret7 and ret8 and ret9 == True: #check to make sure each video has a frame.
			#each stitch iteration below stitches the desired frame to the desired location in the output
			#Cam 0 reshape
			stitch_0 = vid_edit_0(frame, frame_width, frame_height)
			
			#Cam1 reshape and stitch
			stitch_1 = vid_edit_1(frame1, frame_width, frame_height, stitch_0)
			
			#Cam2 reshape and stitch
			stitch_2 = vid_edit_2(frame2, frame_width, frame_height, stitch_1)
			
			#Cam3 reshape and stitch
			stitch_3 = vid_edit_3(frame3, frame_width, frame_height, stitch_2)
			
			#Cam4 reshape and stitch
			stitch_4 = vid_edit_4(frame4, frame_width, frame_height, stitch_3)
			
			#Cam5 reshape and stitch
			stitch_5 = vid_edit_5(frame5, frame_width, frame_height, stitch_4)
			
			#Cam6 reshape and stitch
			stitch_6 = vid_edit_6(frame6, frame_width, frame_height, stitch_5)
			
			#Cam7 reshape and stitch
			stitch_7 = vid_edit_7(frame7, frame_width, frame_height, stitch_6)
			
			#Cam8 reshape and stitch
			stitch_8 = vid_edit_8(frame8, frame_width, frame_height, stitch_7)
			
			#Cam9 reshape and stitch
			stitch_9 = vid_edit_9(frame9, frame_width, frame_height, stitch_8)
			
			#Threshold
			#calls Threshold() function to remove excess black space from the boarder. 
			output_vid = threshold(stitch_9)
									
			#Save Video
			#this saves the stitched frames together as a frame to the save video file initiated above.
			out.write(output_vid)
			
			#Save Photo
			#To save just an image of all stitched frames remove the '#' from the line below
			#cv2.imwrite('yolo.png', stitch_8)
			
			#Show Preview
			#In order to show a preview of the stitched video via a matplotlib like popout, remove the '#' from the 2 lines below
			#cv2.namedWindow('Frame', cv2.WINDOW_NORMAL) 
			#cv2.imshow('Frame', output_vid)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
	#print(frame_count)
	cap.release()
	cv2.destroyAllWindows()
	
	
def threshold(result): 
	"""Parameters: Result is the resulting frame of all the stitched frame. 
	This function will remove extra black space from the boarders of the stitched frame.
	Output: Outputs the stitched frame with all excess black boarders removed"""
	# transform the panorama image to grayscale and threshold it 
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

	# Finds contours from the binary image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# get the maximum contour area
	c = max(cnts, key=cv2.contourArea)

	# get a bbox from the contour area
	(x, y, w, h) = cv2.boundingRect(c)

	# crop the image to the bbox coordinates
	final = result[y:y + h, x:x + w]
	return final
	
def vid_edit_0(frame, frame_width, frame_height):
	"""Parameters: frame is the frame from the first video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[16,87],[593,78],[27,476],[620,461]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right] #[x,y]
	pts2 = np.float32([[200,0],[840,0],[200,480],[840,480]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_0 = cv2.warpPerspective(frame,M,(frame_width,frame_height)) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_0

def vid_edit_1(frame1, frame_width, frame_height, stitch_0):
	"""Parameters: frame is the frame from the second video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[19,60],[587,70],[0,473],[600,478]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[200,278],[837,274],[200,750],[840,750]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_1 = cv2.warpPerspective(frame1,M,(frame_width,frame_height),stitch_0, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_1
	
def vid_edit_2(frame2, frame_width, frame_height, stitch_1):
	"""Parameters: frame is the frame from the third video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[19,30],[620,37],[3,470],[606,477]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[210,536],[876,523],[204,1010],[858,1010]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_2 = cv2.warpPerspective(frame2,M,(frame_width,frame_height),stitch_1, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_2
	
def vid_edit_3(frame3, frame_width, frame_height, stitch_2):
	"""Parameters: frame is the frame from the fourth video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[11,37],[618,24],[10,436],[626,423]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[185,815],[870,798],[173,1295],[880,1278]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_3 = cv2.warpPerspective(frame3,M,(frame_width,frame_height),stitch_2, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_3
	
def vid_edit_4(frame4, frame_width, frame_height, stitch_3):
	"""Parameters: frame is the frame from the fifth video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[22,19],[630,27],[16,466],[624,474]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[157,1080],[873,1068],[156,1575],[874,1575]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_4 = cv2.warpPerspective(frame4,M,(frame_width,frame_height),stitch_3, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_4
	
def vid_edit_5(frame5, frame_width, frame_height, stitch_4):
	"""Parameters: frame is the frame from the sixth video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[28,25],[632,30],[24,470],[628,475]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[130,1349],[855,1355],[130,1834],[858,1834]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_5 = cv2.warpPerspective(frame5,M,(frame_width,frame_height),stitch_4, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_5
	
def vid_edit_6(frame6, frame_width, frame_height, stitch_5):
	"""Parameters: frame is the frame from the seventh video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[12,7],[634,17],[5,465],[625,475]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[100,1591],[875,1598],[97,2079],[867,2079]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_6 = cv2.warpPerspective(frame6,M,(frame_width,frame_height),stitch_5, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_6
	
def vid_edit_7(frame7, frame_width, frame_height, stitch_6):
	"""Parameters: frame is the frame from the eighth video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[19,19],[615,36],[5,460],[620,473]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[158,1880],[910,1870],[158,2362],[920,2362]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_7 = cv2.warpPerspective(frame7,M,(frame_width,frame_height),stitch_6, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_7
		
def vid_edit_8(frame8, frame_width, frame_height, stitch_7):
	"""Parameters: frame is the frame from the nineth video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[2,9],[605,9],[2,413],[603,413]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[166,2148],[921,2128],[166,2605],[920,2605]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_8 = cv2.warpPerspective(frame8,M,(frame_width,frame_height),stitch_7, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_8
	
def vid_edit_9(frame9, frame_width, frame_height, stitch_8):
	"""Parameters: frame is the frame from the tenth video to be resized for stitching,
	frame_width is the desired width of the video frame, frame_height is the desired height of the video frame,
	Function: Manually warps the input four corners (pts1) to the output four corners (pts2)
	Output: An array of the warped image. """
	pts1 = np.float32([[19,0],[639,24],[1,456],[621,480]]) #the four corners of the desied area in the input img. [top left, top right, bottom left, bottom right]
	pts2 = np.float32([[146,2348],[957,2346],[148,2828],[957,2828]]) #the four corners of the desied area in the output img that the input image will be warped to. [top left, top right, bottom left, bottom right] #[x,y]
	
	M = cv2.getPerspectiveTransform(pts1,pts2) #creates 3x3 matrix M
	
	vid_9 = cv2.warpPerspective(frame9,M,(frame_width,frame_height),stitch_8, borderMode=cv2.BORDER_TRANSPARENT) #warps frame based on M to a canvas of size (cols,rows)
	
	return vid_9

def main():
	n = 10 #the number of videos to be imported 
	
	vid = importVideos(n) #calls to import videos
	
	frame_width = int(vid[1].get(3)*2) #takes the width of the frame from the second video 
	frame_height = int(vid[1].get(4)*n)#takes the height of the frame from the second video 
	#print(frame_width, frame_height)
	fps = int(vid[0].get(5)) #Calculates FPS from the first imported video
	stitch_vid(vid, n, frame_width, frame_height, fps) #Calls the function to stitch video and create save file 
	#NOTE: If you want to change the name of the save file 

main()
	
