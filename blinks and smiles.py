import cv2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import math
from PIL import Image
from subprocess import call
import os
import threading
import time
import os
from scipy.spatial import distance as dist

#Importing Haar cascade and DLIB's facial landmarks detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture (webcam)
video = cv2.VideoCapture(0)

blinks = 0

def smile_aspect_ratio(mouth):
	A = dist.euclidean(mouth[3], mouth[9])
	B = dist.euclidean(mouth[2], mouth[10])
	C = dist.euclidean(mouth[4], mouth[8])
	L = (A+B+C)/3
	D = dist.euclidean(mouth[0], mouth[6])
	mar=L/D
	return mar

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

counter = 0

while(True):
	ret, frame = video.read()
	cv2.imshow('Original video feed', frame)

	#Convert the frame to grayscale
	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Activating Haar cascade classifier to detect faces
	faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.5, minNeighbors = 5)

	for(x, y, w, h) in faces :

		pillowImage = Image.fromarray(frame[y:y+h, x:x+w])
		#Resizing dimensions
		resizedHeight = 300
		resizedWidth = 300
		######
		faceCropped = np.array(pillowImage.resize((resizedHeight, resizedWidth), Image.ANTIALIAS))

		#Initialize dlib's rectangle to start plotting points over shape of the face
		dlibRect = dlib.rectangle(0, 0, resizedHeight, resizedWidth)
		shape = predictor(cv2.cvtColor(faceCropped, cv2.COLOR_BGR2GRAY), dlibRect)
		shapeCopy = shape
		shape = face_utils.shape_to_np(shape)

		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
 
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0


		if ear < 0.25:
			counter += 1		

		if(counter > 2):
			blinks = blinks+1
			counter = 0
			print("BLINK! (Eye aspect ratio : {} Blink counter : {})".format(ear, blinks))

		(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		mouth= shape[mStart:mEnd]
		sar = smile_aspect_ratio(mouth)

		if sar < 0.20:
			print("SMILE!")

			
	if cv2.waitKey(20) & 0xFF == ord('q') :
		break

video.release()
cv2.destroyAllWindows()