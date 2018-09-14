from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/train/', methods=['POST'])
def train_POST():
	import tensorflow as tf
	tf.keras.backend.clear_session()

	incomingStream = request.form.get('stream')
	print(incomingStream)
	splitted = incomingStream.split("[IMAGE_STREAM_DELIMITER]")
	print(len(splitted))

	return "TRUE"

@app.route('/recognize/', methods=['POST'])
def recognize_POST():
	import tensorflow as tf
	tf.keras.backend.clear_session()

	import os
	import io
	import cv2
	import base64 
	import numpy as np
	from PIL import Image

	# Take in base64 string and return PIL image
	def stringToImage(base64_string):
		imgdata = base64.b64decode(base64_string)
		return Image.open(io.BytesIO(imgdata))

	# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
	def toRGB(image):
		return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


	def stringToBase64(s):
		return base64.b64encode(s.encode('utf-8'))


	incomingImage = request.form.get('picture')
	decoded = toRGB(stringToImage(base64.decodestring(stringToBase64(incomingImage))))


	API_KEY = request.form.get('API_KEY')
	# Check if API_KEY is correct
	API_KEY_CORRECT_FLAG = False
	lines = [line.rstrip('\n') for line in open('API_KEYS')]
	for line in lines:
		tempList = line.split()
		if(tempList[0] == API_KEY):
			API_KEY_CORRECT_FLAG = True

	if(API_KEY_CORRECT_FLAG == False):
		return jsonify(
			responseType = "ERROR",
			errorDescription = "API_KEY is invalid!"
		)


	incomingUserName = request.form.get('userName')
	# Check if userName's keras model exists in HOME:API_KEY
	masterPath = "/home/chinmay/MASTERS/Flask/HOME:" + API_KEY + "/" + incomingUserName + ".keras"
	if not os.path.exists(masterPath):
		return jsonify(
			responseType = "ERROR",
			errorDescription = "Username is invalid!"
		)

	def recognizeFace(frame):
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
		import tensorflow as tf
		from tensorflow import keras
		import os
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		
		print("[LOG] -> LOADING MODEL...")
		model = keras.models.load_model(masterPath)
		print("[LOG] -> MODEL LOADED.")

		# Basically finds distance between 2 points
		# Arguments:
		# 	-> tempshape: DLIB's predictor which plots facial landmark points
		# 	-> point1 & point2: Points between which distance is to be found out
		def getDistance(tempshape, point1, point2):
			point1x = tempshape.part(point1).x
			point1y = tempshape.part(point1).y
			point2x = tempshape.part(point2).x
			point2y = tempshape.part(point2).y
						
			dx = (point2x - point1x) ** 2
			dy = (point2y - point1y) ** 2
			distance = math.sqrt(dx + dy)
			return distance


		print("[LOG] -> IMPORTING CASCADE AND DLIB...")
		#Importing Haar cascade and DLIB's facial landmarks detector
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		print("[LOG] -> CASCADE AND DLIB IMPORTED.")

		targetPoints = []

		# Read map.txt -> Can be tuned!
		# map.txt holds a collection of points which will be used to recognize a face
		# map.txt holds a list of pairs between which will define a set of lines to be considered by recognizer
		lines = [line.rstrip('\n') for line in open('map.txt')]
		for line in lines:
			tempList = line.split()
			targetPoints.append(int(tempList[0]))

		# Holds number of ratios as defined by the map
		totalTargets = int(len(targetPoints))

		#Convert the frame to grayscale
		grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#Activating Haar cascade classifier to detect faces
		faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.5, minNeighbors = 5)

		print("[LOG] -> STARTING FACE DETECTION.")
		for(x, y, w, h) in faces :
			print("[LOG] -> FOUND A FACE.")
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

			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				baseLine = getDistance(shapeCopy, 28, 27)
				ratios = []

				for x in targetPoints:
					currentLine = getDistance(shapeCopy, x, 27)
					currentRatio = float(currentLine)/float(baseLine)
					ratios.append(currentRatio)

				foundFlag = 0

				print("[LOG] -> SENDING FACE TO KERAS...")

				# Keras neural net
				npratios = []
				npratios.append(ratios)
				npratios = np.array(npratios)

				import time

				start_time = time.time()
				kerasOutput = model.predict(npratios)
				detection_time = time.time()-start_time
				#print("--- Detection time: %s seconds ---" % (time.time() - start_time))
				#print("\nKERAS O/P: {}".format(kerasOutput))
				maxval = -1
				maxid = -1
				for x in range(0, len(kerasOutput[0])):
					if (kerasOutput[0][x] > maxval):# and (kerasOutput[0][x] > 0.90):
						maxval = kerasOutput[0][x]
						maxid = x

				#print("\nMAX CONFIDENCE FOR : {}".format(usernames[maxid]))

				if(maxid == 2):
					confidence = kerasOutput[0][2]
					print("[LOG] -> RETURNING SUCCESS...")
					return jsonify(
						responseType = "SUCCESS",
						successDescription = "Face recognized successfully",
						confidencePercentage = str(confidence),
						detectionTime = str(detection_time)
					)
				else:
					print("[LOG] -> RETURNING FAILURE...")
					return jsonify(
						responseType = "FAILURE",
						failureDescription = "Face not recognized",
					)
		
		print("[LOG] -> RETURNING FAILURE...")
		return jsonify(
			responseType = "FAILURE",
			failureDescription = "Face not detected",
		)

	# decoded contains image which can be read by opencv
	result = recognizeFace(decoded)
	return result