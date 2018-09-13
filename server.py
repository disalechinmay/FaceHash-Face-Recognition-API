from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def hello_world():
	import tensorflow as tf
	tf.keras.backend.clear_session()

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
	#cv2.imwrite("/home/chinmay/MASTERS/myGENImage.jpg", decoded)
	#cv2.imshow("TEST", decoded)


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
		
		print("LOADING MODEL...")
		model = keras.models.load_model('kerasFaceHash')
		print("MODEL LOADED.")

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


		print("IMPORTING CASCADE AND DLIB...")
		#Importing Haar cascade and DLIB's facial landmarks detector
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
		print("CASCADE AND DLIB IMPORTED.")

		# Precision in % -> Tunes the recognizer according to our need
		precision = 0.95

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

		detections = 0
		detectionIndices = []


		#Convert the frame to grayscale
		grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#Activating Haar cascade classifier to detect faces
		faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.5, minNeighbors = 5)

		print("STARTING FACE DETECTION.")
		for(x, y, w, h) in faces :
			print("FOUND A FACE.")
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

				# Stores usernames of trained faces
				usernames = []
				lines = [line.rstrip('\n') for line in open('dataStore.txt')]
				for line in lines:
					tempList = line.split()
					usernames.append(tempList[0])

				print("SENDING FACE TO KERAS...")

				# Keras neural net
				npratios = []
				npratios.append(ratios)
				npratios = np.array(npratios)

				import time

				start_time = time.time()
				kerasOutput = model.predict(npratios)
				print("--- Detection time: %s seconds ---" % (time.time() - start_time))
				print("\nKERAS O/P: {}".format(kerasOutput))
				# print("\nMAP: [Bobby, Omkar, Chinmay, Sumit, Arjun]")
				maxval = -1
				maxid = -1
				for x in range(0, len(kerasOutput[0])):
					if (kerasOutput[0][x] > maxval):# and (kerasOutput[0][x] > 0.90):
						maxval = kerasOutput[0][x]
						maxid = x

				#print("\nMAX CONFIDENCE FOR : {}".format(usernames[maxid]))

				if(maxid != -1):
					userName = usernames[maxid]

					print("MAX CONFIDENCE FOR {}".format(userName))
					return userName
					break


		del model
		del face_cascade
		del predictor
		del precision
		del targetPoints
		del lines
		del detectionIndices
		del detections
		del grayFrame
		del faces	

		return -1	

	# decoded contains image which can be read by opencv
	result = recognizeFace(decoded)

	if(result == -1):
		return jsonify(
			foundFlag = 0,
			foundFace = "NOT FOUND"
		)
	else:
		return jsonify(
			foundFlag = 1,
			foundFace = result
		)