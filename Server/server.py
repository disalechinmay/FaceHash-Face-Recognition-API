from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/train/', methods=['POST'])
def train_POST():
	import tensorflow as tf
	tf.keras.backend.clear_session()
	
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


	print("[LOG] -> Started retrieving images...")

	incomingStream = request.form.get('stream')
	#print(incomingStream)
	splitted = incomingStream.split("[IMAGE_STREAM_DELIMITER]")
	print("[LOG] -> All images received")
	print("[LOG] -> No of images received : {}".format(len(splitted)))

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
	import array
	import io
	import base64 
	import numpy as np
	import tensorflow as tf
	from tensorflow import keras
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	# Average out all the values:

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

	#Importing Haar cascade and DLIB's facial landmarks detector
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

	#TOTAL LANDMARKS DETECTED ARE FROM 0-67
	totals = []
	counts = []
	for i in range(0, totalTargets):
		totals.append(0.0)
		counts.append(0)

	# Take in base64 string and return PIL image
	def stringToImage(base64_string):
		imgdata = base64.b64decode(base64_string)
		return Image.open(io.BytesIO(imgdata))

	# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
	def toRGB(image):
		return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


	def stringToBase64(s):
		return base64.b64encode(s.encode('utf-8'))

	print("[LOG] -> Started analysing images...")
	for x in splitted :
		frame = toRGB(stringToImage(base64.decodestring(stringToBase64(x))))

		#Convert the frame to grayscale
		grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#Activating Haar cascade classifier to detect faces
		faces = face_cascade.detectMultiScale(grayFrame, scaleFactor = 1.5, minNeighbors = 5)

		for(x, y, w, h) in faces :
			#Cropping and resizing face area
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

			baseLine = getDistance(shapeCopy, 28, 27)
		
			for z in range(0, totalTargets):
				temp = getDistance(shapeCopy, targetPoints[z], 27)
				currentRatio = float(temp)/float(baseLine)

				prevTotal = totals[z]
				prevCount = counts[z]
				totals.remove(totals[z])
				counts.remove(counts[z])
				totals.insert(z, prevTotal+currentRatio)
				counts.insert(z, prevCount+1)
	print("[LOG] -> All images analyzed.")

	curatedRatios = []

	print("[LOG] -> Calculating average...")
	if(counts[0] > 50):
		for x in range(0, totalTargets):
			averageValue = totals[x]/counts[x]
			curatedRatios.append(float(averageValue))
	else:
		return jsonify(
				responseType = "FAILURE",
				failureDescription = "For this to work, at least 50 photos should be sent such that a face will be clearly visible in each one of them."
			)

	print("[LOG] -> Calculation done.")
	print(curatedRatios)

	X = []
	temp = [5.67469951822, 3.01786596509, 3.38644243788, 4.40128707661, 5.6145555955, 6.68856446685, 7.81805994584, 8.67407231537, 1.0, 3.80675806429, 3.97654258777, 3.86155507889, 1.45054649363, 2.74933829225, 2.09696092969, 3.15649919849, 1.45481561561, 5.07613841469, 4.9357296169, 5.17252304232, 6.09019191374, 6.25542463203, 6.29758693006, 5.57055012231, 5.52019500597]
	X.append(temp)
	X.append(curatedRatios)
	temp = []
	temp = [7.6438755024, 2.96538201425, 4.85876054204, 6.37478439871, 7.84219382332, 9.14543948112, 10.7819584314, 11.2441548848, 1.0, 4.78171347933, 4.85244813287, 4.78617682862, 2.04350583494, 3.94124976952, 3.04815828402, 4.53756179591, 1.89986316624, 6.71392975227, 6.55156968539, 7.04501472427, 8.19822194073, 8.25487553558, 8.25869210105, 7.39306423257, 7.35406102551]
	X.append(temp)

	X = np.array(X)
	y = np.array([[0], [1], [2]])

	model = keras.Sequential()
	model.add(keras.layers.Dense(25, input_dim=25, activation='softmax'))
	model.add(keras.layers.Dense(12, activation='softmax'))
	model.add(keras.layers.Dense(3, activation='softmax'))

	class EarlyStoppingByLossVal(keras.callbacks.Callback):
	    def __init__(self, monitor='loss', value=0.01, verbose=0):
	        super(keras.callbacks.Callback, self).__init__()
	        self.monitor = monitor
	        self.value = value
	        self.verbose = verbose

	    def on_epoch_end(self, epoch, logs={}):
	        current = logs.get(self.monitor)
	        if current is None:
	            print("Early stopping requires %s available!" % self.monitor)
	            exit()

	        if current < self.value:
	            if self.verbose > 0:
	                print("Epoch %05d: early stopping THR" % epoch)
	            self.model.stop_training = True
	earlyStop = [EarlyStoppingByLossVal()]

	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

	model.fit(X, y, batch_size=1, epochs=100000, callbacks=earlyStop)

	os.chdir("/home/chinmay/MASTERS/Flask/HOME:" + API_KEY + "/")
	model.save(incomingUserName + ".keras")
	os.chdir("/home/chinmay/MASTERS/Flask/")


	return jsonify(
			responseType = "SUCCESS", 
			successDescription = "Model trained successfully."
		)

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

				if(maxid == 1):
					confidence = kerasOutput[0][1]
					print("[LOG] -> RETURNING SUCCESS...")
					print("\nKERAS O/P: {}".format(kerasOutput))
					return jsonify(
						responseType = "SUCCESS",
						successDescription = "Face recognized successfully",
						confidencePercentage = str(confidence),
						detectionTime = str(detection_time)
					)
				else:
					print("[LOG] -> RETURNING FAILURE...")
					print("\nKERAS O/P: {}".format(kerasOutput))
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