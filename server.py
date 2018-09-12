from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def hello_world():

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


	test = request.form.get('picture')
	print(test)


	decoded = toRGB(stringToImage(base64.decodestring(stringToBase64(test))))
	cv2.imwrite("/home/chinmay/MASTERS/myGENImage.jpg", decoded)

	#cv2.imshow("TEST", decoded)

	return "Hi boi!"