# FaceHash-Face-Recognition-API
Minimal API for face recognition.   
<br>
[![start here](https://img.shields.io/badge/Coverage-75%25-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

<br><br>
[![start here](https://img.shields.io/badge/DOCUMENTATION-What%20is%20this%20all%20about%3F-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)
.

The main aim of this project is to create a minimal API for beginners so that they could use face recognition in their applications.
> Also it has been noticed that face recognition systems usually have a lot of dependencies.
> Installing all these dependencies on every client is not at all feasible.
> A simple way to solve this is to create a server which will handle all the work.
> An image to be searched for faces will be sent to the server and it will respond with appropriate information.


<br><br>
[![start here](https://img.shields.io/badge/DOCUMENTATION-How%20this%20system%20works%3F-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

1. Client sends an image and an username to the server using appropriate API call.
2. Server processes the image in following manner :
    - Detects faces using Haar cascades. ([Frontal face detector](https://github.com/opencv/opencv/tree/master/data/haarcascades))
    - Detect facial landmarks of detected face using [DLIB](https://github.com/davisking/dlib)
    - Encode these landmarks
    - Run the Keras model which will recognize the user's face. **(Respective Keras model will be fetched from 'HOME:`API_KEY`'.)**
3. Return a JSON response to the client.

<br>


**Training a new face will work in a similar way. Only the client will send 10 photos (casted as String(base64(Image))) and then Keras model will be retrained.** <br>\
**![CHECK THIS](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/Front%20end/client.html) FOR DETAILS REGARDING HOW TO CONVERT IMAGES TO BE SENT TO SERVER**


<br><br>
[![start here](https://img.shields.io/badge/DOCUMENTATION-How%20to%20use%20it%3F-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

**AS API IS NOT HOSTED, RUN LOCALLY.**
<br>
### To recognize a face using respective Keras model saved under 'HOME:`API_KEY`':

        POST `API_KEY`, username and image to 127.0.0.1:5000/recognize/ (Flask test server) according to guidelines mentioned below:
        - Image should be converted to base64.
        - Convert the base64 bytes to string and send.
        - API_KEY and username should be strings.

        
        RESPONSE FROM SERVER IN CASE OF SUCCESSFUL RECOGNITION:
        {
            responseType : "SUCCESS",
            successDescription : "Face recognized successfully.",
            detectionTime : 0.02, // Seconds taken by Keras model to generate output
            confidencePercentage : 0.95
        }
        
        RESPONSE FROM SERVER IN CASE OF FAILED RECOGNITION:
        {
            responseType : "FAILURE",
            successDescription : "Face not recognized." | "Face not detected."
        }
        
### To add a new face to the respective Keras model saved under 'HOME:`API_KEY`':

        POST `API_KEY`, 50 ~ 100 images and username to 127.0.0.1:5000/train (Flask test server) according to guidelines mentioned below:
        - Image should be converted to base64.
        - Convert the base64 bytes to string and send.
        - API_KEY and username should be strings.


        
        RESPONSE FROM SERVER: // Will take ton of time (1 ~ 5 minutes) because of retraining Keras model
        {
            responseType : "SUCCESS",
            successDescription : "Model trained successfully."
        }
        
        
## POTENTIAL PROBLEMS:
1. Will server respond while model is being retrained ?
2. If yes, how will you handle requests to the model that is being retrained?
3. Should developers be given to choice to handle Keras hyperparameters?
4. Should developers be given such kind of flexibility or just keep the API as minimal as possible?


USE API_KEY = "API_KEY_TEST_1"
