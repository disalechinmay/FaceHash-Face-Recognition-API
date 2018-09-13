# FaceHash-Face-Recognition-API
Minimal API for face recognition.   
<br>
[![start here](https://img.shields.io/badge/Coverage-5%25-yellow.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

<br>
[![start here](https://img.shields.io/badge/Request-START%20HERE!-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

**The project will be built in quite an unusual way.**
- **Documentation of this project will be created first in `README.md`.**
- **Later on, code will be added according to the documentation.**



<br><br>
[![start here](https://img.shields.io/badge/DOCUMENTATION-What%20is%20this%20all%20about%3F-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

The main aim of this project is to create a minimal API for beginners so that they could use face recognition in their applications.
> Also it has been noticed that face recognition systems usually have a lot of dependencies.
> Installing all these dependencies on every client is not at all feasible.
> A simple way to solve this is to create a server which will handle all the work.
> An image to be searched for faces will be sent to the server and it will respond with appropriate information.


<br><br>
[![start here](https://img.shields.io/badge/DOCUMENTATION-How%20this%20system%20works%3F-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

1. Client sends an image to the server using appropriate API call.
2. Server processes the image in following manner :
    - Detects faces using Haar cascades. ([Frontal face detector](https://github.com/opencv/opencv/tree/master/data/haarcascades))
    - Detect facial landmarks of detected face using [DLIB](https://github.com/davisking/dlib)
    - Take ratios between a select few of these landmarks. [Check here](https://github.com/disalechinmay/FaceHash-Face-Recognition)
    - Run the Keras model which will recognize faces. **(Every `API_KEY` will have a single Keras model on the server.)**
3. Return a JSON response to the client.

**Training a new face will work in a similar way. Only the client will stream 100 photos and then Keras model will be retrained.**<br>
**! WARNING : The API will be disabled for that particular `API_KEY` during retraining the model !**


<br><br>
[![start here](https://img.shields.io/badge/DOCUMENTATION-How%20to%20use%20it%3F-brightgreen.svg?longCache=true&style=for-the-badge)](https://github.com/disalechinmay/FaceHash-Face-Recognition-API/blob/master/README.md)

### To recognize a face using Keras model saved under alias of `API_KEY`:

        POST the image to 127.0.0.1:5000 (127.0.0.1:5000/API_KEY) (Flask test server) according to guidelines mentioned below:
        - Image should be converted to base64.
        - Convert the base64 bytes to string and send.

        
        RESPONSE FROM SERVER:
        {
            foundFlag: 1 | 0,   // 1 if face is recognized successfully, else 0
            foundFace: userName // username of recognized face as saved on the backend under alias of `API_KEY`.
        }
        
### To add a new face to Keras model saved under alias of `API_KEY`:

        POST 100 images and username to 127.0.0.1:5000/train (127.0.0.1:5000/API_KEY/train) (Flask test server) according to guidelines mentioned below:
        - Image should be converted to base64.
        - Convert the base64 bytes to string and send.

        
        RESPONSE FROM SERVER: // Will take ton of time (3 ~ 5 minutes) because of retraining Keras model
        {
            trainingSuccess: 1 | 0,   // 1 if trained successfully, else 0
        }
        
        
## POTENTIAL PROBLEMS:
1. Will server respond while model is being retrained ?
2. If yes, how will you handle requests to the model that is being retrained?
3. Why 100 photos to retrain? Should developers be given a choice here?
4. Should developers be given to choice to handle Keras hyperparameters?
5. Should developers be given such kind of fleibility or just keep the API as minimal as possible?

## THINGS IMPLEMENTED:
- Sending base64 image to server.
- Successfully recognized a face.

## YET TO BE IMPLEMENTED:
- `API_KEY` generation and integration
- Adding new face functionality i.e. retraining model and handling conflicts involved.
