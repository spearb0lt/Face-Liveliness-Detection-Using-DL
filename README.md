# UNDERSTANDING THE PROJECT

## Python Scripts

### 1.	gather_examples.py
This script extracts face images from video streams to create a dataset for training or testing.


#### Command-line arguments:

--input: Path to the input video file.

--output: Directory to save cropped face images.

--detector: Path to the OpenCV face detector files (deploy.prototxt and .caffemodel).

--confidence: Minimum confidence threshold for face detection.

--skip: Number of frames to skip between detections.

### 2.	liveness_demo.py
This script runs a real-time demonstration of the liveness detection model using a webcam feed.

#### Takes command-line arguments for:

--model: Path to the trained liveness detection model.

--le: Path to the label encoder.

--detector: Path to the face detector.

--confidence: Minimum confidence threshold for face detection.


### 3.	livenessnet.py
o	Defines the neural network architecture used for liveliness detection.

Network Architecture:
1.	Input Layer:
Accepts input images of specified dimensions (width, height, depth).
2.	Convolutional Layers:
Two sets of Conv2D → ReLU → BatchNormalization → MaxPooling2D layers.
Dropout applied after each set to reduce overfitting.
3.	Fully Connected Layer:
A dense layer with 64 units and ReLU activation.
Dropout applied for regularization.
4.	Output Layer:
A dense layer with a softmax activation function to output probabilities for "real" or "fake."


### 4.	train.py
The main training script that loads the dataset, trains the model, and saves it for future use.

#### Command-line arguments:

--dataset: Path to the dataset directory containing subdirectories for "real" and "fake" images

--model: Path to save the trained model.

--le: Path to save the label encoder.

--plot: Path to save the training plot.


Pre-trained Models and Pickled Objects
### 5.	liveness.model
The trained liveliness detection model.
Used in liveness_demo.py for real-time face liveness classification.

### 6.	le.pickle
A serialized label encoder object that maps numerical class indices to human-readable labels ("real" or "fake").
Used in liveness_demo.py to convert model predictions into labels.
Dataset

### 7.	dataset/
Contains subdirectories:
1.	fake/: Sample fake face images (e.g., 2464.png, 2465.png).
2.	real/: Sample real face images (e.g., 1604.png, 1605.png).

Face Detector
### 8.	face_detector/
Contains files for face detection:

#### 1.	deploy.prototxt: 
A configuration file used by the Caffe deep learning framework. It defines the architecture of the neural network for the face detection task.

##### Key Components:

1.	Input Layer:
Specifies the input dimensions for the network (e.g., 300x300x3 for an RGB image of size 300x300).
2.	Convolutional Layers:
Series of convolutional layers designed to extract spatial features from images.
3.	Bounding Box Regression:
The network predicts bounding box coordinates for detected faces in the image.
4.	Confidence Scores:
Alongside bounding boxes, the network predicts confidence scores for whether the detected region contains a face.
5.	Layer Connections:
Defines how layers are connected, including pooling, activation, and normalization layers.

##### Usage in Project:
Loaded in gather_examples.py and liveness_demo.py to define the network architecture for detecting faces.

#### 2.	res10_300x300_ssd_iter_140000.caffemodel: 
A pre-trained model file for the SSD (Single Shot Multibox Detector) framework using the ResNet-10 backbone. Stores the trained weights and biases for all layers of the network.

##### Key Details:

1.	Pre-Trained on Face Datasets:
Trained on datasets like WIDER FACE to detect human faces with high accuracy and reliability.
2.	ResNet-10 Backbone:
A lightweight version of the ResNet architecture optimized for speed and performance in real-time applications.
3.	Single Shot Multibox Detector (SSD):
A fast and efficient object detection framework. Detects objects (faces) in a single pass through the network.

##### Outputs:

•	Bounding box coordinates.

•	Confidence scores for each detected face.

##### Purpose:

•	Provides the learned weights required to detect faces in images or video frames.

•	Can identify faces of varying sizes and orientations in real time.

##### Usage in Project:

•	Loaded in gather_examples.py and liveness_demo.py using OpenCV’s cv2.dnn.readNetFromCaffe function.

•	Combined with deploy.prototxt to initialize the face detection model and run inference.

#### How They Work Together

##### 1.	Initialization:

o	The deploy.prototxt file is loaded to define the network structure.

o	The res10_300x300_ssd_iter_140000.caffemodel file is loaded to provide the trained weights for each layer.

##### 2.	Inference:

o	Input images (e.g., a video frame) are preprocessed and passed through the network.

o	The network outputs:

•	Bounding boxes for detected faces.

•	Confidence scores indicating the likelihood that the region contains a face.



### 9.	videos/
o	Contains demonstration videos:
1.	fake.mp4: Likely a video of a fake face.
2.	real.mp4: Likely a video of a real face.










# Steps To Use This Project

## 1. Run "gather_examples.py"

python3 gather_examples.py --input videos/fake.mp4 --output dataset/fake --detector face_detector --skip 1
python3 gather_examples.py --input videos/real.mp4 --output dataset/real --detector face_detector --skip 1

## 2. Run "livenessnet.py" 

python3 livenessnet.py

## 3. Run "train.py"

python3 train.py --dataset dataset --model liveness.model --le le.pickle

## 4. Run "liveness_demo.py"

python3 liveness_demo.py --model liveness.model --le le.pickle --detector face_detector


