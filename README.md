# Explanation

## Python Scripts

1.	gather_examples.py
o	Likely used for collecting and organizing examples of real and fake face images/videos to create a dataset.
2.	liveness_demo.py
o	A demonstration script that likely shows the face liveliness detection model in action.
3.	livenessnet.py
o	Defines the neural network architecture used for liveliness detection.
4.	train.py
o	The main training script that loads the dataset, trains the model, and saves it for future use.

Pre-trained Models and Pickled Objects
5.	liveness.model
o	The trained liveliness detection model.
6.	le.pickle
o	Likely a pickled label encoder for encoding class labels (e.g., "real" or "fake").

Dataset

7.	dataset/
Contains subdirectories:
1.	fake/: Sample fake face images (e.g., 2464.png, 2465.png).
2.	real/: Sample real face images (e.g., 1604.png, 1605.png).

Face Detector
8.	face_detector/
Contains files for face detection:
1.	deploy.prototxt: Configuration file for a pre-trained face detection model.
2.	res10_300x300_ssd_iter_140000.caffemodel: Pre-trained weights for the face detection model.
Videos
9.	videos/
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


