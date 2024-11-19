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


