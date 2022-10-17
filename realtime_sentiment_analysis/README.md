# Real-time-Sentiment-Analysis

### Introduction

This is my self-study practice to get used to Tensorflow and Keras. I'm interested in sentiment analysis topic, so this time I wonder if I can try to make some real-time sentiment analysis by cropping frames (images) from camera and then returning the prediction result. The idea is coding a simple CNN to train model or use transfer learning, then use the model to predict real-time image.

### Dataset
FER2013 - Facial Expression Recognition 2013
Link: https://www.kaggle.com/datasets/msambare/fer2013
FER2013 is a well-studied dataset and has been used in ICML competitions and several research papers. It is one of the more challenging datasets with human-level accuracy only at 65±5% and the highest performing published works achieving 75.2% test accuracy.

### Repository Description:
train.ipynb: all implementation for preprocessing and training
camera_realtime1.py : implementation for real-time camera and cropping image.
cnn_model.h5: model from simple CNN

### Result
Simple CNN model: I got 0.6095 accuracy (test set). Of course, not a very good performance even though I tried to make it better.

<img width="330" alt="image" src="https://user-images.githubusercontent.com/53163183/166243266-ffae376f-fa9c-40af-9c4f-5cf449095b6e.png">

Transfer learning (ResNet50): 0.5678 accuracy (test set) :)

<img width="330" alt="image" src="https://user-images.githubusercontent.com/53163183/166243200-bf8d02a4-871c-4a4d-aad7-bafc0ad34d0d.png">

When testing on camera it somewhat works (of course not with exceptional performance).

### Lesson Learned
Data augmentation and resampling are helpful for this problem.
Use transfer learning if not correct then easily being overfitted.
Get used to Keras and somewhat know how it works.
