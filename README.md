# Helmet Detection System

This project is designed to detect whether a person is wearing a helmet or not using computer vision techniques. It involves training a Convolutional Neural Network (CNN) to classify images of people with and without helmets and then using this model to make real-time predictions from a webcam feed.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Helmet Detection](#helmet-detection)
- [Screenshots](#screenshots)

## Introduction

The helmet detection system is a project that uses TensorFlow and OpenCV to detect helmets in real-time video streams. It consists of two main scripts:
- `train_model.py`: Script to train the helmet detection model.
- `detect_helmet.py`: Script to detect helmets in real-time using the trained model.

## Dataset

The dataset for this project is taken from Kaggle. It consists of images and their corresponding annotations in XML format.

You can download the dataset from Kaggle [here](https://www.kaggle.com).

## Model Training

The `train_model.py` script reads the dataset, preprocesses the images, and trains a CNN to classify images into two categories: "With Helmet" and "Without Helmet". The model is saved in the `Models` directory after training.

### Training the Model

To train the model, run:
```bash
python train_model.py
```

## Helmet Detection
The detect_helmet.py script loads the trained model and a pre-trained face detection model to detect faces in a video stream and classify whether the detected faces are wearing helmets or not.

### Running Helmet Detection
To start the helmet detection, run:
```bash
python detect_helmet.py
```

## Screenshots
Here are some screenshots showcasing the helmet detection system in action:

![Screenshot 2024-08-17 153913](https://github.com/user-attachments/assets/bbbfe30e-45dd-4753-931c-1314b22412ed)

![Screenshot 2024-08-17 154130](https://github.com/user-attachments/assets/63e33ffb-6136-4276-8ec0-2b3a5d1340f8)

