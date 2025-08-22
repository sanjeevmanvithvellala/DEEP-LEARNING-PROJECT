# DEEP-LEARNING-PROJECT

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras in Google Colab. The goal is to train a model that can accurately recognize 10 categories of real-world objects.

## Features

- End-to-end image classification pipeline
- Uses Convolutional Neural Network (CNN) for feature learning
- Model trained on CIFAR-10 (60,000 images)
- Data normalization and visualization of samples
- Live training progress with accuracy & loss curves
- Prediction visualizations on test images
- Built entirely using open-source tools
- Fully runnable in Google Colab

## Dataset

- CIFAR-10 from tensorflow.keras.datasets
- 10 Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Image size: 32x32 pixels, RGB
- 50,000 training images + 10,000 test images

## Tools & Technologies Used
- Tool / Library	Purpose
- TensorFlow/Keras	Model building and training
- Matplotlib	Data & result visualization
- NumPy	Numerical operations
- Google Colab	Free GPU-enabled environment
- Python 3.10+	Programming Language

## Model Architecture

Input Layer (32x32x3)
→ Conv2D (32 filters, 3x3) + ReLU
→ MaxPooling2D (2x2)
→ Conv2D (64 filters, 3x3) + ReLU
→ MaxPooling2D (2x2)
→ Flatten
→ Dense (64 neurons) + ReLU
→ Dense (10 classes) + Softmax

## Future Improvements
- Use Data Augmentation to boost accuracy
- Add Dropout for regularization
- Use Transfer Learning with pretrained models like ResNet50
- Hyperparameter tuning with KerasTuner
