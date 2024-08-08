# MNIST_with_Torch
# CNN-MNIST Classifier

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify handwritten digits from the MNIST dataset. The model is trained and evaluated on the MNIST dataset, achieving high accuracy in predicting the correct digit for each image.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview

This project demonstrates the use of Convolutional Neural Networks (CNNs) for image classification tasks. The MNIST dataset, a popular benchmark for image classification, consists of 28x28 grayscale images of handwritten digits (0-9). The CNN is trained on this dataset to accurately classify the images into one of the ten digit classes.

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains 70,000 images of handwritten digits, split into 60,000 training images and 10,000 test images.

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28x28 pixels
- **Number of Classes**: 10 (digits 0-9)

## Model Architecture

The CNN architecture used in this project consists of three convolutional layers followed by max-pooling layers, and two fully connected layers for classification.

### Model Layers:

1. **Conv Layer 1**: 16 filters, 3x3 kernel, ReLU activation, followed by 2x2 Max-Pooling
2. **Conv Layer 2**: 32 filters, 3x3 kernel, ReLU activation, followed by 2x2 Max-Pooling
3. **Conv Layer 3**: 64 filters, 3x3 kernel, ReLU activation, followed by 2x2 Max-Pooling
4. **Fully Connected Layer 1**: 128 neurons, ReLU activation
5. **Output Layer**: 10 neurons (one for each digit class)

### Loss Function and Optimizer:

- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam

## Installation

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
