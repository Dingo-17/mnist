---
title: "MNIST Image Classification using PyTorch"
output: github_document
---

# MNIST Image Classification using PyTorch

This project applies deep learning techniques to classify handwritten digits from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). We implement and compare three models to achieve this classification:
1. Logistic Regression
2. Feedforward Neural Network (FNN)
3. Convolutional Neural Network (CNN)

The purpose of this project is to demonstrate the progression of classification performance from simpler to more complex models and analyze the resulting test accuracy for each model.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Project Structure
- **data/**: Directory where the MNIST dataset will be downloaded.
- **README.md**: This file, explaining the project and its contents.
- **code.ipynb**: Jupyter Notebook containing code for data preprocessing, model building, training, and evaluation.
  
## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required Python libraries:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```
3. Connect to Google Colab if working in the cloud or ensure you have a GPU-enabled environment for training acceleration.


# MNIST Classification Project

## Dataset
The MNIST dataset consists of 60,000 grayscale images for training and 10,000 for testing. Each image is 28x28 pixels and represents a digit from 0 to 9. We will downsample this dataset to 30,000 training images and 5,000 testing images to reduce training time and resource requirements.

## Preprocessing
The images are transformed using PyTorch’s `torchvision.transforms`:
- **Resizing** to 28x28 pixels
- **Random rotation**
- **Random perspective transformation**
- **Normalization** using mean and standard deviation for MNIST (0.1307 and 0.3081)

## Models
### 1. Logistic Regression
A simple baseline model that flattens the images and applies a linear layer followed by a sigmoid activation. It is trained to classify the digits by applying logistic regression principles.

### 2. Feedforward Neural Network (FNN)
This model has:
- 1 hidden layer with 256 neurons
- ReLU activation function
- Output layer for 10 classes

### 3. Convolutional Neural Network (CNN)
A more complex architecture using two convolutional blocks followed by fully connected layers:
- **Conv Block 1**: Conv layer with 16 filters, ReLU activation, and max-pooling
- **Conv Block 2**: Conv layer with 32 filters, ReLU activation, and max-pooling
- **Fully Connected Layers**: Includes a layer with 128 neurons, followed by a final classification layer.

## Training and Evaluation
Each model is trained using PyTorch’s Adam optimizer with CrossEntropy loss. After training, accuracy and loss are computed for each epoch.

Training details:
- **10 epochs**
- **Batch size of 64**

## Results
The models are evaluated based on test accuracy and visualized through a confusion matrix.

| Model                       | Test Accuracy |
|-----------------------------|---------------|
| Logistic Regression         | ~[Accuracy]   |
| Feedforward Neural Network  | ~[Accuracy]   |
| Convolutional Neural Network| ~[Accuracy]   |

Additionally, a confusion matrix visualizes the performance of the CNN model on test data, helping identify any specific digit classes that are more prone to misclassification.

## Visualization
The code provides several visualizations:
- **Accuracy vs. Epochs**: Training accuracy over epochs for each model.
- **Label Distribution**: Histogram of training and testing label distribution.
- **Sample Images**: Random samples from the MNIST dataset.
- **Confusion Matrix**: Heatmap for CNN model performance.

## Acknowledgments
This project is based on references and diagrams from:
- Kaggle MNIST Data
- PyTorch documentation and tutorials

