import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from collections import Counter
from PIL import Image
from skimage import io, transform
import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from google.colab import drive

"""##GPU

Sets up CUDA device to use with torch.


"""

torch.manual_seed(42) # For consistency
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(device)

device

"""# **Part 1:** Data Preprocessing
Dataset ([MNIST](https://en.wikipedia.org/wiki/MNIST_database)) dataset.

<div>
<img src="https://s2.loli.net/2023/03/26/GwFJhNeskzE5Ptx.webp", width="800"/>
</div>

Diagram Reference: [Link](https://en.wikipedia.org/wiki/MNIST_database)

[MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) dataset consists of 60,000 28x28 grayscale images in 10 classes.

There are 60,000 training images and 10,000 test images.

To avoid having long training time and potentially running out of GPU, we will downsample and train the model with 30,000 training images and 5,000 testing images.


We would be using this dataset to train 3 different models:

1.   Logistic Regression
2.   Feedforward (Fully-Connected) Neural Network
3.   Convolutional Neural Network

And analyze the difference between these models by looking at the test accuracy and loss.

## 1.1 Pytorch Dataset and DataLoader



<div>
<img src="https://s2.loli.net/2023/03/30/yxbP8gXCroO1Y7c.png", width="800"/>
</div>

Diagram Reference: [Link](https://www.kaggle.com/code/uvxy1234/cifar-10-implementation-with-pytorch)

Dataset class is defined with 3 components:

1.   __init__ : setting up the parameters being used in the class (e.g., `transforms` which corresponds to the transformation being applied)
2.   __len__ : so that len(dataset) returns the size of the dataset.
3.   __getitem__ to support the indexing such that `dataset[i]` can be used to get `i`ith sample (in our case Image, label pair).

### 1.1.1 Instantiate Dataset (for train/test dataset)


**NOTE**: The values `0.1307` and `0.3081` are mean and standard deviation, respectively, of the MNIST dataset [Ref](https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/4)
"""

scale = 28


distortion_scale = 0.3

transform = transforms.Compose([
    transforms.Resize((scale, scale)),  # Resize images to 28x28
    transforms.ToTensor(),              # Convert images to tensors
    transforms.RandomRotation((-45, 45)),  # Rotate images randomly within a range of -45 to 45 degrees
    transforms.RandomPerspective(distortion_scale=distortion_scale, p=0.5),  # Apply random perspective shift with a distortion of 0.3 to 50% of the data
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the tensors using the given mean and standard deviation
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

"""

---
The code below:
* Initiates the `Dataset` object for the training set as `train_dataset`
* Downsamples training by getting the train data at even-numbered indices, specified as `train_subset`
* Initiates the `Dataset` object for the testing set as `test_dataset`
* Downsamples testing by getting the test data at even-numbered indices, specified as `test_subset`
---"""

from torch.utils.data import Subset

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_indices = list(range(0, len(train_dataset), 2))
test_indices = list(range(0, len(test_dataset), 2))

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

transforms_train = []
for i in range(len(train_dataset.transform.transforms)):
  transforms_train.append(str(train_dataset.transform.transforms[i]))
transforms_test = []
for i in range(len(test_dataset.transform.transforms)):
  transforms_test.append(str(test_dataset.transform.transforms[i]))

"""### 1.1.2 Dataloader - Train / Test
Now that we have `trainDataset` and `testDataset`, let's create dataloaders using these two datasets.

We can load the dataset into dataloaders using the `DataLoaders` object.
"""

batch = 64


train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True, num_workers=0)
test_loader = DataLoader(test_subset, batch_size=batch, shuffle=True, num_workers=0)

"""## 1.2 Summarizing our Dataset
In this section, we will be looking at the distribution of the dataset (e.g., how many instances belong to class with label `0`) and visualize what we are dealing with (i.e., plot out the sample images)

### 1.2.1 Looking at the distribution of labels
We can look at the distribution of labels by retrieving the labels of all possible instances of the subset of data pulled from `train_dataset` and `test_dataset` (i.e., `train_subset` and `test_subset`) for the training and testing data, respectively. We defined this in section `1.1.1`.
"""

# Train data
train_num_labels = len(set(train_dataset.targets.numpy()))

# Create a DataLoader for the train_subset
train_loader_bar_plot = DataLoader(train_subset, batch_size=len(train_subset), shuffle=True, num_workers=0)

# Create a dictionary for the train_subset
train_labels = next(iter(train_loader_bar_plot))[1].numpy()
train_subset_dict = dict(Counter(train_labels))


print(train_subset_dict)

# Use the original test dataset (with 10,000 images) object to obtain the number of label classes in test data
# Creating dictionary for test dataset

# Test data
test_num_labels = len(set(test_dataset.targets.numpy()))

# Create a DataLoader for the test_subset
test_loader_bar_plot = DataLoader(test_subset, batch_size=len(test_subset), shuffle=True, num_workers=0)

# Create a dictionary for the test_subset
test_labels = next(iter(test_loader_bar_plot))[1].numpy()
test_subset_dict = dict(Counter(test_labels))


print(test_subset_dict)

"""### 1.2.2 Visualize through bar charts
Now we are going to visualize the distribution of labels using bar charts for both training and testing set which we store the distributions in their respective dictionary objects in section 1.2.1.
"""

# TRAIN Data
# Import additional libraries
import matplotlib.ticker as ticker

# Function to create bar plots
def create_barplot(data_dict, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sort the keys
    sorted_keys = sorted(data_dict.keys())
    sorted_values = [data_dict[key] for key in sorted_keys]

    ax.bar(sorted_keys, sorted_values)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xticks(sorted_keys)  # Set x-axis ticks for each bar

    for i, v in enumerate(sorted_values):
        ax.text(sorted_keys[i], v + 5, str(v),
                horizontalalignment='center', fontweight='bold')

    plt.show()

# Create bar plots for training and testing labels
create_barplot(train_subset_dict, 'Training set labels and corresponding frequencies', 'Labels', 'Frequency')

# TEST Data
create_barplot(test_subset_dict, 'Testing set labels and corresponding frequencies', 'Labels', 'Frequency')

"""### 1.2.3 Visualize the Training Dataset!

Since everything tends to make more sense when one could literally see it, we now visualize the images in the `train_dataset` given a fixed set of indices.

---




"""

torch.manual_seed(42)
sample_idxs = [10, 300, 700, 2708, 5035, 8000] # DO NOT MODIFY

# loop through the length of tickers and keep track of index

# Create a 2x3 subplot grid to visualize the images
fig, axes = plt.subplots(2, 3, figsize=(8, 4))

# Loop through the sample_idxs and keep track of index
for idx, sample_idx in enumerate(sample_idxs):
    # Get the image and label from train_dataset
    image, label = train_dataset[sample_idx]

    # Calculate the row and column of the current image in the subplot grid
    row, col = idx // 3, idx % 3

    # Plot the image in grayscale
    axes[row, col].imshow(image.squeeze(), cmap='gray')

    # Assign the title of each image to be their respective labels
    axes[row, col].set_title(f"Label: {label}")

    # Hide grid lines and axes labels
    axes[row, col].axis('off')

# Display the plot
plt.show()

"""# **Part 2:** Classification Models

We now have the data needed to train a multi-class object classifier. We will start simple with a logistic regression classifier as a baseline for our performance, before we move onto more complex neural networks.

In this case, we are looking at the remaining part in the pipeline which were grayed out before as follows:

<div>
<img src="https://s2.loli.net/2023/03/30/ZCBFQvkXuoJpI8K.png", width = "800"/>
</div>

Diagram Reference: [Link](https://www.kaggle.com/code/uvxy1234/cifar-10-implementation-with-pytorch)

## 2.1 Logical Logistic Regression - Baseline

Let's first try solving this problem with a Logistic Regression classifier.


<div>
<img src='https://i.stack.imgur.com/fKvva.png',width='600'/>
</div>

Diagram Reference: [Link](https://stats.stackexchange.com/questions/366707/a-logistic-regression-with-neural-network-mindset-vs-a-shallow-neural-network)

### 2.1.1 Logistic Regression Model Architecture
"""

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(28 * 28, 10)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.flatten(x)  # Flattening the input tensor
        outputs = self.linear(x)
        outputs = self.sigmoid(outputs)

        return outputs

"""**Let**'s print the model summary"""

LogReg()

"""### 2.1.2 Training Logistic Regression Model

---
We will use `train_loader` built in 1.1.2 to train logistic regression model.

The optimizer is set as Adam.

"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Sending the data to device (CPU or GPU)
# # Step 1: instantiate the logistic regression to variable logreg
# logreg = LogReg().to(device)
# 
# # Step 2: set the loss criterion as CrossEntropyLoss
# criterion = nn.CrossEntropyLoss()
# 
# optimizer = optim.Adam(logreg.parameters(), lr=1e-4) #lr - learning step
# epoch = 10
# 
# loss_LIST_log = []
# acc_LIST_log = []
# 
# # Train the Logistic Regression
# for epoch in range(epoch):
#   running_loss = 0.0
#   correct = 0
#   total = 0
#   for inputs, labels in train_loader:
#       labels = labels.type(torch.LongTensor) # Cast to Float
#       inputs, labels = inputs.to(device), labels.to(device)
# 
#       # Step 1: Reset the optimizer tensor gradient every mini-batch
#       optimizer.zero_grad()
# 
#       # Step 2: Feed the network the train data
#       outputs = logreg(inputs)
# 
#       # Step 3: Get the prediction using argmax
#       _, predicted = torch.max(outputs.data, 1)
# 
#       # Step 4: Find average loss for one mini-batch of inputs
#       loss = criterion(outputs, labels)
# 
#       # Step 5: Do a back propagation
#       loss.backward()
# 
#       # Step 6: Update the weight using the gradients from back propagation by learning step
#       optimizer.step()
# 
#       # Step 7: Get loss and add to accumulated loss for each epoch
#       running_loss += loss.item()
#       # Step 8: Get number of correct prediction and increment the number of correct and total predictions after this batch
#       correct += (predicted == labels).sum().item()
#       total += labels.size(0)
# 
#   # Step 9: Calculate training accuracy for each epoch (should multiply by 100 to get percentage), store in variable called 'accuracy', and add to acc_LIST_log
#   accuracy = (correct / total) * 100
#   acc_LIST_log.append(accuracy)
#   # Step 10: Get average loss for each epoch and add to loss_LIST_log
#   loss_LIST_log.append(running_loss / len(train_loader))
# 
# 
#   # print statistics
#   print("The loss for Epoch {} is: {}, Accuracy = {}".format(epoch, running_loss/len(train_loader), accuracy))
#

"""### 2.1.3 Plotting Training Accuracy vs Epochs for Logistic Regression"""

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(5, 3))
sns.set_style("whitegrid")
plt.plot(range(1, len(acc_LIST_log) + 1), acc_LIST_log)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy (%)")
plt.title("Training Accuracy vs Epochs for Logistic Regression")
plt.show()

"""### 2.1.4 Logistic Regression Model Accuracy


"""

total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.type(torch.LongTensor) # Cast to Float
        images, labels = images.to(device), labels.to(device)


        # Get the output
        outputs = logreg(images)

        # Get the prediction using argmax
        _, predicted = torch.max(outputs.data, 1)
        # Get number of correct prediction and add to correct and total
        total += labels.size(0)
        correct += (predicted == labels).sum()
# Calculate test accuracy for logistic regression (should multiple by 100)
test_acc_log = (correct / total) * 100

print('Test Accuracy: ' + str(test_acc_log.item()))

"""## 2.2 Feedforward Neural Networks

<div>
<img src='https://s2.loli.net/2022/11/21/dvqstVUzcQPChD1.png', width='400'/>
</div>

Diagram reference: [Link](https://en.wikipedia.org/wiki/Feedforward_neural_network)

Since logistic regression isn't that great at the classification problem above, we need more representation power in our model. We will now define a feedforward neural network.

We create the *FNN* class below to define a feedforward neural network with **only 1 hidden layers with ```out_features``` of 256**. Note that the last layer must have the same number of classes as the output size!

### 2.2.1 Feedforward Neural Network Model Architecture
"""

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the neural network layers
        self.input_layer = nn.Linear(28 * 28, 256)
        self.hidden_activation = nn.ReLU()
        self.output_layer = nn.Linear(256, 10)


    def forward(self, x):
        #Implement the operations on input data
        x = x.view(x.size(0), -1)  # Flattening the input tensor
        x = self.input_layer(x)
        x = self.hidden_activation(x)
        outputs = self.output_layer(x)

        return outputs

"""Let's print the model summary"""

FNN()

"""### 2.2.2 Training FNN Model
---

The optimizer is set as Adam.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Sending the data to device (CPU or GPU)
# 
# # Step 1: instantiate the FNN model to variable fnn
# fnn = FNN().to(device)
# 
# # Step 2: set the loss criterion as CrossEntropyLoss
# criterion = nn.CrossEntropyLoss()
# 
# optimizer = optim.Adam(fnn.parameters(), lr=1e-4) #lr - learning step
# epoch = 10
# 
# acc_LIST_FNN = []
# loss_LIST_FNN = []
# 
# # Train the FNN
# for epoch in range(epoch):
#   running_loss = 0.0
#   correct = 0
#   total = 0
#   for inputs, labels in train_loader:
#       labels = labels.type(torch.LongTensor) # Cast to Long
#       inputs, labels = inputs.to(device), labels.to(device)
# 
#       # Step 1: Reset the optimizer tensor gradient every mini-batch
#       optimizer.zero_grad()
# 
#       # Step 2: Feed the network the train data
#       outputs = fnn(inputs)
# 
#       # Step 3: Get the prediction using argmax
#       _, predicted = torch.max(outputs.data, 1)
# 
#       # Step 4: Find average loss for one mini-batch of inputs
#       loss = criterion(outputs, labels)
# 
#       # Step 5: Do a back propagation
#       loss.backward()
# 
#       # Step 6: Update the weight using the gradients from back propagation by learning step
#       optimizer.step()
# 
#       # Step 7: Get loss and add to accumulated loss for each epoch
#       running_loss += loss.item()
# 
#       # Step 8: Get number of correct prediction and increment the number of correct and total predictions after this batch
#       total += labels.size(0)
#       correct += (predicted == labels).sum().item()
# 
#   # Step 9: Calculate training accuracy for each epoch (should multiply by 100 to get percentage), store in variable called 'accuracy', and add to acc_LIST_FNN
#   accuracy = 100 * correct / total
#   acc_LIST_FNN.append(accuracy)
# 
#   # Step 10: Get average loss for each epoch and add to loss_LIST_FNN
#   avg_loss = running_loss / len(train_loader)
#   loss_LIST_FNN.append(avg_loss)
# 
# 
#   # print statistics
#   print("The loss for Epoch {} is: {}, Accuracy = {}".format(epoch, running_loss/len(train_loader), accuracy))
#

"""### 2.2.3 Plotting Training Accuracy vs Epochs FNN
---

Plot the training accuracy vs epochs.

Chart Specifications:
1. The accuracy should be in the y-axis and epochs in x-axis.
2. Add chart title.
3. Epoch label should start with 1 (for audience interpretability).
---
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 3))
plt.plot(range(1, len(acc_LIST_FNN) + 1), acc_LIST_FNN)
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs Epochs (FNN)')
plt.xticks(range(1, len(acc_LIST_FNN) + 1))
plt.show()

"""### 2.2.4 FNN Model Accuracy

Calculate the Test Accuracy for the FNN Model we trained above (the technique for doing this is the same as computing the test accuracy for the logistic regression classifier above).
"""

total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.type(torch.LongTensor) # Cast to Float
        images, labels = images.to(device), labels.to(device)


        # Get the output
        outputs = fnn(images)

        # Get the prediction using argmax
        _, predicted = torch.max(outputs.data, 1)

        # Get number of correct prediction and add to correct and total
        total += labels.size(0)
        correct += (predicted == labels).sum()

# Calculate test accuracy for FNN (should multiple by 100)
test_acc_FNN = (correct / total) * 100

print('Test Accuracy: ' + str(test_acc_FNN.item()))

"""##2.3 "Convoluted" Convolutional Neural Networks




<div>
<img src='https://s2.loli.net/2022/11/21/L6pUz2chXWRGn31.png', width='800'>
</div>

Diagram Reference: [Link](https://www.analyticsvidhya.com/blog/2020/10/what-is-the-convolutional-neural-network-architecture/)

### 2.3.0 Calculating Output Dimensions of Convolution and Pooling Layers

<div>
<img src='https://s2.loli.net/2023/03/30/lKpjPLVHcuRC8n2.png',width='300'/>
</div>

Diagram Reference: [Link](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
"""

import math

def feature_map_dim(input_dim, padding, kernel_size, stride):

  output_dim = math.floor((input_dim - kernel_size + 2 * padding) / stride) + 1
  return output_dim

"""### 2.3.1 Convolutional Neural Network Model Architecture

---
"""

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Step 1: Initialize 1 - 3 convolution blocks (consists of a convolution layer, an activation function, a MaxPooling layer)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolution Block 1

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Step 2: Flatten the 2D image into a 1D tensor
        self.flatten = nn.Flatten(start_dim=1)

        # Step 3: Initialize 1-3 fully-connected layers
        self.fc1 = nn.Linear(in_features=1568, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)



    def forward(self, x):


        # Step 1. Pass the images (x) through convolution block 1
        x = self.conv_block1(x)

        x = self.conv_block2(x)

        # Step 2. Flatten the image
        x = self.flatten(x)


        # Step 3. Pass the output through the fully-connected layers (remember to include activation function(s))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        outputs = nn.functional.softmax(x, dim=1)


        return outputs  #changed from outputs

"""Let's print out the model summary"""

CNN()

"""### 2.3.2 Training CNN Model
---

---


"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Sending the data to device (CPU or GPU)
# # Step 1: instantiate the CNN model to variable cnn
# cnn = CNN().to(device)
# 
# # Step 2: set the loss criterion as CrossEntropyLoss
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(cnn.parameters(), lr=1e-4) #lr - learning step
# epoch = 10
# 
# acc_LIST_CNN = []
# loss_LIST_CNN = []
# 
# # Train the CNN
# for epoch in range(epoch):
#   running_loss = 0.0
#   correct = 0
#   total = 0
#   for inputs, labels in train_loader:
#       labels = labels.type(torch.LongTensor) # Cast to Float
#       inputs, labels = inputs.to(device), labels.to(device)
# 
#       # Step 1: Reset the optimizer tensor gradient every mini-batch
#       optimizer.zero_grad()
# 
#       # Step 2: Feed the network the train data
#       outputs = cnn(inputs)
# 
#       # Step 3: Get the prediction using argmax
#       predicted = outputs.argmax(dim=1)
# 
# 
#       # Step 4: Find average loss for one mini-batch of inputs
#       loss = criterion(outputs, labels)
# 
#       # Step 5: Do a back propagation
#       loss.backward()
# 
#       # Step 6: Update the weight using the gradients from back propagation by learning step
#       optimizer.step()
# 
#       # Step 7: Get loss and add to accumulated loss for each epoch
#       running_loss += loss.item()
# 
#       # Step 8: Get number of correct prediction and increment the number of correct and total predictions after this batch
#       total += labels.size(0)
#       correct += (predicted == labels).sum().item()
# 
#   # Step 9: Calculate training accuracy for each epoch (should multiply by 100 to get percentage), store in variable called 'accuracy', and add to acc_LIST_CNN
#   accuracy = 100 * correct / total
#   acc_LIST_CNN.append(accuracy)
# 
#   # Step 10: Get average loss for each epoch and add to loss_LIST_CNN
#   loss_LIST_CNN.append(running_loss/len(train_loader))
# 
#   # print statistics
#   print("The loss for Epoch {} is: {}, Accuracy = {}".format(epoch, running_loss/len(train_loader), accuracy))
#

"""### 2.3.3 Plotting Training Accuracy vs Epochs CNN

"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(5, 3))

# Plot the training accuracy vs epochs
ax.plot(range(1, len(acc_LIST_CNN) + 1), acc_LIST_CNN)

# Set x and y axis labels
ax.set_xlabel('Epochs')
ax.set_ylabel('Training Accuracy')

# Set the title
ax.set_title('Training Accuracy vs Epochs (CNN)')

# Display the plot
plt.show()

"""### 2.3.4 CNN Model Test Accuracy"""

total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.type(torch.LongTensor) # Cast to Float
        images, labels = images.to(device), labels.to(device)
        # Get the output
        outputs = cnn(images)

        # Get the prediction using argmax
        _, predicted = torch.max(outputs.data, 1)

        # Get number of correct prediction and add to correct and total
        total += labels.size(0)
        correct += (predicted == labels).sum()

# Calculate test accuracy for CNN (should multiple by 100)
test_acc_CNN = 100 * correct / total

print(f'Test Accuracy: ' + str(test_acc_CNN.item()))

"""## 2.4. Final Accuracies
---



"""

# Simply run this cell, please do not modify
print(f'Test Accuracy for Logistic Regression: ' + str(test_acc_log.item()))
print(f'Test Accuracy for FNN: ' + str(test_acc_FNN.item()))
print(f'Test Accuracy for CNN: ' + str(test_acc_CNN.item()))

"""## 2.5 Confusion Matrix

### 2.5.1 Create a confusion matrix
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def cm_generator(test_loader):

  # The goal is to obtain two lists of prediction and actual labels.
  # Then, using these two lists, create a confusion matrix dataframe
  y_true = []
  y_pred = []

  with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

  # Create the confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Create a pandas dataframe with row and column labels
  confusion_matrix_df = pd.DataFrame(cm, index=[str(i) for i in range(10)], columns=[str(i) for i in range(10)])

  return confusion_matrix_df

# Generate confusion matrix dataframe
confusion_matrix_df = cm_generator(test_loader)
print(confusion_matrix_df)

"""### 2.5.2 Visualizing Confusion Matrix"""

import seaborn as sns

def visualize_confusion_matrix(confusion_matrix_df):
    plt.figure(figsize=(8, 4))
    sns.heatmap(confusion_matrix_df, annot=True, fmt="g", cmap="Greens")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

visualize_confusion_matrix(confusion_matrix_df)
