# Problem Definition

The goal of this project is to build a Deep Learning model capable of classifying handwritten digits from 0 to 9.
MNIST dataset contains grayscale images of handwritten digits. Each image represents one of ten possible classes, corresponding to the digit value.

Given an input image: X ∈ R^(28×28)
the model learns a function:
f(X) → y
where y is the predicted digit class.


# DataSet
[picturesss]
The MNIST (Modified National Institute of Standards and Technology) dataset is one of the most widely used benchmark datasets in the field of computer vision and deep learning.
It consists of handwritten digit images collected from a large number of people. The main objective is to train a machine learning model that can automatically recognize digits from images. Detailed dataset information in the table below.

| Property           | Description          |
| ------------------ | -------------------- |
| Dataset Name       | MNIST                |
| Task               | Image Classification |
| Number of Classes  | 10                   |
| Image Type         | Grayscale            |
| Image Size         | 28 × 28 pixels       |
| Number of Channels | 1                    |
| Training Samples   | 60,000               |
| Test Samples       | 10,000               |
| Pixel Value Range  | 0 - 255              |

## images (X)

Each sample in the MNIST dataset consists of two main components:
1. Image (Input Feature)
The input is a grayscale image represented as a matrix:
$$ X \in R^{28 \times 28} $$
Each pixel contains an intensity value representing the amount of brightness.
A pixel value of:
0 → black background
255 → maximum intensity (white)
Image Shape:
28 × 28

An Example:
[
 [0,0,0,...],
 [0,15,120,...],
 [0,200,255,...],
 ...
]

## labels (y)

Each image has an associated label representing the actual digit.

| Image           | Label          |
| ------------------ | -------------------- |
| Handwritten digit "3"       | 3               |
| Handwritten digit "7"               | 7 |

The model tries to learn the relationship:  Image→Digit

# Loading and Preprocessing The Dataset

1. Converting Images into Tensors

The original MNIST images are stored as grayscale images with pixel intensity values ranging from:

$$ 0 \rightarrow 255 $$

where:

0 represents a black pixel
255 represents maximum brightness

Neural networks cannot directly process image files, so the images are converted into PyTorch tensors.

After conversion:

$$ Pixel\ Values \in [0,255] $$

become:

$$ Tensor\ Values \in [0,1] $$

This allows PyTorch to perform efficient tensor operations during training.

2. Normalization

After converting images into tensors, normalization is applied.

Normalization transforms the pixel values using:

$$ x_{normalized}=\frac{x-\mu}{\sigma} $$

where:
x is the original pixel value ,  μ is the mean of the dataset  and  σ the standard deviation of the dataset.

Normalization scales the input data to have a mean close to 0 and a standard deviation close to 1. This improves training stability and helps the neural network converge faster during optimization.


```python
transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.1307,),(0.3081,))])
trainDataset = datasets.MNIST(root="data",train=True,download=True,transform=transform)
testDataset = datasets.MNIST(root="data",train=False,download=True,transform=transform)
```

# using dataloaders to apply Mini-Batch Gradiant Descent

DataLoaders are used to efficiently load and organize data into mini-batches during training.

instead of:
All 60,000 images
        |
        v
Forward Pass
        |
        v
Loss Calculation
        |
        v
Weight Update

We use:

Dataset
   |
   v
Batch 1 → Forward → Loss → Update
Batch 2 → Forward → Loss → Update
Batch 3 → Forward → Loss → Update
            ....
 
 this method of training reduces memory usage, make training faster and provides more stable and efficient gradient updates.
```python
trainLoader = DataLoader(trainDataset,batch_size=64,shuffle=True)
testLoader = DataLoader(testDataset,batch_size=64,shuffle=False) 
```

## Model


structure of neural network:
[pic2]

| Layer | Output Size | Function |
|------|-------------|----------|
| Input | 784 | Flattened MNIST image |
| Fully Connected | 128 | Linear + ReLU |
| Fully Connected | 64 | Linear + ReLU |
| Output | 10 | Probability of belonging to every Digit class (0-9) |


**We don't use Softmax function at last layer, because it is automatically applies in CrossEntropy Function for Calculating loss**

# Train and test the Model

The training process is implemented using a custom training loop in PyTorch.

For each epoch, the model is set to training mode and receives mini-batches of images from the DataLoader. The optimizer gradients are cleared, the forward pass is performed, and the Cross-Entropy loss is calculated between the predicted outputs and the true labels.
The gradients are then computed using backpropagation, and the optimizer updates the model parameters to minimize the loss. A learning rate scheduler can optionally be used to constantly lower the learning rate during training.
The model performance is monitored by calculating the training loss and accuracy after each epoch.

## model results
it's a simple database and only 5 epoch were enough to get a good acuuracy.

epoch 1, Avg Loss : 7.8393454916076735

epoch 2, Avg Loss : 3.2589192316518165

epoch 3, Avg Loss : 2.357329752325313

epoch 4, Avg Loss : 1.817657215629879

epoch 5, Avg Loss : 1.4462757629353291

**Train Accuracy :  0.965**

**Test Accuracy : 0.978**

Train Accuracy is high, which indicates that **the model doesn't have Underfitting** and Test Acuuracy is not lower which shows that **the model doesn't have Overfitting**

**Confusion Matrix** : 
[pic3]

## seeing few images of wrong predictions
[pic4]

# Inference

On the last part, we test model with datas that are not even in the testSet.
This is an image of a handwritten 7 drawn using Microsoft Paint and resized to 28×28 pixels.

[pic5]

[pic6]



