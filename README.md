# Fully Connected NN vs. CNN for MNIST Classification

## Overview

This project investigates the impact of architectural design on handwritten digit classification using the MNIST dataset. 
A fully connected neural network is compared with a convolutional neural network to analyze how exploiting spatial information affects image classification performance.
The experiments are conducted under the same training conditions, and models are evaluated using classification accuracy, loss curves, and error analysis.

<img width="634" height="315" alt="image" src="https://github.com/user-attachments/assets/db7cace1-28f3-4dde-a986-32ce206cb690" />


## Models

### 1. Fully Connected Neural Network (Baseline)

The first model is a multilayer perceptron (MLP) used as a baseline approach. The input image is flattened into a 784-dimensional vector and processed through fully connected layers.

Architecture:

```
Input (28×28)
      |
   Flatten
      |
Linear(784 → 128)
      |
     ReLU
      |
Linear(128 → 64)
      |
     ReLU
      |
Linear(64 → 10)
```

---

### 2. Convolutional Neural Network (CNN)

The second model is a CNN designed to preserve spatial information and learn hierarchical visual features through convolutional operations.

The architecture includes convolutional blocks with Batch Normalization, ReLU activation, max pooling, and dropout regularization.

Architecture:

```
Input (1×28×28)

Conv(32) → Batch normalization → ReLU
Conv(32) → Batch normalization → ReLU
MaxPool

Conv(64) → Batch normalization → ReLU
Conv(64) → Batch normalization → ReLU
MaxPool

Flatten

Linear(3136 → 128)
ReLU
Dropout(0.2)

Linear(128 → 10)
```

---

# Experimental Results

The final performance comparison **in 10 Epochs** is summarized below:


| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|------|---------------|-----------------|--------------|----------------|
| Fully Connected Neural Network | 98.23% | 98.22% | 98.22% | 98.22% |
| Convolutional Neural Network | **99.63%** | **99.63%** | **99.62%** | **99.63%** |

The CNN achieved a **1.40% improvement in classification accuracy** compared with the fully connected baseline.

The fully connected neural network achieved a strong baseline performance with an accuracy of 98.23%. This demonstrates that dense architectures are capable of learning discriminative representations even when applied directly to image data.
However, the CNN significantly improved the classification performance, reaching an accuracy of 99.63%. The improvement is mainly attributed to the ability of convolutional layers to preserve spatial information and automatically learn hierarchical visual features.
Unlike the fully connected model, where the image is flattened into a one-dimensional feature vector, the CNN processes the image structure through local receptive fields and parameter sharing. This allows the model to better capture important patterns such as edges, curves, and digit-specific stroke structures.

## Learning Curves

**Fully Connected Neural Network:**
<img width="545" height="489" alt="image" src="https://github.com/user-attachments/assets/72a27a85-5bb2-44ac-92cf-8d110231250b" />


**Convolutional Neural Network:**
<img width="531" height="483" alt="image" src="https://github.com/user-attachments/assets/053888db-4688-40dc-9bb9-5389418d0e4b" />


## Confusion Matrix

**Fully Connected Neural Network:**

<img width="507" height="451" alt="image" src="https://github.com/user-attachments/assets/2c18d0f6-60c5-47b8-83d1-b3be141df694" />

**Convolutional Neural Network:**

<img width="505" height="456" alt="image" src="https://github.com/user-attachments/assets/bd068fcb-4b56-4b10-ba1c-6e50120ba538" />


