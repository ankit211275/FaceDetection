# **Face Detection Model**

A custom-built face detection model using TensorFlow, trained from scratch with labeled data. The model detects faces and provides bounding box coordinates both for **live** or any given image.

### Requirements

TensorFlow 2.x

NumPy

OpenCV

Matplotlib

### Model Overview

This model leverages transfer learning by using a pre-trained VGG16 model as a feature extractor. The VGG16 architecture is used to extract high-level features from images, which are then utilized for both classification and regression tasks.

#### Transfer Learning Approach:

**Feature Extraction:** The pre-trained VGG16 model, trained on the ImageNet dataset, is used as a fixed feature extractor. The model layers up to the fully connected layers are frozen, and only the final layers are fine-tuned for our task.

**Classification Task:** A custom fully connected layer is added on top of the VGG16 model to classify the input image as either containing a face (1) or not (0). This is handled by a binary cross-entropy loss.

**Regression Task:** Another custom layer is added for the regression task, where we predict the coordinates of the bounding box that encloses the face. The output of this layer consists of four values representing [x_min, y_min, x_max, y_max]. This task is trained using a mean squared error (MSE) loss.

**Loss Functions:**

**Classification Loss:** Binary Cross-Entropy for face classification.
**Localization Loss:** Mean Squared Error (MSE) for bounding box coordinates.
