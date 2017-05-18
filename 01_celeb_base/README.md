## Celeb_base

#### Implementation

The model was created using TensorFlow.

#### Preprocessing
1. Crop image to 100x100

#### Model:
* layers 1 - 3: Convolution
* layer 4: Fully connected
* layer 5: Dropout
* layer 6: Fully connected
* layer 7: Softmax

#### Cost
* Cross entropy of image labels

&nbsp;
### Description

#### Preprocessing
Input data was prepared in advance. Celebrity images that did not have two eyes were filtered out. Valid faces were then cropped to 100x100.

#### Model
The model used contains three convolution layers followed by three fully connected layers and an output layer.

#### Results
Training often reaches 1.0, however the top test accuracy is 0.625.

At the bottom of the notebook is a T-SNE plot. T-SNE is a method to visualize multiple dimensions to lower dimensions. This plot shows the points of each image in two dimensions. Each class of image has a specific color. Ideally each class should be tightly clustered together, but the results of the notebook do not show such clustering.
