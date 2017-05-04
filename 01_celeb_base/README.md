## Celeb_base

#### Implementation

The model was created using TensorFlow.

#### Preprocessing
1. Filter on images with two eyes and crop image

#### Model:
* layers 1 - 3: Convolution
* layer 4: Fully connected
* layer 5: Dropout
* layer 6: Fully connected
* layer 7: Softmax

#### Cost
* Cross entropy

#### T-SNE

T-SNE is a method to visualize high-dimensional data by projecting them
to lower dimensions. In this case two dimensions to view the clustering
of celebrity entities.
