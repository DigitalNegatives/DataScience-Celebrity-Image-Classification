## celeb_keras

The code will be coming soon.

#### Description
This model is similar to the 02_celeb_transfer_learning but the bottlenecks are now used as a regularizer. Also the code is rewritten in Keras.

#### Implementation

The code was written in Keras, but the backend remained TensorFlow

#### Preprocessing
1. Prepare bottleneck values for all images
2. Filter on images with two eyes and crop image
3. Crop faces
4. Align faces

#### Model:
* layers 1 - 3: Convolution
* layer 4: Fully connected
* layer 5: Dropout
* layer 6: Fully connected
* layer 7: Softmax

#### Cost
1. Cross entropy of image labels
2. Bottlenecks as regularizer
