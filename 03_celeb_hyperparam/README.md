## Celeb_hyperparameter

The code will be coming soon.

#### Description
This model is similar to the 02_celeb_transfer_learning but with the inclusion of hyperparameter search.

For each run a different set of parameters are used to conduct the training. This will determine which set of parameters provide the best test results.

Additionally the input images have been cropped to eliminate much of the background. Also the faces have been aligned so that each faces are not tilted as much as possible.

#### Parameters
* learning rate
* activation function
* number of filters
* filter sizes
* dropout rate

#### Implementation

The model was created using TensorFlow.

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
1. MSE of the bottlenecks
2. Cross entropy of image labels
