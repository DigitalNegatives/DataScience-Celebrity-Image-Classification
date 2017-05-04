## Celeb_base

#### Description

This model is similar to the 01_celeb_base but with the inclusion of transfer
learning. Transfer learning is leveraging a pre-trained model to help guide the training of the target model.

Inception is Google's image classification model which has been trained on
ImageNet data which contains a thousand categories.
The last layer prior to the output layer in the Inception model is informally
known as the Bottleneck layer. Google allows access to this layer to take
advantage of the training the model has completed. All of the celebrity images were passed through the Inception model to capture the bottleneck values for each image.

This model uses two phases to train. The first phase trains to the bottleneck values. The second phase trains off the image labels.

This is all done in the hope that it will improve the performance of the target model in classifying celebrities.


#### Implementation

The model was created using TensorFlow.

#### Preprocessing
1. Prepare bottleneck values for all images
2. Filter on images with two eyes and crop image

#### Model:
* layers 1 - 3: Convolution
* layer 4: Fully connected
* layer 5: Dropout
* layer 6: Fully connected
* layer 7: Softmax

#### Cost
1. MSE of the bottlenecks
2. Cross entropy of image labels
