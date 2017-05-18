## celeb_transfer_learning

#### Implementation
The model was created using TensorFlow.

#### Preprocessing
1. Aligned images, cropped images to 150x150, and applied a gaussian mask
2. Prepare bottleneck values for all images

#### Model
* layers 1 - 3: Convolution
* layer 4: Fully connected
* layer 5: Dropout
* layer 6: Fully connected (bottleneck)
* layer 7: Softmax

#### Cost main_inception_2step
1. MSE of the bottlenecks
2. Cross entropy of image labels

#### Cost main_inception_cosine
1. Cross entropy of image labels
2. Cosine distance bottleneck regularizer


&nbsp;
### Description

#### Preprocessing
Instead of just cropping raw images to 100x100 as in 01_celeb_base, additional processing was done to reduce variance and to highlight important areas of the face.

First each face was aligned to be in the same location. This would rotate any faces that were at a slant. The idea was to align all the faces so that the location of the eyes, nose and mouth has as little variance as possible between all of the images.

Next a 2d gaussian filter was placed on each image. This would highlight the center of each image and darken the outer portion of the image. The idea was to place less emphasis on hair and any background in the image.

Lastly the image was cropped to 150x150, instead of 100x100. This provided more pixels for the model to learn from.

&nbsp;

#### Model
There are two notebooks in this section that implement transfer learning, which leverages the learning of a pre-trained model to help guide the training of the model under development.

Inception is Google's image classification model that has been trained on ImageNet data which contains a thousand categories. The last layer prior to the output layer in the Inception model is informally known as the Bottleneck layer. Google allows access to this layer to take advantage of the training the model has completed. All of the celebrity images were passed through the Inception model to capture the bottleneck values for each image.

#### main_inception_2step
This model uses two phases to train. The first phase trains to the bottleneck values.

The Inception model bottleneck layer consists of 2048 nodes. To match this the bottleneck layer, the layer before the output, of this model also consists of 2048 nodes. The difference between the captured bottleneck values from the Inception model and this model's bottleneck values are trained to be minimized.

The second phase trains off the image labels.

Other changes compared to 01_celeb_base is that the mean of the train image set is subtracted from each training batch. This is to help the model learn the variance instead of the mean.

#### main_inception_cosine
This model was derived from main_inception_2step and implements a regularizer in the image classification optimizer. The regularizer is the cosine distance between the bottlenecks generated from the Inception model and the bottlenecks from this model's bottleneck layer.

&nbsp;
#### Results
At the bottom of each notebook is a T-SNE plot. T-SNE is a method to visualize multiple dimensions to lower dimensions. This plot shows the points of each image in two dimensions. Each class of image has a specific color. Ideally each class should be tightly clustered together, but the results of both notebooks do not show such clustering. The accuracy also proves this as the best accuracy was 0.56. This is not surprising considering that the data set is not very large.

The 01_celeb_base model was much simpler yet produced just as accurate results. This goes to show that even with the addition of various techniques there is no guarantee that it will outperform a simpler model.

There are many other factors that could be tested to help improve the results such as trying different activation functions, increasing the number of filters in the convolution layers, increasing the number of nodes in the fully-connected layers, and adding more layers to the network.


#### Data Set Note
The Microsoft data set is very dirty. The images for each celebrity often include an incorrect person. Images also include drawings instead of photos. The small subset of data in this repo was used to easily execute the notebooks but does amplify the dirty data issue. Dealing with dirty data is the reality of data science.


References:
The link below was used for reference and some of the code is based off of this repo.
https://github.com/pkmital/CADL/blob/master/session-3/lecture-3.ipynb

The below repo was used as a basis for aligning the faces.
https://github.com/matthewearl/faceswap/blob/master/faceswap.py
