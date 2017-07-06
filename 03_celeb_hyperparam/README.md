## Hyperparameter

#### Implementation
The model was created using TensorFlow.

#### Preprocessing
1. Aligned images, cropped images to 150x150, and applied a Gaussian mask
2. Prepare bottleneck values for all images

#### Model
* layers 1 - 3: Convolution
* layer 4: Fully connected
* layer 5: Dropout
* layer 6: Fully connected (bottleneck)
* layer 7: Softmax

#### Cost main_inception_cosine
1. Cross entropy of image labels
2. Cosine distance bottleneck regularizer

&nbsp;
### Description

In the previous section a number of preprocessing techniques were used, but unfortunately the test results did not show an improvement. Convolutional networks however have a number of parameters. The parameters that were used for the previous section may not of been the optimal settings. This section uses hyperparameter search to test a number of different parameter settings.

&nbsp;
#### Parameters
filters: is the number of filters to produce at each of the convolutional layers.
fiter_size: is the size of the convolutional kernel.
activation_conv: is the type of activation used in the convolutional layers
activation_fc: is the type of activation used in the fully connected layers
learning_rate_label: is the learning rate used during the classification training
dropout_train: is the dropout percentage used during training. This term is a bit misleading as this value is actually how much of the connections to keep. For example a value of .75 implies that 75% of the connections remain while 25% will be removed.

&nbsp;
#### Code
To help automate the parameter search process the code is no longer in a Jupiter notebook. Also, to make the code easier to read many of the functions have been moved to model.py. main.py is the file to execute. Lastly, this version on is only using cosine distance as a regulator.

&nbsp;
#### Hyperparameter
The library used to conduct the hyperparameter search is hpyeropt. This library provides an easy way to provide a range or a choice of parameters. 

hp.choice allows for one value to be picked from an array of values. This is used in dropout_train. hp.uniform allows for a uniform distribution of value to be picked between a minimum and maximum value. Below shows an example of both types.

'Dropout_train':hp.choice('dropout_train', [0.5, 0.6, 0.7])
'learning_rate_label':hp.uniform('learning_rate_label', 0.0001, 0.0007)

&nbsp;
#### Configuration
The configuration for the complete network including the hyperparameter configuration is in the configuration.py file.

&nbsp;
#### Results
A 160 different configurations were tested. Of the 160, 16 resulted in an accuracy of 87.50%, which is a great improvement from before. There were 5 that provided the best result of 93.75%. The 5 that provided the best results are very consistent. The only difference among the five is the learning rate. The setting for all the parameters are shown below. The activation methods are the major reasons for the great improvement. The activations for the previous section were both relu.


| filter 	| filter_sizes 	| activation_conv 	| activation_fc 	| dropout_train 	| learning_rate_label 	|
|----------	|--------------	|-----------------	|---------------	|---------------	|---------------------	|
| 64,64,32 	| 5,5,3 		| tanh 				| sigmoid			| 0.6 				| 0.000180 - 0.000203 	|

&nbsp;
#### Report
An html file that contains all the parameters for each of the test was generated and placed in the results directory.
