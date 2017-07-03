import tensorflow as tf

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable_xavier(name, shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.get_variable(
                            name=name,
                            shape=shape,
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    return initial

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1,1,1,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool_2x2(x, ksize, strides):
    return tf.nn.max_pool(x, ksize=ksize,
                        strides=strides, padding='SAME')


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#X must be 4D
def convolutions(X, n_filters, filter_sizes, filter_strides, act=tf.nn.relu):
    current_in = X
    n_input_chs = X.get_shape().as_list()[3]
    Ws = []
    shapes = []

    # Build the encoder
    for layer_i, n_output_chs in enumerate(n_filters):
        with tf.variable_scope('convolution/{}'.format(layer_i)):
            shapes.append(current_in.get_shape().as_list())
            W = weight_variable_xavier(
                                name = "W{}".format(layer_i),
                                shape = [filter_sizes[layer_i],
                                 filter_sizes[layer_i],
                                 n_input_chs, n_output_chs])
            pre_activation = conv2d(current_in, W, filter_strides)
            h = act(pre_activation)
            Ws.append(W)
            current_in = h
            n_input_chs = n_output_chs
            print("h.shape:", h.get_shape().as_list())
    shapes.append(current_in.get_shape().as_list())
    print("shapes:",shapes)
    return h, Ws

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    if (len(input_tensor.get_shape().as_list())==4):
        input_shape = input_tensor.get_shape().as_list()
        input_tensor = tf.reshape(input_tensor, [-1, input_shape[1]*input_shape[2]*input_shape[3]])

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations, preactivate

def drop_layer(h, keep_prob):
    with tf.name_scope('dropout'):
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(h, keep_prob)
    return dropped
