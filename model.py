import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(input_, output_dim, k_h, k_w):
    w = weight_variable([k_h, k_w, input_.get_shape()[-1], output_dim])
    b = bias_variable([output_dim])
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    for idx in range(num_layers):
        g = f(linear(input_, size, scope='highway_lin_%d' % idx))

        t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

        output = t * g + (1. - t) * input_
        input_ = output

    return output


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    # matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
    # bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    matrix = weight_variable(shape=[input_size, output_size])
    bias_term = bias_variable(shape=[output_size])

    return tf.matmul(input_, matrix) + bias_term


def model():
    batch_size = 50
    max_words_in_sentence = 50
    max_word_length = 30
    char_vocab_size = 100
    embedding_size = 16
    kernel_widths = [1, 2, 3, 4, 5, 6, 7]
    kernel_features = [25 * w for w in kernel_widths]
    num_highway_layers = 2

    input_ = tf.placeholder(tf.int32, [batch_size, max_words_in_sentence, max_word_length])

    embeddings = tf.truncated_normal([char_vocab_size, embedding_size])
    cnn_input = tf.nn.embedding_lookup(embeddings, input_)

    cnn_input = tf.reshape(cnn_input, [-1, 1, max_word_length, embedding_size])

    cnn_output = []
    for kernel_width, kernel_feature_size in zip(kernel_widths, kernel_features):
        reduced_size = max_word_length - kernel_width + 1
        conv = conv2d(cnn_input, kernel_feature_size, 1, kernel_width)
        pool = tf.nn.max_pool(conv, [1, reduced_size], strides=[1, 1, 1, 1], padding='VALID')
        cnn_output.append(tf.squeeze(pool, [1, 2]))

    cnn_output = tf.concat(cnn_output, 1)
    highway(cnn_output, cnn_output.shape[-1], num_layers=num_highway_layers)
