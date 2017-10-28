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


def create_rnn_cell(rnn_size, dropout):
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    if dropout > 0.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
    return cell


def model():
    batch_size = 20
    max_words_in_sentence = 50
    max_word_length = 30
    char_vocab_size = 100
    num_output_classes = 20
    embedding_size = 16
    kernel_widths = [1, 2, 3, 4, 5, 6, 7]
    kernel_features = [25 * w for w in kernel_widths]
    num_highway_layers = 2
    rnn_size = 650
    dropout = 0.5

    input_ = tf.placeholder(tf.int32, [-1, max_words_in_sentence, max_word_length])

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
    highway_output = highway(cnn_output, cnn_output.shape[-1], num_layers=num_highway_layers)

    cell = create_rnn_cell(rnn_size=rnn_size, dropout=dropout)
    initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)

    highway_output = tf.reshape(highway_output, [batch_size, max_words_in_sentence, -1])
    rnn_input = [tf.squeeze(x, [1]) for x in tf.split(highway_output, max_words_in_sentence, 1)]

    outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell, rnn_input,
                                                         initial_state=initial_rnn_state, dtype=tf.float32)

    logits = []
    matrix = weight_variable(shape=[outputs[0].shape[-1], num_output_classes])
    bias_term = bias_variable(shape=[num_output_classes])
    for output in outputs:
        logits.append(tf.matmul(output, matrix) + bias_term)

    return {}


def loss(logits, batch_size, max_words_in_sentence):
    targets = tf.placeholder(tf.int16, [batch_size, max_words_in_sentence])
    target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, max_words_in_sentence, 1)]
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_list))
    return targets, loss
