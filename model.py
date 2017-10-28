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


def model():
    batch_size = 50
    max_words_in_sentence = 50
    max_word_length = 30
    char_vocab_size = 100
    embedding_size = 16
    kernel_widths = [1, 2, 3, 4, 5, 6, 7]
    kernel_features = [25 * w for w in kernel_widths]

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
