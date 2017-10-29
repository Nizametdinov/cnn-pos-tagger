import tensorflow as tf
import time
import numpy as np

from data_reader import DataReader
from vocab import Vocab
from tensor_generator import TensorGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_SAVE_PATH = './model/model.ckpt'

def weight_variable(shape):
    initial = tf.truncated_normal(list(map(int, shape)), stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=list(map(int, shape)))
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
    if dropout is not None:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
    return cell


def model(
        max_words_in_sentence,
        max_word_length,
        char_vocab_size,
        num_output_classes,
        embedding_size=16
):
    kernel_widths = [1, 2, 3, 4, 5, 6, 7]
    kernel_features = [25 * w for w in kernel_widths]
    num_highway_layers = 2
    rnn_size = 650

    input_ = tf.placeholder(tf.int32, [None, max_words_in_sentence, max_word_length])

    initial_embeddings = tf.truncated_normal([char_vocab_size, embedding_size])
    embeddings = tf.Variable(initial_embeddings, name='embeddings')
    cnn_input = tf.nn.embedding_lookup(embeddings, input_)

    cnn_input = tf.reshape(cnn_input, [-1, 1, max_word_length, embedding_size])

    cnn_output = []
    for kernel_width, kernel_feature_size in zip(kernel_widths, kernel_features):
        reduced_size = max_word_length - kernel_width + 1
        conv = conv2d(cnn_input, kernel_feature_size, 1, kernel_width)
        pool = tf.nn.max_pool(conv, [1, 1, reduced_size, 1], strides=[1, 1, 1, 1], padding='VALID')
        cnn_output.append(tf.squeeze(pool, [1, 2]))

    cnn_output = tf.concat(cnn_output, 1)
    highway_output = highway(cnn_output, cnn_output.shape[-1], num_layers=num_highway_layers)

    dropout = tf.placeholder(tf.float32)
    fw_cell = create_rnn_cell(rnn_size=rnn_size, dropout=dropout)
    bw_cell = create_rnn_cell(rnn_size=rnn_size, dropout=dropout)
    # TODO: use trainable initial state
    # initial_rnn_state = fw_cell.zero_state(tf.shape(input_)[0], dtype=tf.float32)

    highway_output = tf.reshape(highway_output, [-1, max_words_in_sentence, int(highway_output.shape[-1])])
    rnn_input = [tf.squeeze(x, [1]) for x in tf.split(highway_output, max_words_in_sentence, 1)]

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
        fw_cell, bw_cell, rnn_input, dtype=tf.float32
    )

    logits = []
    matrix = weight_variable(shape=[outputs[0].shape[-1], num_output_classes])
    bias_term = bias_variable(shape=[num_output_classes])
    for output in outputs:
        logits.append(tf.matmul(output, matrix) + bias_term)

    return input_, logits, dropout


def loss(logits, batch_size, max_words_in_sentence):
    targets = tf.placeholder(tf.int32, [None, max_words_in_sentence], name='targets')
    target_mask = tf.placeholder(tf.float32, [None, max_words_in_sentence], name='targe_mask')
    target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, max_words_in_sentence, 1)]
    target_mask_list = [tf.squeeze(x, [1]) for x in tf.split(target_mask, max_words_in_sentence, 1)]

    loss = tf.reduce_mean(
        tf.multiply(target_mask_list,
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_list)))

    predictions = tf.concat([tf.reshape(tf.argmax(logit, 1), [-1, 1]) for logit in logits], 1)

    correct_predictions = [
        tf.logical_and(
            tf.not_equal(tf.cast(target, tf.int64), 0),
            tf.equal(tf.cast(target, tf.int64), tf.argmax(logit, 1))
        ) for target, logit in zip(target_list, logits)]

    accuracy = sum(
        tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) for correct_prediction in correct_predictions
    ) / sum(tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(target, tf.int64), 0), tf.float32)) for target in target_list)

    return targets, target_mask, loss, accuracy, predictions


def train(loss, learning_rate=1.0, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # SGD learning parameter
    # TODO: why not placeholder?
    learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)

    # collect all trainable variables
    tvars = tf.trainable_variables()
    grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return train_op, learning_rate, global_step, global_norm


def batches(*args, batch_size=20):
    for i in range(0, len(args[0]), batch_size):
        yield [x[i:i+batch_size] for x in args]


def print_classiffication_report(y_true, y_pred, vocab):
    labels = np.array(vocab._index2part)[np.unique([y_true, y_pred])]
    print(classification_report(y_true.flatten(), y_pred.flatten(), target_names=labels, digits=3))


def save_model(sess, path = MODEL_SAVE_PATH):
     saver = tf.train.Saver()
     saver.save(sess, path)

def restore_model(sess, path = MODEL_SAVE_PATH):
     saver = tf.train.Saver()
     saver.restore(sess, path)

def train_model(data_file='data/sentences.xml', epochs=2):
    with tf.Session() as session:
        loader = DataReader(data_file)
        loader.load()
        vocab = Vocab(loader)
        vocab.load()
        tensor_generator = TensorGenerator(loader, vocab)

        batch_size = 20
        report_step = 10

        input_, logits, dropout = model(
            max_words_in_sentence=tensor_generator.max_sentence_length,
            max_word_length=tensor_generator.max_word_length,
            char_vocab_size=vocab.char_vocab_size(),
            num_output_classes=vocab.part_vocab_size()
        )
        # TODO: rename variable loss_
        targets, target_mask, loss_, accuracy, predictions = loss(
            logits=logits,
            batch_size=batch_size,
            max_words_in_sentence=tensor_generator.max_sentence_length
        )
        train_op, learning_rate, global_step, global_norm = train(loss=loss_)

        start_time = time.time()
        session.run(tf.global_variables_initializer())

        input_tensor = tensor_generator.chars_tensor
        target_tensor = tensor_generator.target_tensor
        target_mask_tensor = tensor_generator.target_mask

        train_x, test_x, train_y, test_y, train_mask, test_mask, train_sentences, test_sentences = \
            train_test_split(input_tensor, target_tensor, target_mask_tensor,
                             tensor_generator.sentences, random_state=0, train_size=0.8)
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        for epoch in range(epochs):
            count = 0
            for x, y, y_mask in batches(train_x, train_y, train_mask, batch_size=batch_size):
                if x.shape[0] != batch_size:
                    continue
                count += 1
                loss_value, _, gradient_norm, step = session.run([
                    loss_,
                    train_op,
                    global_norm,
                    global_step
                ], {
                    input_: x,
                    targets: y,
                    target_mask: y_mask + 0.01,
                    dropout: 0.5
                })

                if count % report_step == 0:
                    test_loss_value, accuracy_value, predicted = session.run([
                        loss_, accuracy, predictions
                    ], {
                        input_: test_x[:200],
                        targets: test_y[:200],
                        target_mask: test_mask[:200],
                        dropout: 0.
                    })
                    elapsed = time.time() - start_time
                    print(
                        '\n%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f elapsed = %.4fs, grad.norm=%6.8f' % (
                            step,
                            epoch, count,
                            train_x.shape[0]/batch_size,
                            loss_value,
                            np.exp(loss_value),
                            elapsed,
                            gradient_norm))
                    print('        test loss = %6.8f, test accuracy = %6.8f' % (test_loss_value, accuracy_value))
                    if count % 10 == 0:
                        print_classiffication_report(test_y[:200], predicted, vocab)
                else:
                    print('.', end='', flush=True)
        test_loss_value, accuracy_value, predicted = session.run([
            loss_, accuracy, predictions
        ], {
            input_: test_x,
            targets: test_y,
            target_mask: test_mask,
            dropout: 0.
        })
        print('Final test loss = %6.8f, test perplexity = %6.8f, test accuracy = %6.8f' %
              (test_loss_value, np.exp(test_loss_value), accuracy_value))
        print_classiffication_report(test_y, predicted, vocab)

        save_model(session)


def evaluate_model():
    global vocab, tensor_generator
    data_file = 'data/sentences.xml'
    loader = DataReader(data_file)
    loader.load()
    vocab = Vocab(loader)
    vocab.load()
    tensor_generator = TensorGenerator(loader, vocab)
    with tf.Session() as session:
        input_, logits, dropout = model(
            max_words_in_sentence=tensor_generator.max_sentence_length,
            max_word_length=tensor_generator.max_word_length,
            char_vocab_size=vocab.char_vocab_size(),
            num_output_classes=vocab.part_vocab_size()
        )

        _targets, _target_mask, _loss_, _accuracy, predictions = loss(
            logits=logits,
            batch_size=0,
            max_words_in_sentence=tensor_generator.max_sentence_length
        )

        print('graph init finished')

        restore_model(session)

        print('model restored')

        sentences = [
            ['проверка', 'связи', 'прошла'],
            ['глокая', 'куздра', 'штеко', 'будланула', 'бокра', 'и', 'кудрячит', 'бокрёнка'],
        ]
        input_tensors = tensor_generator.tensor_from_sentences(sentences)

        print(input_tensors)
        predicted = session.run([predictions], {input_: input_tensors, dropout: 0.0})

        for sentence, sentence_prediction in zip(sentences, predicted[0]):
            for word, word_prediction in zip(sentence, sentence_prediction):
                print('word: ', word, ' predicted part of speech: ', vocab._index2part[word_prediction])
            print()


if __name__ == '__main__':
    # train_model()

    evaluate_model()


