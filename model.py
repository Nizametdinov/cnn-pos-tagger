import tensorflow as tf
import time
import numpy as np

from data_reader import DataReader
from vocab import Vocab
from tensor_generator import TensorGenerator
from download_data import OPEN_CORPORA_DEST_FILE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_SAVE_PATH = './model'
MODEL_FILE_NAME = './model/model.ckpt'


def conv2d(input_, output_dim, k_h, k_w, scope):
    with tf.variable_scope(scope):
        w = tf.get_variable('kernel', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable('bias', [output_dim], initializer=tf.zeros_initializer())
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    with tf.variable_scope(scope):
        matrix = tf.get_variable("matrix", [output_size, input_size], dtype=input_.dtype,
                                 initializer=tf.contrib.layers.xavier_initializer())
        bias_term = tf.get_variable("bias", [output_size], dtype=input_.dtype, initializer=tf.zeros_initializer())

    return tf.matmul(input_, matrix) + bias_term


def lstm_cell_with_dropout(rnn_size, dropout):
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    if dropout is not None:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1. - dropout)
    return cell


class CharCnnLstm(object):
    def __init__(self, max_words_in_sentence, max_word_length, char_vocab_size, num_output_classes):
        self.max_words_in_sentence = max_words_in_sentence
        self.max_word_length = max_word_length
        self.char_vocab_size = char_vocab_size
        self.num_output_classes = num_output_classes

        self.embedding_size = 16
        self.kernel_widths = [1, 2, 3, 4, 5, 6, 7]
        self.kernel_features = [25 * w for w in self.kernel_widths]
        self.num_highway_layers = 2
        self.rnn_size = 650

        self.input = tf.placeholder(tf.int32, [None, self.max_words_in_sentence, self.max_word_length])
        self.targets = tf.placeholder(tf.int32, [None, self.max_words_in_sentence], name='targets')
        self.target_mask = tf.placeholder(tf.float32, [None, self.max_words_in_sentence], name='target_mask')
        self.lstm_dropout = tf.placeholder(tf.float32)

        self.loss = None
        self.predictions = None
        self.accuracy = None
        self.learning_rate = None

        self.global_step = None
        self.global_norm = None
        self.train_op = None

        self._saver = None

    def saver(self):
        if not self._saver:
            self._saver = tf.train.Saver()
        return self._saver

    def init_for_evaluation(self):
        embeddings = tf.get_variable('char_embeddings',
                                     [self.char_vocab_size, self.embedding_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        cnn_input = tf.nn.embedding_lookup(embeddings, self.input)

        cnn_output = self._char_cnn(cnn_input)
        highway_output = highway(cnn_output, cnn_output.shape[-1], num_layers=self.num_highway_layers)
        highway_output = tf.reshape(highway_output, [-1, self.max_words_in_sentence, int(highway_output.shape[-1])])
        rnn_outputs = self._lstm(highway_output)
        logits = self._rnn_logits(rnn_outputs)
        self._loss(logits)

    def init_for_training(self, learning_rate=0.5, max_grad_norm=5.0):
        self.init_for_evaluation()

        self.learning_rate = tf.Variable(learning_rate, name='learning_rate', trainable=False)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        tvars = tf.trainable_variables()
        grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def save_model(self, session, path):
        self.saver().save(session, path)

    def restore_latest_or_init(self, session, model_dir):
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint:
            self.saver().restore(session, latest_checkpoint)
            print("model has been restored from", latest_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

    def _char_cnn(self, cnn_input):
        with tf.variable_scope('char_cnn'):
            cnn_input = tf.reshape(cnn_input, [-1, 1, self.max_word_length, self.embedding_size])
            cnn_output = []
            for i, (kernel_width, kernel_feature_size) in enumerate(zip(self.kernel_widths, self.kernel_features)):
                reduced_size = self.max_word_length - kernel_width + 1
                conv = conv2d(cnn_input, kernel_feature_size, 1, kernel_width, scope='conv_%d' % i)
                pool = tf.nn.max_pool(conv, [1, 1, reduced_size, 1], strides=[1, 1, 1, 1], padding='VALID')
                cnn_output.append(tf.squeeze(pool, [1, 2]))
            cnn_output = tf.concat(cnn_output, 1)
        return cnn_output

    def _lstm(self, lstm_input):
        with tf.variable_scope('lstm'):
            fw_cell = lstm_cell_with_dropout(rnn_size=self.rnn_size, dropout=self.lstm_dropout)
            bw_cell = lstm_cell_with_dropout(rnn_size=self.rnn_size, dropout=self.lstm_dropout)

        rnn_input = [tf.squeeze(x, [1]) for x in tf.split(lstm_input, self.max_words_in_sentence, 1)]

        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            fw_cell, bw_cell, rnn_input, dtype=tf.float32
        )
        return outputs

    def _rnn_logits(self, rnn_outputs):
        logits = []
        with tf.variable_scope('softmax'):
            matrix = tf.get_variable('matrix', [rnn_outputs[0].shape[-1], self.num_output_classes],
                                     dtype=rnn_outputs[0].dtype,
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [self.num_output_classes], dtype=rnn_outputs[0].dtype,
                                   initializer=tf.zeros_initializer())
        for output in rnn_outputs:
            logits.append(tf.matmul(output, matrix) + bias)
        return logits

    def _loss(self, logits):
        target_list = [tf.squeeze(x, [1]) for x in tf.split(self.targets, self.max_words_in_sentence, 1)]
        target_mask_list = [tf.squeeze(x, [1]) for x in tf.split(self.target_mask, self.max_words_in_sentence, 1)]

        self.loss = tf.reduce_mean(
            tf.multiply(target_mask_list,
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_list)))

        self.predictions = tf.concat([tf.reshape(tf.argmax(logit, 1), [-1, 1]) for logit in logits], 1)

        correct_predictions = [
            tf.logical_and(
                tf.not_equal(tf.cast(target, tf.int64), 0),
                tf.equal(tf.cast(target, tf.int64), tf.argmax(logit, 1))
            ) for target, logit in zip(target_list, logits)]

        self.accuracy = sum(
            tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) for correct_prediction in correct_predictions
        ) / sum(
            tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(target, tf.int64), 0), tf.float32)) for target in target_list
        )


def batches(*args, batch_size=20):
    for i in range(0, len(args[0]), batch_size):
        yield [x[i:i+batch_size] for x in args]


def print_classiffication_report(y_true, y_pred, vocab):
    labels = np.array(vocab._index2part)[np.unique([y_true, y_pred])]
    print(classification_report(y_true.flatten(), y_pred.flatten(), target_names=labels, digits=3))


def train_model(data_file=OPEN_CORPORA_DEST_FILE, epochs=2):
    loader = DataReader(data_file)
    loader.load()
    vocab = Vocab(loader)
    vocab.load()
    tensor_generator = TensorGenerator(loader, vocab)

    with tf.Session() as session:
        batch_size = 20
        report_step = 10
        save_step = 50

        model = CharCnnLstm(
            max_words_in_sentence=tensor_generator.max_sentence_length,
            max_word_length=tensor_generator.max_word_length,
            char_vocab_size=vocab.char_vocab_size(),
            num_output_classes=vocab.part_vocab_size()
        )
        model.init_for_training()
        model.restore_latest_or_init(session, MODEL_SAVE_PATH)

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
                    model.loss,
                    model.train_op,
                    model.global_norm,
                    model.global_step
                ], {
                    model.input: x,
                    model.targets: y,
                    model.target_mask: y_mask + 0.01,
                    model.lstm_dropout: 0.5
                })

                if count % report_step == 0:
                    test_loss_value, accuracy_value, predicted = session.run([
                        model.loss, model.accuracy, model.predictions
                    ], {
                        model.input: test_x[:200],
                        model.targets: test_y[:200],
                        model.target_mask: test_mask[:200],
                        model.lstm_dropout: 0.
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
                if count % save_step == 0:
                    print('Saving model...')
                    model.save_model(session, MODEL_FILE_NAME)

            print('Saving model...')
            model.save_model(session, MODEL_FILE_NAME)
        test_loss_value, accuracy_value, predicted = session.run([
            model.loss, model.accuracy, model.predictions
        ], {
            model.input: test_x,
            model.targets: test_y,
            model.target_mask: test_mask,
            model.lstm_dropout: 0.
        })
        print('Final test loss = %6.8f, test perplexity = %6.8f, test accuracy = %6.8f' %
              (test_loss_value, np.exp(test_loss_value), accuracy_value))
        print_classiffication_report(test_y, predicted, vocab)


def evaluate_model():
    data_file = OPEN_CORPORA_DEST_FILE
    loader = DataReader(data_file)
    loader.load()
    vocab = Vocab(loader)
    vocab.load()
    tensor_generator = TensorGenerator(loader, vocab)
    with tf.Session() as session:
        model = CharCnnLstm(
            max_words_in_sentence=tensor_generator.max_sentence_length,
            max_word_length=tensor_generator.max_word_length,
            char_vocab_size=vocab.char_vocab_size(),
            num_output_classes=vocab.part_vocab_size()
        )
        model.init_for_evaluation()
        model.restore_latest_or_init(session, MODEL_SAVE_PATH)

        print('graph init finished')

        sentences = [
            ['проверка', 'связи', 'прошла'],
            ['глокая', 'куздра', 'штеко', 'будланула', 'бокра', 'и', 'кудрячит', 'бокрёнка'],
        ]
        input_tensors = tensor_generator.tensor_from_sentences(sentences)

        print(input_tensors)
        predicted = session.run([model.predictions], {model.input: input_tensors, model.lstm_dropout: 0.0})

        for sentence, sentence_prediction in zip(sentences, predicted[0]):
            for word, word_prediction in zip(sentence, sentence_prediction):
                print('word: ', word, ' predicted part of speech: ', vocab.index_to_speech_part_human(word_prediction))
            print()


if __name__ == '__main__':
    train_model()

    # evaluate_model()
