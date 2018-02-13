import tensorflow as tf
import logging


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(tf.layers.dense(input_, size))

            t = tf.sigmoid(tf.layers.dense(input_, size) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def lstm_cell_with_dropout(rnn_size, dropout):
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    if dropout is not None:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1. - dropout)
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

    def init_for_training(self, learning_rate=0.01, max_grad_norm=5.0):
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
            logging.info("model has been restored from: %s" % latest_checkpoint)
        else:
            session.run(tf.global_variables_initializer())

    def _char_cnn(self, cnn_input):
        with tf.variable_scope('char_cnn'):
            cnn_input = tf.reshape(cnn_input, [-1, self.max_word_length, self.embedding_size])
            cnn_output = []
            for i, (kernel_width, number_of_features) in enumerate(zip(self.kernel_widths, self.kernel_features)):
                reduced_size = self.max_word_length - kernel_width + 1
                conv = tf.layers.conv1d(cnn_input, number_of_features, kernel_width, padding='valid')
                # conv.shape => [batch_size * max_words_in_sentence, reduced_size, number_of_features]
                pool = tf.layers.max_pooling1d(conv, reduced_size, strides=1, padding='valid')
                # pool.shape => [batch_size * max_words_in_sentence, 1, number_of_features]
                cnn_output.append(tf.squeeze(pool, 1))
            cnn_output = tf.concat(cnn_output, 1)
            # cnn_output.shape => [batch_size * max_words_in_sentence, sum(self.kernel_features)]
        return cnn_output

    def _lstm(self, lstm_input):
        with tf.variable_scope('lstm'):
            fw_cell = lstm_cell_with_dropout(rnn_size=self.rnn_size, dropout=self.lstm_dropout)
            bw_cell = lstm_cell_with_dropout(rnn_size=self.rnn_size, dropout=self.lstm_dropout)

        rnn_input = [tf.squeeze(x, [1]) for x in tf.split(lstm_input, self.max_words_in_sentence, 1)]

        outputs, _, _ = tf.nn.static_bidirectional_rnn(
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
