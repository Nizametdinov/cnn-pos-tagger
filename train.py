import logging
import numpy as np
import os
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_reader import DataReader
from download_data import OPEN_CORPORA_DEST_FILE
from model import CharCnnLstm
from tensor_generator import TensorGenerator
from vocab import Vocab


MODEL_SAVE_PATH = './model'
MODEL_FILE_NAME = './model/model.ckpt'


def batches(*args, batch_size=20):
    for i in range(0, len(args[0]), batch_size):
        yield [x[i:i+batch_size] for x in args]


def classification_report_with_labels(y_true, y_pred, vocab):
    labels = np.array(vocab._index2part)[np.unique([y_true, y_pred])]
    return classification_report(y_true.flatten(), y_pred.flatten(), target_names=labels, digits=3)


def train_model(data_file=OPEN_CORPORA_DEST_FILE, epochs=2, logger=logging.getLogger()):
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
        model.init_for_training(learning_rate=0.001)
        model.restore_latest_or_init(session, MODEL_SAVE_PATH)

        start_time = time.time()
        session.run(tf.global_variables_initializer())

        input_tensor = tensor_generator.chars_tensor
        target_tensor = tensor_generator.target_tensor
        target_mask_tensor = tensor_generator.target_mask

        train_x, test_x, train_y, test_y, train_mask, test_mask, train_sentences, test_sentences = \
            train_test_split(input_tensor, target_tensor, target_mask_tensor,
                             tensor_generator.sentences, random_state=0, train_size=0.8)
        logger.info('train/test shapes: %s, %s / %s, %s' % (train_x.shape, train_y.shape, test_x.shape, test_y.shape))

        for epoch in range(epochs):
            count = 0
            for x, y, y_mask in batches(train_x, train_y, train_mask, batch_size=batch_size):
                # if x.shape[0] != batch_size:
                #     continue
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
                    logging.info('        test loss = %6.8f, test accuracy = %6.8f' % (test_loss_value, accuracy_value))
                    logging.debug('\n' + classification_report_with_labels(test_y[:200], predicted, vocab))
                    log_level = logging.INFO
                else:
                    log_level = logging.DEBUG
                logging.log(log_level, '%6d: %d [%5d/%5d], train_loss = %6.8f elapsed = %.4fs, grad.norm=%6.8f' % (
                            step, epoch, count, train_x.shape[0] / batch_size,
                            loss_value,
                            time.time() - start_time,
                            gradient_norm))
                if count % save_step == 0:
                    logging.info('Saving model...')
                    model.save_model(session, MODEL_FILE_NAME)

            logging.info('Saving model...')
            model.save_model(session, MODEL_FILE_NAME)
        test_loss_value, accuracy_value, predicted = session.run([
            model.loss, model.accuracy, model.predictions
        ], {
            model.input: test_x,
            model.targets: test_y,
            model.target_mask: test_mask,
            model.lstm_dropout: 0.
        })
        logger.info('Final test loss = %6.8f, test accuracy = %6.8f' % (test_loss_value, accuracy_value))
        logging.info('\n' + classification_report_with_labels(test_y, predicted, vocab))


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.isdir('log'):
        os.mkdir('log')
    file_handler = logging.FileHandler('log/training.log', encoding='UTF-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    train_model(logger=logger)
