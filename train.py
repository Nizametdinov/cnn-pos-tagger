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


def classification_report_with_labels(y_true, y_pred, vocab):
    labels = np.array(vocab._index2part)[np.unique([y_true, y_pred])]
    return classification_report(y_true.flatten(), y_pred.flatten(), target_names=labels, digits=3)


def init_datasets(loader, vocab, logger, batch_size, val_batch_size=100):
    max_word_length = len(loader.get_longest_word()) + 2

    def make_generator(sentences):
        def data_generator():
            WORD_START = '{'
            WORD_END = '}'
            for sentence in sentences:
                targets = []
                mask = [1.] * len(sentence)
                char_tensor = np.zeros((len(sentence), max_word_length))
                for j, (word, target_class) in enumerate(sentence):
                    targets.append(vocab.part_to_index(target_class))
                    word = WORD_START + word + WORD_END
                    for k, char in enumerate(word):
                        char_tensor[j, k] = vocab.char_to_index(char)
                yield char_tensor, targets, mask
        return data_generator

    train_sentences, val_sentences = train_test_split(loader.sentences, random_state=0, train_size=0.8)
    nb_train_batches = int(np.ceil(len(train_sentences) / batch_size))
    logger.info('train/validation set size: %d / %d' % (len(train_sentences), len(val_sentences)))

    data_types = tf.int32, tf.int64, tf.float32
    data_shapes = tf.TensorShape([None, max_word_length]), tf.TensorShape([None]), tf.TensorShape([None])
    padded_shapes = ([None, max_word_length], [None], [None])

    train_dataset = tf.data.Dataset.from_generator(
        make_generator(train_sentences), data_types, data_shapes
    ).padded_batch(batch_size, padded_shapes)

    val_dataset = tf.data.Dataset.from_generator(
        make_generator(val_sentences), data_types, data_shapes
    ).padded_batch(val_batch_size, padded_shapes)

    return train_dataset, val_dataset, max_word_length, nb_train_batches


def train_model(data_file=OPEN_CORPORA_DEST_FILE, epochs=2, logger=logging.getLogger()):
    loader = DataReader(data_file)
    loader.load()
    vocab = Vocab(loader)
    vocab.load()

    batch_size = 20
    report_step = 10

    with tf.Session() as session:
        train_dataset, val_dataset, max_word_length, nb_train_batches = init_datasets(loader, vocab, logger, batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = iterator.make_initializer(train_dataset)
        val_initializer = iterator.make_initializer(val_dataset)

        input_tensor, target_tensor, target_mask_tensor = iterator.get_next()

        model = CharCnnLstm(
            max_words_in_sentence=None,
            max_word_length=max_word_length,
            char_vocab_size=vocab.char_vocab_size(),
            num_output_classes=vocab.part_vocab_size(),
            input_tensor=input_tensor,
            target_tensor=target_tensor,
            target_mask_tensor=target_mask_tensor
        )
        model.init_for_training(learning_rate=0.001)
        model.restore_latest_or_init(session, MODEL_SAVE_PATH)

        for epoch in range(epochs):
            session.run(train_initializer)

            batch = 0
            start_time = time.time()
            while True:
                batch += 1
                try:
                    loss_value, _, gradient_norm, step = session.run([
                        model.loss,
                        model.train_op,
                        model.global_norm,
                        model.global_step
                    ], {
                        model.lstm_dropout: 0.5
                    })
                    log_level = logging.INFO if batch % report_step == 0 else logging.DEBUG
                    logging.log(log_level, '%6d: %d [%5d/%5d], train_loss = %6.8f elapsed = %.4fs, grad.norm=%6.8f' % (
                        step, epoch, batch, nb_train_batches,
                        loss_value,
                        time.time() - start_time,
                        gradient_norm))
                except tf.errors.OutOfRangeError:
                    break

            session.run(val_initializer)

            run_validations(session, model, vocab, target_tensor, logger)

            logging.info('Saving model...')
            model.save_model(session, MODEL_FILE_NAME)


def run_validations(session, model, vocab, target_tensor, logger):
    targets = []
    predictions = []
    loss = 0
    nb_batches = 0
    while True:
        try:
            batch_loss, batch_target, batch_predictions = session.run([
                model.loss, target_tensor, model.predictions
            ], {
                model.lstm_dropout: 0.
            })
            loss += batch_loss
            targets.append(batch_target.ravel())
            predictions.append(batch_predictions.ravel())
            nb_batches += 1
        except tf.errors.OutOfRangeError:
            break
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    loss /= nb_batches
    accuracy = np.sum((predictions == targets) & (targets != 0)) / np.sum(targets != 0)
    logger.info('Validation loss = %6.8f, validation accuracy = %6.8f' % (loss, accuracy))
    logger.info('\n' + classification_report_with_labels(targets, predictions, vocab))


def main():
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


if __name__ == '__main__':
    main()
