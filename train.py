import logging
import numpy as np
import os
import tensorflow as tf
import time

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_reader import OpenCorporaReader
from download_data import OPEN_CORPORA_DEST_FILE
from model import CharCnnLstm
from tensor_generator import TensorGenerator
from vocab import Vocab


tf.flags.DEFINE_string( 'train_dir',  'saved_models', 'models and summaries are saved there')
tf.flags.DEFINE_integer('epochs',     2,              'number of training epochs')
tf.flags.DEFINE_string( 'checkpoint', '',             'if provided variables will be restored from this checkpoint')
FLAGS = tf.flags.FLAGS


def classification_report_with_labels(y_true, y_pred, vocab: Vocab):
    # flatten and remove padding
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()[y_true != 0]
    y_true = y_true[y_true != 0]
    labels = vocab.indices_to_speech_part(np.unique([y_true, y_pred]))
    return classification_report(y_true, y_pred, target_names=labels, digits=3)


def init_datasets(loader, vocab, logger, batch_size, val_batch_size=100):
    max_word_length = len(loader.get_longest_word()) + 2

    train_sentences, val_sentences = train_test_split(loader.sentences, random_state=0, train_size=0.8)
    nb_train_batches = int(np.ceil(len(train_sentences) / batch_size))
    logger.info('train/validation set size: %d / %d' % (len(train_sentences), len(val_sentences)))

    data_types = tf.int32, tf.int64, tf.float32
    data_shapes = tf.TensorShape([None, max_word_length]), tf.TensorShape([None]), tf.TensorShape([None])
    padded_shapes = ([None, max_word_length], [None], [None])

    train_dataset = tf.data.Dataset.from_generator(
        TensorGenerator(train_sentences, vocab, max_word_length), data_types, data_shapes
    ).padded_batch(batch_size, padded_shapes)

    val_dataset = tf.data.Dataset.from_generator(
        TensorGenerator(val_sentences, vocab, max_word_length), data_types, data_shapes
    ).padded_batch(val_batch_size, padded_shapes)

    return train_dataset, val_dataset, max_word_length, nb_train_batches


class TrainInfo:
    __slots__ = 'start_time', 'nb_of_batches', 'epoch', 'batch'


def train_model(data_file=OPEN_CORPORA_DEST_FILE, logger=logging.getLogger()):
    loader = OpenCorporaReader(data_file)
    loader.load()
    vocab = Vocab(loader)
    vocab.load()
    train_dir = Path(FLAGS.train_dir)

    batch_size = 20

    with tf.Session() as session:
        train_dataset, val_dataset, max_word_length, nb_train_batches = init_datasets(loader, vocab, logger, batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = iterator.make_initializer(train_dataset)
        val_initializer = iterator.make_initializer(val_dataset)

        input_tensor, target_tensor, target_mask_tensor = iterator.get_next()

        model = CharCnnLstm(
            max_word_length=max_word_length,
            char_vocab_size=vocab.char_vocab_size(),
            num_output_classes=vocab.part_vocab_size(),
            input_tensor=input_tensor,
            target_tensor=target_tensor,
            target_mask_tensor=target_mask_tensor
        )
        model.init_for_training(learning_rate=0.001)
        model.init_summaries()
        if FLAGS.checkpoint:
            model.restore_model(session, FLAGS.checkpoint)
        else:
            session.run(tf.global_variables_initializer())

        train_summary_writer = tf.summary.FileWriter(str(train_dir / 'board/train'), session.graph)
        test_summary_writer = tf.summary.FileWriter(str(train_dir / 'board/test'))

        step, variable_summaries = session.run([model.global_step, model.variable_summaries])
        test_summary_writer.add_summary(variable_summaries, step)

        train_info = TrainInfo()
        train_info.nb_of_batches = nb_train_batches
        train_info.start_time = time.time()
        for epoch in range(FLAGS.epochs):
            train_info.epoch = epoch + 1
            session.run(train_initializer)

            train_info.batch = 0
            while True:
                train_info.batch += 1
                try:
                    step = train_step(session, model, train_info, train_summary_writer, logger)
                except tf.errors.OutOfRangeError:
                    break

            session.run(val_initializer)

            run_validations(session, model, vocab, target_tensor, test_summary_writer, step, logger)

            logging.info('Saving model...')
            checkpoint_name = f'epoch_{epoch:02d}.ckpt' if epoch != FLAGS.epochs - 1 else 'model.ckpt'
            model.save_model(session, str(train_dir / checkpoint_name))
        train_summary_writer.close()
        test_summary_writer.close()


def train_step(session: tf.Session,
               model: CharCnnLstm,
               train_info: TrainInfo,
               summary_writer: tf.summary.FileWriter,
               logger: logging.Logger,
               report_step: int = 20):
    if train_info.batch % report_step == 0:
        loss_value, _, gradient_norm, step, loss_acc_summary = session.run([
            model.loss,
            model.train_op,
            model.global_norm,
            model.global_step,
            model.loss_acc_summary
        ], {
            model.lstm_dropout: 0.5
        })
        summary_writer.add_summary(loss_acc_summary, step)
        log_level = logging.INFO
    else:
        loss_value, _, gradient_norm, step = session.run([
            model.loss,
            model.train_op,
            model.global_norm,
            model.global_step
        ], {
            model.lstm_dropout: 0.5
        })
        log_level = logging.DEBUG
    elapsed = time.time() - train_info.start_time
    logger.log(
        log_level,
        f'{step:6}: {train_info.epoch} [{train_info.batch:5}/{train_info.nb_of_batches:5}], '
        f'train_loss = {loss_value:6.8f} elapsed = {elapsed:.4f}s, grad.norm={gradient_norm:6.8f}'
    )

    return step


def run_validations(
        session: tf.Session,
        model: CharCnnLstm,
        vocab: Vocab,
        target_tensor: tf.Tensor,
        summary_writer: tf.summary.FileWriter,
        step: int,
        logger: logging.Logger):
    targets = []
    predictions = []
    loss = 0
    nb_batches = 0
    while True:
        try:
            batch_loss, batch_target, batch_predictions, variable_summaries = session.run([
                model.loss,
                target_tensor,
                model.predictions,
                model.variable_summaries
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
    logger.info(f'Validation loss = {loss:6.8f}, validation accuracy = {accuracy:6.8f}')
    logger.info('\n' + classification_report_with_labels(targets, predictions, vocab))

    summary = tf.Summary(value=[
        tf.Summary.Value(tag='loss', simple_value=loss),
        tf.Summary.Value(tag='accuracy', simple_value=accuracy),
    ])
    summary_writer.add_summary(summary, step)
    summary_writer.add_summary(variable_summaries, step)


def main(_argv):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.isdir('log'):
        os.mkdir('log')
    log_file = 'log/training_' + datetime.today().strftime('%Y%m%d_%H%M%S') + '.log'
    file_handler = logging.FileHandler(log_file, encoding='UTF-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    train_model(logger=logger)


if __name__ == '__main__':
    tf.app.run()
