import tensorflow as tf

from data_reader import OpenCorporaReader
from download_data import OPEN_CORPORA_DEST_FILE
from model import CharCnnLstm
from tensor_generator import sentences_to_char_tensor
from vocab import Vocab


tf.flags.DEFINE_string('checkpoint', 'saved_models/model.ckpt', 'model checkpoint path')
FLAGS = tf.flags.FLAGS


def main(_argv):
    data_file = OPEN_CORPORA_DEST_FILE
    loader = OpenCorporaReader(data_file)
    loader.load()
    vocab = Vocab(loader)
    vocab.load()
    max_word_length = len(loader.get_longest_word()) + 2
    with tf.Session() as session:
        model = CharCnnLstm(
            max_word_length=max_word_length,
            char_vocab_size=vocab.char_vocab_size(),
            num_output_classes=vocab.part_vocab_size()
        )
        model.init_for_evaluation()
        model.restore_model(session, FLAGS.checkpoint)

        print('graph init finished')

        sentences = [
            ['проверка', 'связи', 'прошла', 'успешно'],
            ['глокая', 'куздра', 'штеко', 'будланула', 'бокра', 'и', 'кудрячит', 'бокрёнка'],
            ['эти', 'типы', 'стали', 'есть', 'на', 'складе'],
        ]
        input_tensors = sentences_to_char_tensor(sentences, vocab, max_word_length)

        predicted = session.run([model.predictions], {model.input: input_tensors, model.lstm_dropout: 0.0})

        for sentence, sentence_prediction in zip(sentences, predicted[0]):
            for word, word_prediction in zip(sentence, sentence_prediction):
                print('word: ', word, ' predicted part of speech: ', vocab.index_to_speech_part_human(word_prediction))
            print()


if __name__ == '__main__':
    tf.app.run()
