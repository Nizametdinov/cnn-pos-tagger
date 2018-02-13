import tensorflow as tf

from data_reader import DataReader
from download_data import OPEN_CORPORA_DEST_FILE
from model import CharCnnLstm
from tensor_generator import TensorGenerator
from train import MODEL_SAVE_PATH
from vocab import Vocab


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
            ['эти', 'типы', 'стали', 'есть', 'в', 'на складе'],
        ]
        input_tensors = tensor_generator.tensor_from_sentences(sentences)

        print(input_tensors)
        predicted = session.run([model.predictions], {model.input: input_tensors, model.lstm_dropout: 0.0})

        for sentence, sentence_prediction in zip(sentences, predicted[0]):
            for word, word_prediction in zip(sentence, sentence_prediction):
                print('word: ', word, ' predicted part of speech: ', vocab.index_to_speech_part_human(word_prediction))
            print()


if __name__ == '__main__':
    evaluate_model()
