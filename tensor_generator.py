from __future__ import print_function
import numpy as np

from data_reader import DataReader


class TensorGenerator:
    def __init__(self, data_reader):
        self._data_reader = data_reader
        self.sentences = self._data_reader.sentences
        max_sentence_length = len(self._data_reader.get_longest_sentence())
        max_word_length = len(self._data_reader.get_longest_word())
        # import pdb; pdb.set_trace()
        self.chars_tensor = np.zeros([len(self.sentences), max_sentence_length, max_word_length], dtype=np.int32)
        self.target_tensor = np.zeros([len(self.sentences), max_sentence_length], dtype=np.int32)

    def generate_tensors(self):
        for i, sentence in enumerate(self.sentences):
            for j, (word, target_class) in enumerate(sentence):
                self.target_tensor[i, j] = 13  # target_class
                for k, symbol in enumerate(word):
                    self.chars_tensor[i, j, k] = ord(symbol)

SENTENCES_SOURCE = 'data/sentences.xml'

loader = DataReader(SENTENCES_SOURCE)
loader.load()

print('sentences count: ', len(loader.sentences))
longest_sentence = loader.get_longest_sentence()
print('longest sentence: ', longest_sentence)
print('max sentence length: ', len(longest_sentence))
longest_word = loader.get_longest_word()
print('longest word: ', longest_word)
print('longest word chars: ', len(longest_word))
# uniq_chars = loader.get_uniq_chars()
# print('uniq_chars count: ', len(uniq_chars))
# print('uniq_chars: ', uniq_chars)

tg = TensorGenerator(loader)
tg.generate_tensors()
print(tg.chars_tensor.shape)
print(tg.chars_tensor[1])

