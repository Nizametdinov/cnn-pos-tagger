import numpy as np
from typing import List, Tuple
from vocab import Vocab


WORD_START = '{'
WORD_END = '}'


class TensorGenerator:
    def __init__(self, sentences: List[List[Tuple[str, str]]], vocab: Vocab, max_word_length: int):
        self._sentences = sentences
        self._vocab = vocab
        self._max_word_length = max_word_length

    def __call__(self):
        for sentence in self._sentences:
            targets = []
            mask = [1] * len(sentence)
            char_tensor = np.zeros((len(sentence), self._max_word_length), dtype=np.int32)
            for j, (word, target_class) in enumerate(sentence):
                targets.append(self._vocab.part_to_index(target_class))
                word_to_char_indices(word, self._vocab, self._max_word_length, char_tensor[j])
            yield char_tensor, targets, mask


def sentences_to_char_tensor(sentences: List[List[str]], vocab: Vocab, max_word_length: int):
    max_sentence_length = max(len(sentence) for sentence in sentences)
    char_tensor = np.zeros((len(sentences), max_sentence_length, max_word_length), dtype=np.int32)
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            word_to_char_indices(word, vocab, max_word_length, char_tensor[i, j])
    return char_tensor


def word_to_char_indices(word: str, vocab: Vocab, max_word_length: int, out: np.ndarray):
    if len(word) + 2 > max_word_length:
        word = word[:max_word_length - 2]
    word = WORD_START + word + WORD_END
    for k, char in enumerate(word):
        out[k] = vocab.char_to_index(char)
