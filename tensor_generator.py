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
                if len(word) + 2 > self._max_word_length:
                    word = word[:self._max_word_length - 2]
                word = WORD_START + word + WORD_END
                for k, char in enumerate(word):
                    char_tensor[j, k] = self._vocab.char_to_index(char)
            yield char_tensor, targets, mask
