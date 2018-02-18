from tensor_generator import TensorGenerator
import numpy as np
import pytest


class FakeVocab:
    char2index = {c: i for i, c in enumerate('{}abcdefghijklmnopqrstuvwxyz')}
    part2index = {'PAD': 0, 'ADJF': 1, 'DET': 2, 'NOUN': 3, 'NPRO': 4, 'VERB': 5}

    def part_to_index(self, speech_part):
        return self.part2index[speech_part]

    def char_to_index(self, char):
        return self.char2index[char]


def test_tensor_generator():
    sentences = [[
        ('he', 'NPRO'),
        ('was', 'VERB'),
        ('contemplating', 'VERB'),
        ('the', 'DET'),
    ], [
        ('next', 'ADJF'),
        ('move', 'NOUN'),
    ]]
    sut = TensorGenerator(sentences, FakeVocab(), 5)
    gen = sut()

    expected_char_tensor_1 = np.array([
        [0,  9,  6,  1,  0],
        [0, 24,  2, 20,  1],
        [0,  4, 16, 15,  1],
        [0, 21,  9,  6,  1]
    ])
    char_tensor_1, target_list_1, target_mask_1 = next(gen)
    assert np.all(char_tensor_1 == expected_char_tensor_1)
    assert target_list_1 == [4, 5, 5, 2]
    assert target_mask_1 == [1, 1, 1, 1]

    expected_char_tensor_2 = np.array([
        [0, 15,  6, 25, 1],
        [0, 14, 16, 23, 1],
    ])
    char_tensor_2, target_list_2, target_mask_2 = next(gen)
    assert np.all(char_tensor_2 == expected_char_tensor_2)
    assert target_list_2 == [1, 3]
    assert target_mask_2 == [1, 1]

    with pytest.raises(StopIteration):
        next(gen)
