import math

class Vocab:
    EMPTY_CHAR = ' '

    def __init__(self, data_reader):
        self._data_reader = data_reader

        self._char2index = {}
        self._index2char = []

        self._part2index = {}
        self._index2part = []

        self._char_freq_threshold = 0
        self._skipped_chars = []

        self._loaded = False

    def _feed_char(self, char):
        if char in self._skipped_chars:
            return

        if char not in self._char2index:
            index = len(self._char2index)
            self._char2index[char] = index
            self._index2char.append(char)

    def _feed_speech_part(self, speech_part):
        if speech_part not in self._part2index:
            index = len(self._part2index)
            self._part2index[speech_part] = index
            self._index2part.append(speech_part)

    def _load_initial_chars(self):
        self._feed_char(self.EMPTY_CHAR)
        self._feed_char('{')
        self._feed_char('}')

    def _load_initial_parts(self):
        self._feed_speech_part('UNKNOWN')

    def _load_chars(self):
        uniq_chars = self._data_reader.get_uniq_chars()
        self._load_initial_chars()
        for c in sorted(uniq_chars):
            self._feed_char(c)

    def _load_speech_parts(self):
        uniq_speech_parts = self._data_reader.get_uniq_speech_parts()
        self._load_initial_parts()
        for part in sorted(uniq_speech_parts):
            self._feed_speech_part(part)

    def _calculate_char_freq_threshold(self):
         _char, max_freq = self._data_reader.get_chars_freq()[-1]
         self._char_freq_threshold = math.ceil(max_freq * 0.01) # 1% of max freshold

    def _find_low_freq_chars(self):
        self._calculate_char_freq_threshold()

        for char, freq in self._data_reader.get_chars_freq():
            if freq <= self._char_freq_threshold:
                self._skipped_chars.append(char)
            else:
                break

    def load(self):
        self._find_low_freq_chars()

        self._load_chars()
        self._load_speech_parts()

        self._loaded = True

    def char_to_index(self, char):
        if not self._loaded:
            raise BaseException('chars not loaded')
        return self._char2index.get(char) or self._char2index[self.EMPTY_CHAR]

    def part_to_index(self, speech_part):
        if not self._loaded:
            raise BaseException('chars not loaded')
        return self._part2index[speech_part]

    def char_vocab_size(self):
        return len(self._index2char)

    def part_vocab_size(self):
        return len(self._index2part)

if __name__ == '__main__':
    # debug
    from data_reader import DataReader
    SENTENCES_SOURCE = 'data/sentences.xml'

    data_reader = DataReader(SENTENCES_SOURCE)
    data_reader.load()

    vocab = Vocab(data_reader)
    vocab.load()

    print('skipped chars threshold', vocab._char_freq_threshold)
    print('skipped chars', vocab._skipped_chars)
    print('vocab', vocab._char2index)
