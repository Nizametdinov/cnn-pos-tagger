class Vocab:
    def __init__(self, data_reader):
        self._data_reader = data_reader

        self._char2index = {}
        self._index2char = []

        self._part2index = {}
        self._index2part = []

        self._loaded = False

    def _feed_char(self, char):
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
        self._feed_char(' ')
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

    def load(self):
        self._load_chars()
        self._load_speech_parts()
        self._loaded = True

    def char_to_index(self, char):
        if not self._loaded:
            raise BaseException('chars not loaded')
        return self._char2index[char]

    def part_to_index(self, speech_part):
        if not self._loaded:
            raise BaseException('chars not loaded')
        return self._part2index[speech_part]

    def char_vocab_size(self):
        return len(self._index2char)

    def part_vocab_size(self):
        return len(self._index2part)
