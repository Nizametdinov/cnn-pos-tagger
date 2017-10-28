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

    def _feed_speach_part(self, speach_part):
        if speach_part not in self._part2index:
            index = len(self._part2index)
            self._part2index[speach_part] = index
            self._index2part.append(speach_part)

    def _load_initial_chars(self):
        self._feed_char(' ')
        self._feed_char('{')
        self._feed_char('}')

    def _load_chars(self):
        uniq_chars = self._data_reader.get_uniq_chars()
        self._load_initial_chars()
        for c in sorted(uniq_chars):
            self._feed_char(c)

    def _load_speach_parts(self):
        uniq_speach_parts = self._data_reader.get_uniq_speach_parts()
        for part in sorted(uniq_speach_parts):
            self._feed_speach_part(part)

    def load(self):
        self._load_chars()
        self._load_speach_parts()
        self._loaded = True

    def char_to_index(self, char):
        if not self._loaded:
            raise BaseException('chars not loaded')
        return self._char2index[char]

    def part_to_index(self, speach_part):
        if not self._loaded:
            raise BaseException('chars not loaded')
        return self._part2index[speach_part]
