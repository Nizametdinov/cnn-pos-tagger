class Vocab:
    def __init__(self, data_reader):
        self._data_reader = data_reader
        self._char2index = {}
        self._index2char = []
        self._loaded = False

    def _feed(self, char):
        if char not in self._char2index:
            index = len(self._char2index)
            self._char2index[char] = index
            self._index2char.append(char)

    def _load_initial_chars(self):
        self._feed(' ')
        self._feed('{')
        self._feed('}')

    def load_chars(self):
        uniq_chars = self._data_reader.get_uniq_chars()

        self._load_initial_chars()

        for c in sorted(uniq_chars):
            self._feed(c)

        self._loaded = True

    def to_index(self, char):
        if not self._loaded:
            raise BaseException('chars not loaded')
        return self._char2index[char]

# SENTENCES_SOURCE = 'data/sentences.xml'

# loader = DataReader(SENTENCES_SOURCE)
# loader.load()

# from vocab import Vocab

# vocab = Vocab(loader)
# vocab.load_chars()