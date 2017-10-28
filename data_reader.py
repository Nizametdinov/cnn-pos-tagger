import xml.etree.ElementTree as etree

class DataReader:
    SKIPPED_SPEACH_PARTS = set(['PNCT', 'NUMB', 'SYMB'])
    STOP_CHARS_CODES = set(range(ord('a'), ord('z'))).union(set(range(ord('A'), ord('Z'))))

    def __init__(self, xml_filename):
        self._xml_filename = xml_filename
        self.sentences = []
        self._uniq_chars = None
        self._uniq_speach_parts = None
        self._loaded = False

    def load(self):
        tokens = self._get_tokens(self._xml_filename)
        self.sentences = self._get_sentences(tokens)
        self._loaded = True

    def _get_tokens(self, xml_filename):
        tree = etree.parse(xml_filename)
        tokens = tree.findall('.//tokens')
        return tokens

    def _word_has_stop_chars(self, word):
        for c in word:
            if ord(c) in self.STOP_CHARS_CODES:
                return True
        return False

    def _get_sentence(self, tokens_entry):
        sentence = []
        for token in tokens_entry:
            speech_part = token.find('.//g').attrib['v']
            if speech_part in self.SKIPPED_SPEACH_PARTS:
                continue
            word = token.attrib['text']
            if self._word_has_stop_chars(word):
                raise ValueError('sentence has invalid chars')
            sentence.append([word, speech_part])
        return sentence

    def _get_sentences(self, tokens):
        sentences = []
        for tokens_entry in tokens:
            try:
                sentences.append(self._get_sentence(tokens_entry))
            except ValueError as _error:
                pass
        return sentences

    def get_uniq_chars(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        if self._uniq_chars:
            return self._uniq_chars.copy()
        else:
            uniq_chars = set()
            for sentence in self.sentences:
                for word, _speech_part in sentence:
                    for c in word:
                        uniq_chars.add(c)
            self._uniq_chars = uniq_chars
        return uniq_chars.copy()

    def get_longest_sentence(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        return max(self.sentences, key=lambda sentence: len(sentence))

    def get_longest_word(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        max_length = 0
        longest_word = ''

        for sentence in self.sentences:
            for word, _speach_part in sentence:
                word_len = len(word)
                if word_len > max_length:
                    max_length = word_len
                    longest_word = word
        return longest_word

    def get_uniq_speach_parts(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        if self._uniq_speach_parts:
            return self._uniq_speach_parts.copy()
        else:
            uniq_speach_parts = set()
            for sentence in self.sentences:
                for _word, speech_part in sentence:
                    uniq_speach_parts.add(speech_part)
            self._uniq_speach_parts = uniq_speach_parts
        return uniq_speach_parts.copy()

# SENTENCES_SOURCE = 'data/sentences.xml'

# loader = DataReader(SENTENCES_SOURCE)
# loader.load()

# print(loader.get_uniq_speach_parts())

# print('sentences count: ', len(loader.sentences))
# longest_sentence = loader.get_longest_sentence()
# print('longest sentence: ', longest_sentence)
# print('max sentence length: ', len(longest_sentence))
# longest_word = loader.get_longest_word()
# print('longest word: ', longest_word)
# print('longest word chars: ', len(longest_word))
# uniq_chars = loader.get_uniq_chars()
# print('uniq_chars count: ', len(uniq_chars))
# print('uniq_chars: ', uniq_chars)
