import xml.etree.ElementTree as etree

class DataReader:
    SKIPPED_SPEACH_PARTS = set(['PNCT', 'NUMB', 'SYMB'])
    STOP_CHARS_CODES = set(range(ord('a'), ord('z'))).union(set(range(ord('A'), ord('Z'))))

    def __init__(self, xml_filename):
        self._xml_filename = xml_filename
        self.sentences = []
        self._uniq_chars = set()
        self._uniq_speech_parts = set()
        self._loaded = False

    def _get_tokens(self, xml_filename):
        tree = etree.parse(xml_filename)
        tokens = tree.findall('.//tokens')
        return tokens

    def _word_has_stop_chars(self, word):
        for c in word:
            if ord(c) in self.STOP_CHARS_CODES:
                return True
        return False

    def _add_uniq_chars(self, word):
        for c in word:
            self._uniq_chars.add(c)

    def _add_uniq_speech_parts(self, speech_part):
        self._uniq_speech_parts.add(speech_part)

    def _get_sentence(self, tokens_entry):
        sentence = []
        for token in tokens_entry:
            speech_part = token.find('.//g').attrib['v']
            if speech_part in self.SKIPPED_SPEACH_PARTS:
                continue
            word = token.attrib['text']
            if self._word_has_stop_chars(word):
                raise ValueError('sentence has invalid chars')

            self._add_uniq_chars(word)
            self._add_uniq_speech_parts(speech_part)

            sentence.append((word, speech_part))
        return sentence

    def _get_sentences(self, tokens):
        sentences = []
        for tokens_entry in tokens:
            try:
                sentences.append(self._get_sentence(tokens_entry))
            except ValueError as _error:
                pass
        return sentences

    def load(self):
        tokens = self._get_tokens(self._xml_filename)
        self.sentences = self._get_sentences(tokens)
        self._loaded = True

    def get_uniq_chars(self):
        if not self._loaded:
            raise BaseException('data not loaded')
        return self._uniq_chars.copy()

    def get_uniq_speech_parts(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        return self._uniq_speech_parts.copy()

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
            for word, _speech_part in sentence:
                word_len = len(word)
                if word_len > max_length:
                    max_length = word_len
                    longest_word = word
        return longest_word
