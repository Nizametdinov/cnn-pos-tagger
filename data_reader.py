import xml.etree.ElementTree as etree
import operator

class DataReader:
    SKIPPED_SPEECH_PARTS = set(['PNCT', 'NUMB', 'SYMB', 'LATN', 'ROMN'])
    STOP_CHARS_CODES = range(ord('a'), ord('z'))

    def __init__(self, xml_filename):
        self._xml_filename = xml_filename
        self.sentences = []
        self._uniq_chars = {}
        self._uniq_speech_parts = {}
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
            current_char_count = self._uniq_chars.get(c) or 0
            self._uniq_chars[c] = current_char_count + 1

    def _add_uniq_speech_parts(self, speech_part):
        current_parts_count = self._uniq_speech_parts.get(speech_part) or 0
        self._uniq_speech_parts[speech_part] = current_parts_count + 1

    def _get_sentence(self, tokens_entry):
        sentence = []
        for token in tokens_entry:
            speech_part = token.find('.//g').attrib['v']
            if speech_part in self.SKIPPED_SPEECH_PARTS:
                continue
            word = token.attrib['text'].lower()
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

        return self._uniq_chars.keys()

    def get_uniq_speech_parts(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        return self._uniq_speech_parts.keys()

    def get_longest_sentence(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        return max(self.sentences, key=lambda sentence: len(sentence))

    def get_chars_freq(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        return sorted(self._uniq_chars.items(), key = operator.itemgetter(1))

    def get_speech_parts_freq(self):
        if not self._loaded:
            raise BaseException('data not loaded')

        return sorted(self._uniq_speech_parts.items(), key = operator.itemgetter(1))

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

if __name__ == '__main__':
    SENTENCES_SOURCE = 'data/sentences.xml'

    loader = DataReader(SENTENCES_SOURCE)
    loader.load()

    print('unique chars', loader.get_chars_freq())
    print('unique speech_parts', loader.get_speech_parts_freq())

    print('sentences count: ', len(loader.sentences))
    longest_sentence = loader.get_longest_sentence()
    print('longest sentence: ', longest_sentence)
    print('max sentence length: ', len(longest_sentence))
    longest_word = loader.get_longest_word()
    print('longest word: ', longest_word)
    print('longest word chars: ', len(longest_word))

