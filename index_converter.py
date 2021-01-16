class IndexConverter:
    def __init__(self, text, tokenized_text):
        self.text = text
        self.tokenized_text = tokenized_text.copy()
        self.wi_to_ci_tuple = self.__calc_wi_to_ci_tuple()
        self.ci_tuple_to_wi = {v: k for k, v in self.wi_to_ci_tuple.items()}

    def __calc_wi_to_ci_tuple(self):
        # clean up the tokenized text, by handling quotation marks
        # https://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing
        tokenized_text = [token.replace('``', '"').replace("''", '"') for token in self.tokenized_text]
        wi_to_ci_tuple = {}
        curr_word = tokenized_text.pop(0)
        wi = ci = 0

        while ci < len(self.text):
            if curr_word == '[UNK]':
                wi, curr_word, ci = wi + 1, tokenized_text.pop(0), ci + 1 # handles BERT tokenizer
                continue
            elif curr_word == '`' == tokenized_text[0] or curr_word == "\'" == tokenized_text[0]:
                tokenized_text.pop(0)
                wi, curr_word, ci = wi + 2, tokenized_text.pop(0), ci + 2 # bert tokenizer / wordpieces:    "word" -> ` ` word \  \
                continue
            elif curr_word in ['[SEP]', '[CLS]']:
                if len(tokenized_text) > 0:
                    curr_word = tokenized_text.pop(0)
                wi = wi + 1   # tokenized text of longer documents include [SEP][CLS] inside and at the beginning
                continue

            start, end = ci, ci + len(curr_word)
            if self.text[start:end] == curr_word or (curr_word == '"' and self.text[start:end+1] == '\'\''):
                wi_to_ci_tuple[wi] = (start, end)
                if len(tokenized_text) == 0:
                    break  # no need to check boundaries, this exits the loop before reaching a non existing index
                else:
                    wi, curr_word, ci = wi + 1, tokenized_text.pop(0), end

                if curr_word == '"' and self.text[start:end+1] == '\'\'':
                    ci += 1

            else:
                ci += 1

        return wi_to_ci_tuple

    def __wi_tuple_to_to_ci_tuple(self, wi_start, wi_end):
        ci_start = ci_end = -1
        for wi, (curr_ci_start, curr_ci_end) in self.wi_to_ci_tuple.items():
            if wi == wi_start:
                ci_start = curr_ci_start
            if wi == wi_end:
                ci_end = curr_ci_end
            if ci_start != -1 and ci_end != -1:
                return ci_start, ci_end
        raise Exception(
            f'couldn\'t find wi_start, wi_end: ({wi_start, wi_end}) text: {self.text[ci_start:ci_end]} in tokenized_text: {self.tokenized_text}')

    def __ci_tuple_to_wi_tuple(self, ci_start, ci_end):
        wi_start = wi_end = -1
        for (curr_ci_start, curr_ci_end), wi in self.ci_tuple_to_wi.items():
            if curr_ci_start == ci_start:
                wi_start = wi
            if curr_ci_end == ci_end:
                wi_end = wi
            if wi_start != -1 and wi_end != -1:
                return wi_start, wi_end
        raise Exception(
            f'couldn\'t find ci_start, ci_end: ({ci_start, ci_end}) text: {self.text[ci_start:ci_end]} in tokenized_text: {self.tokenized_text}')

    def __debug(self):
        for wi, (ci_start, ci_end) in self.wi_to_ci_tuple.items():
            print(f'wi: {wi} w: {self.tokenized_text[wi]} c: {self.text[ci_start:ci_end]}')

    def to_word_index(self, char_index_tuples: tuple):
        """ Takes one tuple or a list of tuples, with each tuple consisting of a start- and an end-index, which refer """
        is_only_one_tuple = not isinstance(char_index_tuples[0], tuple)
        if is_only_one_tuple:
            return self.__ci_tuple_to_wi_tuple(char_index_tuples[0], char_index_tuples[1])
        else:
            return [self.__ci_tuple_to_wi_tuple(ci_start, ci_end) for ci_start, ci_end in char_index_tuples]

    def to_char_index(self, word_index_tuples: tuple):
        is_only_one_tuple = not isinstance(word_index_tuples[0], tuple) and not isinstance(word_index_tuples[0], list)
        if is_only_one_tuple:
            return self.__wi_tuple_to_to_ci_tuple(word_index_tuples[0], word_index_tuples[1])
        else:
            return [self.__wi_tuple_to_to_ci_tuple(wi_start, wi_end) for wi_start, wi_end in word_index_tuples]