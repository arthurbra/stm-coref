import re
from functools import lru_cache

import nltk


@lru_cache(maxsize=None)
def _find_acronyms(text, entities):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    phrase_to_acronym = {}
    for (start, end) in entities:
        content = text[start:end]
        contains_phrase_and_its_acronym = len(content.split()) > 1 and content.count('(') == content.count(
            ')') == 1 and \
                                          '(' not in content.split()[0] and ')' not in content.split()[0] and \
                                          content.index('(') + 1 < content.index(')')
        if contains_phrase_and_its_acronym:
            acronym_candidate = content.split('(')[1].split(')')[0]
            acronym_letters = [letter for letter in acronym_candidate.lower()]

            phrase_candidate = content.split('(')[0].lower()
            # removes e.g. 'a' and 'the' at the beginning of an entity
            if phrase_candidate.split()[0] in stop_words:
                phrase_candidate = ' '.join(phrase_candidate.split()[1:])
            phrase_first_chars = [s[0].lower() for s in phrase_candidate.split()]
            if acronym_letters[0] in phrase_first_chars:
                phrase_index_of_first_char = phrase_first_chars.index(acronym_letters[0])
                phrase_candidate = ' '.join(phrase_candidate.split()[phrase_index_of_first_char:])
                # need to update it, since phrase_candidate has changed
                phrase_first_chars = [s[0].lower() for s in phrase_candidate.split()]

            # determine if the candidate is a real acronym
            num_acr_letters_found_in_first_chars = 0
            for ac in acronym_letters:
                if ac in phrase_first_chars:
                    num_acr_letters_found_in_first_chars += 1
                    phrase_first_chars.pop(phrase_first_chars.index(ac))
            phrase_contains_similar_first_chars = num_acr_letters_found_in_first_chars / len(acronym_letters) >= 0.5
            acr_is_one_word = ' ' not in acronym_candidate and len(acronym_candidate) > 1
            acr_contains_one_upper_case_char = len([c for c in acronym_candidate if 'A' <= c <= 'Z']) > 0
            is_acronym = acr_is_one_word and phrase_contains_similar_first_chars and acr_contains_one_upper_case_char

            if is_acronym:
                phrase_to_acronym[phrase_candidate] = acronym_candidate

    return phrase_to_acronym


@lru_cache(maxsize=None)
def _get_word_indices_to_pos_tag(text):
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
                    wi, curr_word, ci = wi + 1, tokenized_text.pop(0), ci + 1  # handles BERT tokenizer
                    continue
                elif curr_word == '`' == tokenized_text[0] or curr_word == "\'" == tokenized_text[0]:
                    tokenized_text.pop(0)
                    wi, curr_word, ci = wi + 2, tokenized_text.pop(
                        0), ci + 2  # bert tokenizer / wordpieces:    "word" -> ` ` word \  \
                    continue
                elif curr_word in ['[SEP]', '[CLS]']:
                    if len(tokenized_text) > 0:
                        curr_word = tokenized_text.pop(0)
                    wi = wi + 1  # tokenized text of longer documents include [SEP][CLS] inside and at the beginning
                    continue

                start, end = ci, ci + len(curr_word)
                if self.text[start:end] == curr_word or (curr_word == '"' and self.text[start:end + 1] == '\'\''):
                    wi_to_ci_tuple[wi] = (start, end)
                    if len(tokenized_text) == 0:
                        break  # no need to check boundaries, this exits the loop before reaching a non existing index
                    else:
                        wi, curr_word, ci = wi + 1, tokenized_text.pop(0), end

                    if curr_word == '"' and self.text[start:end + 1] == '\'\'':
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
            is_only_one_tuple = not isinstance(word_index_tuples[0], tuple) and not isinstance(word_index_tuples[0],
                                                                                               list)
            if is_only_one_tuple:
                return self.__wi_tuple_to_to_ci_tuple(word_index_tuples[0], word_index_tuples[1])
            else:
                return [self.__wi_tuple_to_to_ci_tuple(wi_start, wi_end) for wi_start, wi_end in word_index_tuples]

    # actual code
    tokenized_text = nltk.word_tokenize(text)
    converter = IndexConverter(text, tokenized_text)
    word_indices_to_pos_tag = {}
    for _word_index, (_, _pos_tag) in enumerate(nltk.pos_tag(tokenized_text)):
        _start, _end = converter.to_char_index((_word_index, _word_index))
        word_indices_to_pos_tag[(_start, _end)] = _pos_tag
    return word_indices_to_pos_tag


def normalize_phrase(entity_start, entity_end, doc_text, doc_entities):
    def is_verb(_pos_tag):
        return _pos_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    def is_noun(_pos_tag):
        return _pos_tag in ['NN', 'NNS', 'NNP', 'NNPS']

    def is_plural(_word):
        return _word[-1] == 's' or _word[-2:] == 'es'  # seems to be more effective than using pos tags

    lemmatizer = nltk.WordNetLemmatizer()
    entity_phrase = doc_text[entity_start:entity_end]

    # finds acronyms and replaces them with their full phrases
    # if a phrase contains an acronym and its full phrase -> remove the acronym since it is redundant
    contained_acronym = False
    for phrase, acronym in _find_acronyms(doc_text, tuple(doc_entities)).items():
        if acronym in re.split('[ -]', entity_phrase):
            entity_phrase = entity_phrase.replace(acronym, phrase)
            contained_acronym = True
        elif f'({acronym})' in re.split('[ -]', entity_phrase):
            entity_phrase = entity_phrase.replace(f' ({acronym})', '').replace(f'({acronym})', '')
            contained_acronym = True

    # only makes first word of an entity lowercase if it isn't an acronym
    if len(entity_phrase) >= 2 and entity_phrase[0].isupper() and entity_phrase[1].islower():
        entity_phrase = entity_phrase[0].lower() + entity_phrase[1:]

    # pos-tags are less accurate when generated for a phrase rather than for an entire text
    # that's why WortUtils stores pos-tags generated for an entire text
    normalized_words = []
    if contained_acronym:
        # since replacing an acronym with its phrase or removing an acronym changes the number of words
        # in the phrase and DocumentInfo only stores the pos-tags for the original text
        # need to generate pos-tags for the words in the entity
        for word_index, (word, pos_tag) in enumerate(nltk.pos_tag(entity_phrase.split())):
            is_last_word = word_index == len(entity_phrase.split()) - 1
            if len(entity_phrase.split()) == 1 and (is_verb(pos_tag) or is_noun(pos_tag)):
                normalized_words.append(
                    lemmatizer.lemmatize(word, pos='v' if is_verb(pos_tag) else 'n'))
            elif is_last_word and is_plural(word) and is_noun(pos_tag):
                normalized_words.append(lemmatizer.lemmatize(word, pos='n'))
            else:
                normalized_words.append(word)
    else:
        # find the correct pos-tags stored in WordUtils
        entity_word_indices, pos_tags = [], []
        found_start, found_end = False, False
        for (start, end), pos_tag in _get_word_indices_to_pos_tag(doc_text).items():
            if start == entity_start:
                found_start = True
            if found_start:
                entity_word_indices.append((start, end))
                pos_tags.append(pos_tag)
            found_end = end == entity_end
            if found_end:
                break

        for word_index, (word_indices, word, pos_tag) in enumerate(zip(entity_word_indices, entity_phrase.split(), pos_tags)):
            is_last_word = word_index == len(entity_phrase.split()) - 1
            pos_tag = _get_word_indices_to_pos_tag(doc_text)[word_indices]
            if len(entity_phrase.split()) == 1 and (is_verb(pos_tag) or is_noun(pos_tag)):
                normalized_words.append(lemmatizer.lemmatize(word, pos='v' if is_verb(pos_tag) else 'n'))
            elif is_last_word and is_plural(word) and is_noun(pos_tag):
                normalized_words.append(lemmatizer.lemmatize(word, pos='n'))
            else:
                normalized_words.append(word)

    # removes determiners such as 'a', 'the', 'these'
    if len(normalized_words) > 1 and normalized_words[0].lower() in set(nltk.corpus.stopwords.words('english')):
        normalized_words = normalized_words[1:]

    return ' '.join(normalized_words)