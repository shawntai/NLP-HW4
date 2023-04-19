#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np

import string

# import tensorflow

import gensim
import transformers

from typing import List

from collections import defaultdict


def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.
    """
    s = "".join(
        " " if x in string.punctuation else x
        for x in s.lower()
        if x not in stopwords.words("english")
    )
    return s.split()


def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    # lemma = "break"
    # pos = "n"
    results = []
    lemmas = wn.lemmas(lemma, pos)
    # for l in lemmas:
    # print(l)
    # print()
    synsets = [lemma.synset() for lemma in lemmas]
    for synset in synsets:
        # print()
        # print(synset)
        for l in synset.lemmas():
            # print(l)
            if l.name() != lemma and l.name() not in results:
                results.append(l.name())
    return results


def smurf_predictor(context: Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return "smurf"


def wn_frequency_predictor(context: Context) -> str:
    lemma = context.lemma
    pos = context.pos
    occurrences = defaultdict(int)
    lemmas = wn.lemmas(lemma, pos)
    # for l in lemmas:
    #     print(l)
    synsets = [lemma.synset() for lemma in lemmas]
    for synset in synsets:
        # print()
        # print(synset)
        for l in synset.lemmas():
            # print(l)
            if l.name() != lemma:
                occurrences[l.name()] += l.count()
    return max(occurrences, key=occurrences.get)


def wn_simple_lesk_predictor(context: Context) -> str:
    lemma = context.lemma
    # print("lemma", lemma)
    pos = context.pos
    lemmas = wn.lemmas(lemma, pos)
    best_score, best_word = -1, None
    # synsets = [lemma.synset() for lemma in lemmas]
    # reference: https://edstem.org/us/courses/35237/discussion/2916036
    for l in lemmas:
        s = l.synset()
        context_tokenized = tokenize(
            " ".join(context.left_context + context.right_context)
        )
        def_exp_tokenized = tokenize(
            s.definition()
            + " "
            + " ".join(s.examples())
            + " "
            + " ".join([x.definition() for x in s.hypernyms()])
            + " "
            + " ".join([" ".join(x.examples()) for x in s.hypernyms()])
        )
        overlap = set(context_tokenized).intersection(def_exp_tokenized)
        target_freq = l.count()
        for u in s.lemmas():
            score = 10000 * len(overlap) + 100 * target_freq + u.count()
            if score > best_score and u.name().lower() != lemma.lower():
                best_score = score
                best_word = u.name()
    return best_word


class Word2VecSubst(object):
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            filename, binary=True
        )

    def predict_nearest(self, context: Context) -> str:
        synonyms = get_candidates(context.lemma, context.pos)
        return max(
            synonyms,
            key=lambda x: self.model.similarity(x, context.lemma)
            if x in self.model.key_to_index
            else -1,
        )


class BertPredictor(object):
    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained(
            "distilbert-base-uncased"
        )

    def predict(self, context: Context) -> str:
        input_toks = self.tokenizer.encode(
            " ".join(context.left_context)
            + " [MASK] "
            + " ".join(context.right_context)
        )
        input_mat = np.array(input_toks).reshape(1, -1)
        predictions = self.model.predict(input_mat, verbose=False)[0]
        mask_index = input_toks.index(103)
        words_sorted = np.argsort(predictions[0, input_toks.index(103)])[::-1]
        candidates = get_candidates(context.lemma, context.pos)
        for word in self.tokenizer.convert_ids_to_tokens(words_sorted):
            if word in candidates:
                return word
        return None


if __name__ == "__main__":
    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = "GoogleNews-vectors-negative300.bin.gz"
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml("lexsub_trial.xml"):
        # for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest(context)
        prediction = BertPredictor().predict(context)
        print(
            "{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction)
        )

    # print(get_candidates("slow", "a"))
