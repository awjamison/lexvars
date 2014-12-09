# Author: Tal Linzen <linzen@nyu.edu>
# 2011-2014
# License: BSD (3-clause)

import __builtin__
import csv
import collections
import itertools
import os

import numpy as np

from nltk.corpus import wordnet as wn
from celex import Celex

if 'clx' not in globals():
    path_to_clx = os.path.expanduser("~/Dropbox/corpora/CELEX_V2/english")
    clx = Celex(path_to_clx)
    clx.load_lemmas()
    clx.load_wordforms()
    clx.map_lemmas_to_wordforms()

path_to_ds = os.path.expanduser("~/Dropbox/SufAmb/all_subjects.csv")


class LexVars(object):
    '''
    Example:

    clx = Celex(path_to_celex)
    lv = LexVars(clx)
    lv.inflectional_entropy('shoe')
    '''

    def __init__(self, path=path_to_ds):
        self.load(path)

    def load(self, path):
        self.data = list(csv.DictReader(open(path)))
        self.words = set(rec['Word'] for rec in self.data)

    def wordnet_synsets(self):
        '''
        Number of WordNet "synsets" (roughly, senses) for the given word.
        This variable collapses across different parts of speech, meanings,
        etc.
        '''
        for record in self.data:
            synsets = wn.synsets(record['Word'])
            if len(synsets) > 0:
                record['n_synsets'] = len(synsets)
            else:
                record['n_synsets'] = 0

    def synset_ratio(self):

        for record in self.data:
            n_synsets = len(wn.synsets(record['Word'], 'n'))
            v_synsets = len(wn.synsets(record['Word'], 'v'))
            if n_synsets > 0:
                record['vtn_synsets'] = v_synsets / float(n_synsets)
            else:
                record['vtn_synsets'] = 0

    def synset_entropy(self, word):
        raise NotImplementedError('Not implemented.')

    def synset_entropy_ratio(self, word):
        raise NotImplementedError('Not implemented.')

    def pos_freq(self, lemma, pos):
        '''
        Total frequency of the lemma when used as the given part of speech:

        >>> lv.pos_freq('wind', 'noun')
        3089
        '''
        lemmas = clx.lemma_lookup(lemma)
        return sum(lemma.Cob for lemma in lemmas if lemma.ClassNum == pos)

    def _smoothed_log_ratio(self, a, b):
        return np.log2((a + 1.) / (b + 1.))

    def log_noun_to_verb_ratio(self, lemma):
        for record in self.data:
            noun_freq = self.pos_freq(record['Word'], 'noun')
            verb_freq = self.pos_freq(record['Word'], 'verb')
            ratio = _smoothed_log_ratio(noun_freq, verb_freq)
            record['log_ntv_ratio'] = ratio

    def inflectional_entropy(self, smooth=1, verbose=False):
        '''
        This function collapses across all relevant lemmas, e.g. the noun 
        "build" and the verb "build", or the various "wind" verbs.

        Caution: if there are two ways to express the same inflection, the
        function will treat them as the same cell in the inflection 
        distribution (e.g. "hanged" and "hung"). Probably worth adding this
        as an option in a future version.

        This function supports the following three types of inflectional 
        entropy, but there are many more ways to carve up the various 
        inflections.

        Paradigm 1: separate_bare

        bare forms are separated into nominal and verbal, but the
        verbal bare form is not further differentiated between present 
        plural agreeing form and infinitive

        ache (singular), aches (plural), ache (verb -- infinitive, 
        present tense except third singular),
        aches (3rd singular present),
        aching (participle), ached (past tense),
        ached (participle -- passive and past_tense)

        Paradigm 2: collapsed_bare

        Same as separate_bare but collapsing across bare forms:

        ache (singular noun and all bare verbal forms --
        so all forms with no overt inflection), aches (plural),
        aches (3rd singular present), aching (participle),
        ached (past tense), ached (participles)

        Paradigm 3: no_bare

        Same as collapsed_bare, only without bare form:

        aches (plural), aches (3rd singular present),
        aching (participle), ached (past tense), ached (participles)
        '''

        for record in self.data:
            clx_lemmas = clx.lemma_lookup(record['Word'])
            # Use __builtin__ here in case sum is overshadowed by numpy
            all_wordforms = __builtin__.sum((clx.lemma_to_wordforms(clx_lemma)
                                             for clx_lemma in clx_lemmas), [])

            counter = collections.Counter()

            for wf in all_wordforms:
                infl = wf.FlectType
                freq = wf.Cob
                if (infl[0] == 'present_tense' and infl[1] != '3rd_person_verb'
                    or infl[0] == 'infinitive'):
                    counter['bare_verb'] += freq
                if infl[0] == 'singular':
                    counter['bare_noun'] += freq
                if infl[0] == 'plural':
                    counter['noun_plural'] += freq
                if infl[0] == 'past_tense':
                    counter['past_tense'] += freq
                if infl == ['positive']:
                    counter['positive'] += freq
                if infl == ['comparative']:
                    counter['comparative'] += freq
                if infl == ['superlative']:
                    counter['superlative'] += freq
                if infl == ['headword_form']:
                    counter['headword_form'] += freq
                if infl == ['present_tense', '3rd_person_verb', 'singular']:
                    counter['third_sg'] += freq
                if infl == ['participle', 'present_tense']:
                    counter['part_ing'] += freq
                if infl == ['participle', 'past_tense']:
                    counter['part_ed'] += freq

            common = ['noun_plural', 'third_sg', 'part_ing', 'part_ed',
                      'past_tense', 'comparative', 'superlative']
            bare = ['bare_noun', 'bare_verb', 'positive', 'headword_form']
            common_freqs = [counter[i] for i in common if i in counter]
            bare_freqs = [counter[i] for i in bare if i in counter]

            if verbose:
                print counter

            record['infl_ent_separate_bare'] = self.entropy(bare_freqs + common_freqs, smooth)
            record['infl_ent_collapsed_bare'] = self.entropy([sum(bare_freqs)] + common_freqs, smooth)
            record['infl_ent_no_bare'] = self.entropy(common_freqs, smooth)

    def derivational_family_size(self):

        by_morpheme = {}

        for lemma in clx._lemmas:
            if len(lemma['Parses']) < 1:
                continue
            decomposition = lemma['Parses'][0]['Imm']
            morphemes = decomposition.split('+')
            for morpheme in morphemes:
                if morpheme in self.words:
                    by_morpheme.setdefault(morpheme, set()).add(
                            lemma['Head'])

        for record in self.data:
            decompositions = by_morpheme.get(record['Word'])
            if decompositions is not None:
                record['derivational_family_size'] = len(decompositions)
            else:
                record['derivational_family_size'] = None

    def derivational_family_entropy(self):

        by_morpheme = {}

        for lemma in clx._lemmas:
            if len(lemma['Parses']) < 1:
                continue
            decomposition = lemma['Parses'][0]['Imm']
            morphemes = decomposition.split('+')
            if morphemes[0] in self.words:
                by_morpheme.setdefault(morphemes[0], list()).append(lemma)

        for record in self.data:
            derived = by_morpheme.get(record['Word'])
            freqs = [x['Cob'] for x in derived]
            record['derivational_entropy'] = self.entropy(freqs)

    def entropy(self, freq_vec, smoothing_constant=1):
        '''
        This flat smoothing is an OK default but probably not the best idea:
        might be better to back off to the average distribution for the 
        relevant paradigm (e.g. if singular forms are generally twice as likely 
        as plural ones, it's better to use [2, 1] as the "prior" instead of 
        [1, 1]).
        '''
        vec = np.asarray(freq_vec, float) + smoothing_constant
        if sum(vec) == 0:
            return -1
        probs = vec / sum(vec)
        # Make sure we're not taking the log of 0 (by convention if p(x) = 0
        # then p(x) * log(p(x)) = 0 in the definition of entropy)
        probs[probs == 0] = 1
        return -np.sum(probs * np.log2(probs))