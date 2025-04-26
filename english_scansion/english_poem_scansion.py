"""
This module provides tools for analyzing English poetry, including:
- Stress mark placement
- Meter detection
- Rhyme scheme analysis
- Technical quality assessment

It is part of the **Poetry Scansion Tool** project.
Repository: https://github.com/Koziev/RussianPoetryScansionTool
"""

import re
import json
import collections
import os
import itertools
import pickle

import nltk
import eng_syl
import pronouncing
import pyphen
from g2p_en import G2p

import string
import numpy as np
from nltk.corpus import cmudict
from typing import List, Set, Dict, Tuple, Optional

from tokenization_utils import tokenize_slowly
from whitespace_normalization import normalize_whitespaces


def tokenize(s):
    return [token for token in re.split(r'[.,!?\- ;:…\n"«»]', s) if token]


vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY',
          'EH', 'ER', 'EY',
          'IH', 'IY',
          'OW', 'OY',
          'UH', 'UW']


def extract_base_and_ending(word):
    # Define SUFFIX RULES in order of priority (longest first)
    SUFFIX_RULES = [
        # Common suffixes with spelling changes
        ('ization', ('ize', 'ation')),  # e.g., "organization" → ("organize", "ation")
        ('ational', ('ate', 'al')),  # e.g., "relational" → ("relate", "al")
        ('fulness', ('ful', 'ness')),  # e.g., "thankfulness" → ("thankful", "ness")
        ('ibility', ('ible', 'ity')),  # e.g., "sensibility" → ("sensible", "ity")
        ('ically', ('ic', 'ally')),  # e.g., "logically" → ("logic", "ally")
        ('ality', ('al', 'ity')),  # e.g., "formality" → ("formal", "ity")
        ('ation', ('ate', 'ion')),  # e.g., "creation" → ("create", "ion")
        ('ition', ('ite', 'ion')),  # e.g., "ignition" → ("ignite", "ion")
        ('ator', ('ate', 'or')),  # e.g., "operator" → ("operate", "or")
        ('alism', ('al', 'ism')),  # e.g., "nationalism" → ("national", "ism")
        ('iveness', ('ive', 'ness')),  # e.g., "creativeness" → ("creative", "ness")
        ('fulness', ('ful', 'ness')),  # e.g., "peacefulness" → ("peaceful", "ness")
        ('ousness', ('ous', 'ness')),  # e.g., "seriousness" → ("serious", "ness")
        ('ishing', ('ish', 'ing')),  # e.g., "vanishing" → ("vanish", "ing")
        ('ement', ('e', 'ment')),  # e.g., "judgement" → ("judge", "ment")
        ('fully', ('ful', 'ly')),  # e.g., "carefully" → ("careful", "ly")
        ('ingly', ('ing', 'ly')),  # e.g., "surprisingly" → ("surprising", "ly")
        ('ality', ('al', 'ity')),  # e.g., "personality" → ("personal", "ity")
        ('less', ('', 'less')),  # e.g., "hopeless" → ("hope", "less")
        ('ness', ('', 'ness')),  # e.g., "kindness" → ("kind", "ness")
        ('ship', ('', 'ship')),  # e.g., "friendship" → ("friend", "ship")
        ('able', ('', 'able')),  # e.g., "comfortable" → ("comfort", "able")
        ('ible', ('', 'ible')),  # e.g., "flexible" → ("flex", "ible")
        ('hood', ('', 'hood')),  # e.g., "childhood" → ("child", "hood")
        ('dom', ('', 'dom')),  # e.g., "freedom" → ("free", "dom")
        ('like', ('', 'like')),  # e.g., "childlike" → ("child", "like")
        ('wise', ('', 'wise')),  # e.g., "clockwise" → ("clock", "wise")

        # Diminutive suffixes (new additions)
        ('let', ('', 'let')),  # e.g., "booklet" → ("book", "let")
        ('ling', ('', 'ling')),  # e.g., "duckling" → ("duck", "ling")
        ('ie', ('y', 'ie')),  # e.g., "doggie" → ("doggy", "ie") [colloquial]
        #('y', ('', 'y')),  # e.g., "kitty" → ("kit", "y") [colloquial]
        ('ette', ('', 'ette')),  # e.g., "kitchenette" → ("kitchen", "ette")
        ('kin', ('', 'kin')),  # e.g., "napkin" → ("nap", "kin") [historical]
        ('cule', ('', 'cule')),  # e.g., "molecule" → ("mole", "cule") [rare]
        ('ock', ('', 'ock')),  # e.g., "hillock" → ("hill", "ock")

        # Verb/Adjective suffixes (with consonant doubling checks)
        ('iest', ('y', 'est')),  # e.g., "happiest" → ("happy", "est")
        ('ier', ('y', 'er')),  # e.g., "funnier" → ("funny", "er")
        ('ied', ('y', 'ed')),  # e.g., "cried" → ("cry", "ed")
        ('ying', ('ie', 'ing')),  # e.g., "tying" → ("tie", "ing")
        ('ing', ('', 'ing')),  # e.g., "running" → ("run", "ing")
        ('est', ('', 'est')),  # e.g., "tallest" → ("tall", "est")
        ('er', ('', 'er')),  # e.g., "taller" → ("tall", "er")
        ('ed', ('', 'ed')),  # e.g., "jumped" → ("jump", "ed")
        ('es', ('', 'es')),  # e.g., "boxes" → ("box", "es")
        ('s', ('', 's')),  # e.g., "dogs" → ("dog", "s")
        ('ly', ('', 'ly')),  # e.g., "quickly" → ("quick", "ly")
    ]

    # # Check for IRREGULAR forms first (e.g., "children" → "child")
    # IRREGULAR_MAP = {
    #     'children': ('child', 'ren'),
    #     'geese': ('goose', ''),
    #     'mice': ('mouse', ''),
    #     'men': ('man', ''),
    #     'women': ('woman', ''),
    #     'teeth': ('tooth', ''),
    #     'feet': ('foot', ''),
    #     'knives': ('knife', 's'),
    #     'leaves': ('leaf', 's'),
    # }
    #
    # if word.lower() in IRREGULAR_MAP:
    #     return IRREGULAR_MAP[word.lower()]

    # Apply suffix rules in priority order
    for suffix, (base_repl, ending) in SUFFIX_RULES:
        if word.endswith(suffix):
            base = word[:-len(suffix)] + base_repl

            # Handle consonant doubling (e.g., "running" → "run")
            if (len(base) > 1 and base[-1] == base[-2] and base[-1] in 'bcdfghjklmnpqrstvwxyz'):
                yield (base, ending)  # chillest ==> chill + est
                base = base[:-1]
                yield (base, ending)  # running ==> run + ing
            else:
                yield (base, ending)

    # Default: no suffix found
    yield (word, "")


common_second_parts = {
    # Common Second Parts (Concrete Objects)
    "second_parts": [
        "ball", "berry", "bird", "board", "boat", "book", "box", "bread",
        "bridge", "cake", "card", "cat", "child", "coat", "corn", "cream",
        "dog", "door", "dust", "eye", "fall", "field", "fire", "fish",
        "flower", "fly", "fruit", "glass", "grass", "ground", "hall",
        "hand", "head", "hill", "house", "land", "light", "line", "man",
        "mark", "moon", "night", "note", "pad", "paper", "park", "path",
        "phone", "piece", "plant", "port", "post", "room", "ship",
        "shoe", "shop", "side", "sign", "snow", "star", "stone",
        "stop", "store", "storm", "street", "sun", "table", "tail",
        "tea", "time", "town", "tree", "vine", "walk", "wall", "water",
        "way", "wind", "window", "wood", "word", "work", "world"
    ],

    # Abstract/Conceptual Second Parts
    "abstract_second_parts": [
        "age", "ance", "ation", "dom", "ence", "hood", "ism", "ity",
        "ment", "ness", "ship", "th", "tion", "ture", "ance", "cy",
        "ery", "graph", "ics", "logy", "mony", "nomy", "ology", "onym",
        "osis", "pathy", "phobia", "scope", "sophy", "therapy", "tude",
        "ure", "ware", "ways", "work"
    ],

    # Compound-Friendly Suffixes
    "suffixes": [
        "able", "borne", "bound", "breaker", "bright", "caster",
        "claw", "craft", "crest", "dancer", "doom", "dream",
        "drinker", "eye", "fall", "fang", "fire", "flight",
        "flower", "force", "forged", "gaze", "glow", "guard",
        "hammer", "hand", "heart", "helm", "hold", "hunter",
        "kin", "kiss", "light", "ling", "maker", "mark",
        "moon", "mourn", "proof", "rider", "seeker", "shade",
        "shadow", "shard", "shine", "singer", "skin", "song",
        "spear", "spell", "star", "steel", "step", "stone",
        "storm", "stride", "strider", "strike", "sworn", "thorn",
        "tide", "touched", "walker", "ward", "watcher", "weaver",
        "whisper", "wind", "wing", "wise", "wolf", "wood", "word"
    ],

    # Poetic/Archaic Suffixes
    "poetic_suffixes": [
        "born", "bright", "deep", "dusk", "ever", "fair", "fall",
        "gleam", "glen", "hallow", "leafe", "light", "lily",
        "mere", "mist", "quill", "rune", "shade", "silver",
        "song", "soul", "thorn", "tide", "vale", "veil",
        "whisper", "willow", "winter", "wraith"
    ]
}

common_first_parts = {
    # General Nouns (Concrete Objects/Concepts)
    "general_nouns": [
        "air", "back", "bath", "bed", "bird", "book", "car", "cat",
        "city", "cloud", "country", "day", "dog", "door", "eye", "farm",
        "fire", "fish", "foot", "gold", "hand", "head", "heart", "home",
        "house", "key", "land", "light", "moon", "mountain", "night",
        "note", "paper", "rain", "river", "room", "sea", "ship", "shoe",
        "shop", "sky", "snow", "star", "sun", "table", "tea", "time",
        "tooth", "tree", "water", "wind", "wood", "world"
    ],

    # Verbs (Action-Based)
    "verbs": [
        "break", "check", "cut", "drive", "drop", "feed", "flash",
        "fly", "go", "hang", "hold", "jump", "play", "print",
        "read", "run", "scan", "show", "shut", "spin", "stop",
        "swim", "turn", "walk", "wash", "watch", "work", "write"
    ],

    # Adjectives (Descriptive)
    "adjectives": [
        "black", "blue", "cold", "dark", "deep", "fast", "free",
        "full", "green", "hard", "high", "hot", "long", "low",
        "new", "old", "open", "quick", "red", "short", "slow",
        "small", "soft", "strong", "sweet", "tall", "white", "wide"
    ],

    # Prepositions/Directional
    "prepositions": [
        "after", "down", "in", "off", "on", "out", "over",
        "through", "under", "up", "with", "without"
    ],

    # Technology & Science
    "tech_science": [
        "bio", "cyber", "data", "digital", "eco", "electro",
        "geo", "info", "micro", "nano", "neuro", "photo",
        "radio", "robo", "solar", "tech", "tele", "video"
    ],

    # Food & Drink
    "food_drink": [
        "apple", "banana", "beer", "bread", "cake", "cheese",
        "chocolate", "coffee", "egg", "fish", "fruit", "grape",
        "honey", "ice", "lemon", "milk", "nut", "pepper",
        "rice", "salt", "sugar", "tea", "wine"
    ],

    # Body Parts
    "body_parts": [
        "arm", "bone", "ear", "finger", "hair", "knee", "lip",
        "nose", "skin", "stomach", "tail", "thumb", "toe", "tongue"
    ],

    # Mythology & Fantasy
    "mythology_fantasy": [
        "dragon", "elf", "fairy", "ghost", "giant", "goblin", "griffin",
        "kraken", "mage", "mermaid", "orc", "phoenix", "shadow", "sorcerer",
        "spell", "troll", "unicorn", "vampire", "warlock", "werewolf", "witch",
        "wizard", "wyvern", "yeti", "zombie", "arcane", "celestial", "demon",
        "divine", "enchanted", "infernal", "magic", "mythic", "runed", "sacred"
    ],

    # Poetic Compounds (Nouns/Adjectives)
    "poetic": [
        "moon", "star", "sun", "sky", "dream", "soul", "heart", "tear",
        "blood", "bone", "ash", "dust", "shadow", "light", "dark", "silver",
        "golden", "crimson", "ebony", "ivory", "sapphire", "amber", "gossamer",
        "whisper", "eternal", "fading", "lone", "weeping", "wandering", "wild",
        "storm", "wind", "sea", "flower", "thorn", "raven", "dove", "serpent",
        "wing", "song", "hymn", "lament", "dirge", "chasm", "abyss", "twilight"
    ],

    # Poetic Verbs (Present Participles/Archaic)
    "poetic_verbs": [
        "blessed", "blighted", "dancing", "fallen", "flying", "howling",
        "laughing", "shining", "singing", "sleeping", "weeping", "whispering",
        "withering", "wandering"
    ]
}


class RhymedWords(object):
    def __init__(self, clausula1: str, clausula2: str, score: float):
        self.clausula1 = clausula1
        self.clausula2 = clausula2
        self.score = score

    def __repr__(self):
        if self.score == 1.0:
            return f'{self.clausula1} = {self.clausula2}'
        elif self.score > 0.0:
            return f'{self.clausula1} ≈({self.score})≈ {self.clausula2}'
        else:
            return ''

    @staticmethod
    def unrhymed():
        return RhymedWords(clausula1=None, clausula2=None, score=0.0)

    def __bool__(self) -> bool:
        return self.score > 0.0



phoneme_matching = {('T', 'TH'): 0.85,
                    ('D', 'DH'): 0.85,
                    ('T', 'DH'): 0.70,
                    ('D', 'TH'): 0.70,

                    ('IY1', 'IH1'): 0.85,  # fit - feet
                    }


class RhymeDetector_Phonemes_1to1(object):
    def __self__(self):
        pass

    @staticmethod
    def match_phonemes(f1, f2):
        if f1 == f2:
            return 1.0

        if f1[:2] in vowels and f1[:2] == f2[:2] and ((f1[2] == '0' and f2[2] == '2') or (f1[2] == '2' and f2[2] == '0')):
            # "-way" часто выступает в роли второй части составных слов со вторичным ударением.
            # А в клаузуле "more way" мы имеем безударную "way". Надо сделать, чтобы они считались близкими с точки зрения рифмовки.
            # Norway <==> more way
            return 0.99

        if (f1, f2) in phoneme_matching:
            return phoneme_matching[(f1, f2)]

        if (f2, f1) in phoneme_matching:
            return phoneme_matching[(f2, f1)]

        return 0.0

    def fit(self, phonemes1, phonemes1_str, phonemes2, phonemes2_str) -> bool:
        if len(phonemes1) == len(phonemes2):
            score = np.prod([self.match_phonemes(f1, f2) for f1, f2 in zip(phonemes1, phonemes2)])
            return score

        return 0.0


class RhymeDetector_ClausulaSlameRx(object):
    def __init__(self):
        self.rx_rules = []

        rules = [('IH1 R AH0 L$', 'ER1 AH0 L$', 0.95), # Cyril  <==>  squirrel
                 ('AA1 M AH0 K AH0 L$', 'AA1 M IH0 K AH0 L$', 0.95),  # anatomical <==> economical
                 #('AA1 M IH0 K AH0 L$')),  # comical <==>
                 ]

        for rx1, rx2, score in rules:
            self.rx_rules.append((re.compile(rx1), re.compile(rx2), score))

    def fit(self, phonemes1, phonemes1_str, phonemes2, phonemes2_str) -> bool:
        for rx1, rx2, score in self.rx_rules:
            if rx1.search(phonemes1_str) and rx2.search(phonemes2_str):
                return score

        return 0.0


class EnglishWordPronunciation(object):
    def __init__(self, form, syllables, phonemes):
        self.form = form
        self.syllables = syllables
        self.phonemes = phonemes
        self.stress_signature = []
        self.new_stress_pos = -1

        vowel_count = 0
        for phoneme in phonemes:
            if phoneme[:2] in vowels:
                stress = int(phoneme[2])
                self.stress_signature.append(stress)
                if stress == 1:
                    self.new_stress_pos = vowel_count
                vowel_count += 1
        self.n_vowels = vowel_count

    def get_score(self) -> float:
        return 1.0

    def __repr__(self):
        return self.form


class EnglishWord(object):
    def __init__(self, form, pronunciations, syllabizations):
        self.form = form
        self.pronunciations = [EnglishWordPronunciation(form, syllables, phonemes) for phonemes, syllables in zip(pronunciations, syllabizations)]
        self.next_word = None

    def __repr__(self):
        return self.form

    @staticmethod
    def build_start_node():
        return EnglishWord(form=None, pronunciations=[], syllabizations=[])


class MetreMappingResult(object):
    def __init__(self, prefix, metre_signature):
        self.score = 1.0
        self.word_mappings = []
        self.stress_shift_count = 0
        self.prefix = prefix
        self.metre_signature = metre_signature
        self.cursor = 0

    def count_prev_unstressed_syllables(self):
        num_unstressed_syllables = 0
        for word_mapping in self.word_mappings[::-1]:
            if word_mapping.word.new_stress_pos == -1:
                num_unstressed_syllables += word_mapping.word.n_vowels
            else:
                break
        return num_unstressed_syllables

    @staticmethod
    def build_for_empty_line():
        r = MetreMappingResult(prefix=0, metre_signature=None)
        return r

    @staticmethod
    def build_from_source(src_mapping, new_cursor):
        new_mapping = MetreMappingResult(src_mapping.prefix, src_mapping.metre_signature)
        new_mapping.score = src_mapping.score
        new_mapping.word_mappings = list(src_mapping.word_mappings)
        new_mapping.stress_shift_count = src_mapping.stress_shift_count
        new_mapping.cursor = new_cursor
        return new_mapping

    def add_word_mapping(self, word_mapping):
        self.word_mappings.append(word_mapping)
        self.score *= word_mapping.get_total_score()
        if word_mapping.stress_shift:
            self.stress_shift_count += 1

    def finalize(self):
        # ищем цепочки безударных слогов (000...) длиннее 3х подряд, и штрафуем.
        signature = list(itertools.chain(*[m.word.stress_signature for m in self.word_mappings]))
        s = ''.join(map(str, signature))
        for m in re.findall(r'(0{4,})', s):
            factor = 0.1

            # безударный промежуток внутри строки - то есть слева и справа есть ударные слоги
            l = len(m)
            i = s.index(m)
            if l <= 5 and '1' in s[:i] and '1' in s[i+l:]:
                swi = list(itertools.chain(*[[i]*len(m.word.stress_signature) for i, m in enumerate(self.word_mappings)]))
                if len(set(swi[i: i+l])) <= 2:
                    factor = {4: 0.80, 5: 0.50}[l]
                else:
                    factor = {4: 0.30, 5: 0.20}[l]

            self.score *= factor

        return

    def count_stress_marks(self) -> int:
        n = sum(word_mapping.count_stress_marks() for word_mapping in self.word_mappings)
        return n

    def get_stressed_line(self, show_syllables) -> str:
        s = ' '.join(word_mapping.render_accentuation(show_syllables) for word_mapping in self.word_mappings)
        s = normalize_whitespaces(s)
        return s

    def get_stress_signature_str(self) -> str:
        return ''.join(word_mapping.get_stress_signature_str() for word_mapping in self.word_mappings)

    def __repr__(self):
        if self.word_mappings:
            sx = []

            for word_mapping in self.word_mappings:
                sx.append(str(word_mapping))

            sx.append('〚' + '{:6.2g}'.format(self.score).strip() + '〛')
            return ' '.join(sx)
        else:
            return '««« EMPTY »»»'

    def get_score(self):
        stress_shift_factor = 1.0 if self.stress_shift_count < 2 else pow(0.5, self.stress_shift_count)
        return self.score * stress_shift_factor # * self.src_line_variant_score

    def get_canonic_meter(self):
        if self.metre_signature == (0, 1):
            return 'ямб' if self.prefix == 0 else 'хорей'
        elif self.metre_signature == (1, 0):
            return 'хорей' if self.prefix == 0 else 'ямб'
        elif len(self.metre_signature) == 3:
            m = list(self.metre_signature)
            if self.prefix == 1:
                m = m[-1:] + m[:-1]
            m = tuple(m)
            if m == (1, 0, 0):
                return 'дактиль'
            elif m == (0, 1, 0):
                return 'амфибрахий'
            elif m == (0, 0, 1):
                return 'анапест'
            else:
                raise NotImplementedError()
        else:
            return ''


class WordMappingResult(object):
    def __init__(self, word: EnglishWordPronunciation, TP: int, FP: int, TN: int, FN: int, syllabic_mapping, stress_shift, additional_score_factor):
        self.word = word
        self.TP = TP
        self.FP = FP
        self.TN = TN
        self.FN = FN
        self.syllabic_mapping = syllabic_mapping
        self.metre_score = pow(0.1, FP) * pow(0.95, FN) * additional_score_factor
        self.total_score = self.metre_score * word.get_score()
        self.stress_shift = stress_shift

    def get_total_score(self):
        return self.total_score

    def count_stress_marks(self) -> int:
        return self.syllabic_mapping.count('TP')

    def get_stress_signature_str(self) -> str:
        rendering = []

        for syllable, syllable_stress, meter_mapping in zip(self.word.syllables, self.word.stress_signature, self.syllabic_mapping):
            if meter_mapping == 'TP':
                # Тут ударение. Посмотрим, это основное или вторичное.
                if syllable_stress == 2:
                    # вторичное
                    rendering.append(2)
                elif syllable_stress == 1:
                    # основное
                    rendering.append(1)
                else:
                    raise NotImplementedError()
            else:
                rendering.append(0)

        return ''.join(map(str, rendering))

    def render_accentuation(self, show_syllables=False) -> str:
        rendering = []

        if len(self.syllabic_mapping) == 0:
            # Punctuations
            rendering.append(self.word.form)
        else:
            for syllable, syllable_stress, meter_mapping in zip(self.word.syllables, self.word.stress_signature, self.syllabic_mapping):
                if meter_mapping == 'TP':
                    # Тут ударение. Посмотрим, это основное или вторичное.
                    if syllable_stress == 2:
                        # вторичное
                        rendering.append(inject_secondary_stress(syllable))
                    elif syllable_stress == 1:
                        # основное
                        rendering.append(inject_stress_mark(syllable))
                    else:
                        raise NotImplementedError()
                else:
                    rendering.append(syllable)

        if show_syllables:
            return '|'.join(rendering)
        else:
            return ''.join(rendering)


    def __repr__(self):
        s = self.render_accentuation()
        if self.total_score != 1.0:
            s += '[' + '{:5.2g}'.format(self.total_score).strip() + ']'
        return s


class MetreMappingCursor(object):
    def __init__(self, metre_signature: List[int], prefix: int):
        self.prefix = prefix
        self.metre_signature = metre_signature
        self.length = len(metre_signature)

    def get_stress(self, cursor) -> int:
        """Возвращает ударность, ожидаемую в текущей позиции"""
        if self.prefix:
            if cursor == 0:
                return 0
            else:
                return self.metre_signature[(cursor - self.prefix) % self.length]
        else:
            return self.metre_signature[cursor % self.length]

    def map(self, stressed_words_chain, aligner):
        start_results = [MetreMappingResult(self.prefix, self.metre_signature)]
        final_results = []
        self.map_chain(prev_node=stressed_words_chain[0], prev_results=start_results, aligner=aligner, final_results=final_results)
        final_results = sorted(final_results, key=lambda z: -z.get_score())
        return final_results

    def map_chain(self, prev_node, prev_results, aligner, final_results):
        cur_slot = prev_node.next_word

        cur_results = self.map_word(stressed_word_group=cur_slot, results=prev_results, aligner=aligner)
        if cur_slot.next_word:
            self.map_chain(prev_node=cur_slot, prev_results=cur_results, aligner=aligner, final_results=final_results)
        else:
            for result in cur_results:
                result.finalize()
                final_results.append(result)

    def map_word(self, stressed_word_group, results: [MetreMappingResult], aligner):
        new_results = []

        for prev_result in results:
            for word_mapping, new_cursor in self.map_word1(stressed_word_group, prev_result, aligner):
                if word_mapping.word.new_stress_pos == -1:
                    # Пресекаем появление цепочек из безударных слогов длиной более 4.
                    n = prev_result.count_prev_unstressed_syllables()
                    if word_mapping.word.n_vowels + n >= 6:  # >= 4
                        continue
                next_metre_mapping = MetreMappingResult.build_from_source(prev_result, new_cursor)
                next_metre_mapping.add_word_mapping(word_mapping)
                new_results.append(next_metre_mapping)

        new_results = sorted(new_results, key=lambda z: -z.get_score())

        return new_results

    def map_word1(self, stressed_word_group, result: MetreMappingResult, aligner):
        result_mappings = []
        additional_score_factor = 1.0

        for stressed_word in stressed_word_group.pronunciations:
            cursor = result.cursor
            TP, FP, TN, FN = 0, 0, 0, 0
            syllabic_mapping = []

            if stressed_word.stress_signature == [1]  and self.get_stress(cursor) == 0:
                # Односложное слово, в этом месте ожидается безударный слог - делаем слово безударным с некоторым дисконтом.
                # Дисконт можно делать 0 для служебных слов и некоторым для значащих слов (хорошо бы подтянуть список частот односложных слов,
                # и дисконт привязать к этой частоте)
                TP, FP, FN = 0, 0, 0

                if False: # stressed_word.form in ('have', 'had', 'was', 'were', 'must', 'can', 'should', 'could', 'need', 'or', 'both', 'a', 'of', 'for', 'by'):
                    TN = 1
                else:
                    #TN = 0.95
                    #FP = 0.05
                    TN = 1
                    additional_score_factor *= 0.95

                syllabic_mapping = ['TN']

                cursor += 1
            else:
                for word_sign in stressed_word.stress_signature:
                    metre_sign = self.get_stress(cursor)
                    if metre_sign == 1:
                        if word_sign == 1:
                            # Ударение должно быть и оно есть
                            TP += 1
                            syllabic_mapping.append('TP')
                        elif word_sign == 2:
                            # Ударение должно быть, и есть слабое на этом слоге
                            #TP += 0.5
                            TP += 1
                            additional_score_factor *= 0.98
                            syllabic_mapping.append('TP')
                        elif word_sign == 0:
                            # ударение должно быть, но его нет
                            FN += 1
                            syllabic_mapping.append('FN')
                        else:
                            raise RuntimeError()
                    else:
                        if word_sign == 1:
                            # Ударения не должно быть, но оно есть
                            FP += 1
                            syllabic_mapping.append('FP')
                        elif word_sign == 2:
                            # Ударения не должно быть, и в этом месте оно слабое.
                            TN += 1
                            syllabic_mapping.append('TN')
                        elif word_sign == 0:
                            # Ударения не должно быть, и его нет
                            TN += 1
                            syllabic_mapping.append('TN')
                        else:
                            raise RuntimeError()
                    cursor += 1

            # Проверим сочетание ударения в предыдущем слове и в текущем, в частности - оштрафуем за два ударных слога подряд
            if len(stressed_word.stress_signature) > 0:
                if len(result.word_mappings) > 0:
                    prev_mapping = result.word_mappings[-1]
                    if prev_mapping.syllabic_mapping:  # prev_mapping.word.stress_signature:
                        if prev_mapping.syllabic_mapping[-1] == 'TP':  # prev_mapping.word.stress_signature[-1] == 1:  # предыдущее слово закончилось ударным слогом
                            if syllabic_mapping[0] == 1:  # stressed_word.stress_signature[0] == 1:
                                # большой штраф за два ударных подряд
                                additional_score_factor *= 0.1

            mapping1 = WordMappingResult(stressed_word,
                                         TP, FP, TN, FN,
                                         syllabic_mapping=syllabic_mapping,
                                         stress_shift=False,
                                         additional_score_factor=additional_score_factor)
            result_mappings.append((mapping1, cursor))

        return result_mappings


def inject_secondary_stress(syllable: str) -> str:
    cx = []
    injected = False
    for c in syllable:
        if c in 'aeiou' and not injected:
            cx.append('∘')
            injected = True

        cx.append(c)

    return ''.join(cx)


def inject_stress_mark(syllable: str) -> str:
    cx = []
    injected = False
    for c in syllable:
        if c in 'aeiouyAEIOUY' and not injected:
            cx.append('+')
            injected = True

        cx.append(c)

    return ''.join(cx)


def Aa(s: str) -> str:
    return s[0].upper() + s[1:].lower()


def is_Aa(s: str) -> bool:
    return re.match(r"^[A-Z][a-z]*['’]?[a-z]+$", s) is not None


class RhymeGraphNode(object):
    def __init__(self):
        self.offset_to_right = 0
        self.fit_to_right = None
        self.offset_to_left = 0
        self.fit_to_left = None
        self.rhyme_scheme_letter = '-'


def load_cmudict(file_path):
    cmudict = collections.defaultdict(list)
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line:
                if line.startswith(';;;'):  # Skip comments
                    continue
                line = re.sub(r'#(.+)$', '', line).strip()
                word, *phonemes = line.strip().split()

                m = re.match(r'^(.+)\(\d\)$', word)
                if m:
                    word = m.group(1)

                cmudict[word.lower()].append(phonemes)  # Store in lowercase

    return cmudict


suffix2phones = {
    "est": ["AH0", "S", "T"],
    #"est": ['T', 'AO1', 'L', 'IH0', 'S', 'T'],  # tallest
    "ization": ['AH0', 'Z', 'EY1', 'SH', 'AH0', 'N'],  # organization
    "ational": ['EY1', 'SH', 'AH0', 'N', 'AH0', 'L'],  # relational
    "fulness": ['F', 'AH0', 'L', 'N', 'AH0', 'S'],  # awfulness
    "ibility": ['IH0', 'B', 'IH1', 'L', 'IH0', 'T', 'IY0'],  # sensibility
    "ically": ['IH0', 'K', 'L', 'IY0'],  # logically
    "ality": ['AE1', 'L', 'AH0', 'T', 'IY0'],  # formality
    "ation": ['EY1', 'SH', 'AH0', 'N'],  # creation
    "ition": ['IH1', 'SH', 'AH0', 'N'],  # ignition
    "ator": ['EY2', 'T', 'ER0'],  # operator
    "alism": ['AH0', 'L', 'IH2', 'Z', 'AH0', 'M'],  # nationalism
    "iveness": ['IH0', 'V', 'N', 'AH0', 'S'],  # creativeness
    "fulness": ['F', 'AH0', 'L', 'N', 'AH0', 'S'],  # peacefulness
    "ousness": ['AH0', 'S', 'N', 'AH0', 'S'],  # seriousness
    "ishing": ['IH0', 'SH', 'IH0', 'NG'],  # vanishing
    "ement": ['M', 'AH0', 'N', 'T'],  # judgement
    "fully": ['F', 'AH0', 'L', 'IY0'],  # carefully
    "ingly": ['IH0', 'NG', 'L', 'IY0'],  # surprisingly
    "ality": ['AE1', 'L', 'IH0', 'T', 'IY0'],  # personality
    "less": ['L', 'AH0', 'S'],  # hopeless
    "ness": ['N', 'AH0', 'S'],  # kindness
    "ship": ['SH', 'IH0', 'P'],  # friendship
    "able": ['AH0', 'B', 'AH0', 'L'],  # comfortable
    "ible": ['AH0', 'B', 'AH0', 'L'],  # flexible
    "hood": ['HH', 'UH2', 'D'],  # childhood
    "dom": ['D', 'AH0', 'M'],  # freedom
    "like": ['L', 'AY2', 'K'],  # childlike
    "wise": ['W', 'AY2', 'Z'],  # clockwise
    "let": ['L', 'IH0', 'T'],  # booklet
    "ling": ['L', 'IH0', 'NG'],  # duckling
    "ie": ['IY0'],  # doggie
    "ette": ['N', 'EH1', 'T'],  # kitchenette
    "kin": ['K', 'IH0', 'N'],  # napkin
    "cule": ['K', 'Y', 'UW2', 'L'],  # molecule
    "ock": ['AH0', 'K'],  # hillock
    "iest": ['AH0', 'S', 'T'],  # happiest
    "ier": ['IY0', 'ER0'],  # funnier
    "ied": ['AY1', 'D'],  # cried
    "ying": ['IH0', 'NG'],  # tying
    "ing": ['IH0', 'NG'],  # running
    "er": ['ER0'],  # taller
    "ed": ['T'],  # jumped
    "es": ['AH0', 'Z'],  # boxes
    #"s": ['D', 'AA1', 'G', 'Z'],  # dogs
    "ly": ['L', 'IY0'],  # quickly
        }



def replace_stress_to_secondary(phones):
    new_phones = []
    for phone in phones:
        new_phone = phone

        if phone[:2] in vowels:
            if phone[3] == '1':
                # Primary stress convert to secondary.
                new_phone = phone[:2] + '2'
            elif phone[3] == '2':
                # Supress secondary stress
                new_phone = phone[:2] + '0'

        new_phones.append(new_phone)

    return new_phones



class EnglishPoemScansion(object):
    def __init__(self, line_mappings, meter_name, rhyme_graph, rhyme_scheme, score):
        self.line_mappings = line_mappings
        self.meter_name = meter_name
        self.rhyme_graph = rhyme_graph
        self.rhyme_scheme = rhyme_scheme
        self.score = score

    def get_stressed_lines(self, show_syllables):
        lines = [line_mapping.get_stressed_line(show_syllables=show_syllables) for line_mapping in self.line_mappings]
        return '\n'.join(lines)

    @staticmethod
    def build_from_stanzas(stanzas):
        rhyme_scheme = ' '.join(stanza.rhyme_scheme for stanza in stanzas)
        score = min(stanza.score for stanza in stanzas)

        line_mappings = []
        rhyme_graph = []
        for stanza in stanzas:
            if line_mappings:
                line_mappings.append(MetreMappingResult.build_for_empty_line())
                rhyme_graph.append(0)
            line_mappings.extend(stanza.line_mappings)
            rhyme_graph.extend(stanza.rhyme_graph)

        return EnglishPoemScansion(line_mappings=line_mappings,
                                   meter_name=stanzas[0].meter_name,
                                   rhyme_graph=rhyme_graph,
                                   rhyme_scheme=rhyme_scheme,
                                   score=score)
    


class EnglishPoetryScansion(object):
    def __init__(self, model_dir: str):
        # Load the dictionary
        with open(os.path.join(model_dir, "english_scansion_tool.pkl"), "rb") as f:
            self.pronouncing_dict = pickle.load(f)
            self.word2syllables = pickle.load(f)
            self.head_syllables = pickle.load(f)
            self.tail_syllables = pickle.load(f)

        self.rhyme_detectors = []
        self.rhyme_detectors.append(RhymeDetector_Phonemes_1to1())
        self.rhyme_detectors.append(RhymeDetector_ClausulaSlameRx())
        self.syllabler = eng_syl.syllabify.Syllabel()
        self.hyph_dic = pyphen.Pyphen(lang='en')
        self.g2p = G2p()

    def get_phonemes(self, word0):
        """Get the phoneme sequence for a word"""
        pronunciations = None

        word = word0.lower()
        m1 = re.match(r"^(\w+)’\w*$", word)
        if m1:
            word = word.replace("’", "'")

        if word in self.pronouncing_dict:
            pronunciations = self.pronouncing_dict[word]
            return pronunciations

        m1 = re.match(r"^(\w+)[’'](s|ve|re|m|d)$", word)
        if m1:
            stem = m1.group(1)
            tail = m1.group(2)

            tail_phoneme = {"s": "Z", "ve": "V", "re": "R", "m": "M", "d": "D"}[tail]

            pronunciations = []
            for pronunciation0 in self.get_phonemes(stem):
                pronunciations.append(pronunciation0 + [tail_phoneme])

            return pronunciations

        if word not in self.pronouncing_dict:
            if word.endswith("'"):
                # enemies'
                return self.get_phonemes(word0[:-1])

            if re.match(r'^\W+$', word):
                pronunciations = [[]]
            else:
                for lemma, suffix in extract_base_and_ending(word):
                    if suffix == 's' and lemma in self.pronouncing_dict:
                        suffix_phones = []
                        if lemma[-1] in 'BDGV':
                            suffix_phones = ['Z']
                        elif lemma[-1] in 'SZ':
                            suffix_phones = []
                        else:
                            suffix_phones = ['S']
                        pronunciations = []
                        for pronunciation0 in self.get_phonemes(lemma):
                            pronunciations.append(pronunciation0 + suffix_phones)
                        break

                    if suffix and lemma in self.pronouncing_dict:
                        phones2 = suffix2phones[suffix]
                        pronunciations = []
                        for pronunciation0 in self.get_phonemes(lemma):
                            pronunciations.append(pronunciation0 + phones2)
                        break

                # Handle compound words with frequent first part (like bio-mechanical)
                if not pronunciations:
                    for partition, first_parts in common_first_parts.items():
                        for first_part in first_parts:
                            if word.startswith(first_part) and first_part in self.pronouncing_dict:
                                tail = word[len(first_part):]
                                if tail in self.pronouncing_dict:
                                    head_phones = self.pronouncing_dict[first_part]
                                    tail_phones = self.pronouncing_dict[tail]
                                    for h, t in itertools.product(head_phones, tail_phones):
                                        # Construct two alternative variants where stress is either on first part or on second part.
                                        pronunciations.append(replace_stress_to_secondary(h) + t)
                                        pronunciations.append(h + replace_stress_to_secondary(t))

                # Handle compound words with frequent second part (like human-ware)
                if not pronunciations:
                    for partition, second_parts in common_second_parts.items():
                        for second_part in second_parts:
                            if word.endswith(second_part) and second_part in self.pronouncing_dict:
                                head = word[:len(second_part)]
                                if head in self.pronouncing_dict:
                                    head_phones = self.pronouncing_dict[head]
                                    tail_phones = self.pronouncing_dict[second_part]
                                    for h, t in itertools.product(head_phones, tail_phones):
                                        pronunciations.append(replace_stress_to_secondary(h) + t)
                                        pronunciations.append(h + replace_stress_to_secondary(t))

            if not pronunciations:
                pronunciation = self.g2p(word)
                if pronunciation:
                    pronunciations = [pronunciation]

        if not pronunciations:
            raise RuntimeError(f'Could not generate pronunciation for word "{word}"')

        return pronunciations

    def syllabify_with_pronouncing(self, word, phones=None):
        # Heuristic #1 - detect words with one of now syllable leteters.
        if len(re.findall(r'[aeiouy]', word, flags=re.I)) <= 1:
            return [word]

        lword = word.lower()

        if phones is None:
            phones_list = pronouncing.phones_for_word(lword)
            if not phones_list:
                phones2 = self.get_phonemes(word)
                if phones2:
                    phones = phones2[0]
                else:
                    return [word]
            else:
                phones = phones_list[0].split()

        vowel_count = sum((phone[:2] in vowels) for phone in phones)
        if vowel_count <= 1:
            return [word]

        # Beam search syllabifier
        beam_size = 5
        paths = []
        final_paths = []

        # Search for initial syllable
        for head_len in range(len(word), 0, -1):
            head = word[:head_len]
            if head in self.head_syllables:
                head_freq = self.head_syllables[head]
                if len(head) == len(word):
                    final_paths.append(([head], head_freq))
                else:
                    paths.append(([head], len(head), head_freq))

                    # Take N longest head syllables
                    if len(paths) == beam_size:
                        break

        # Search for next syllable
        while paths:
            new_paths = []

            for path_syllables, path_len, path_total_freq in paths:
                num_resting_chars = len(word) - path_len
                for l in range(num_resting_chars, 1, -1):
                    s = word[path_len: path_len + l]
                    if s in self.tail_syllables:
                        next_freq = self.tail_syllables[s]
                        path = (path_syllables + [s], path_len + l, path_total_freq+next_freq)
                        if path[1] == len(word):
                            final_paths.append((path[0], path[2]))
                        else:
                            new_paths.append(path)

            paths = new_paths

        # Filter the syllabifications by matching with known number of vowels (that is number of syllables)
        if len(final_paths):
            final_paths = [(path_syllables, path_freq) for path_syllables, path_freq in final_paths if len(path_syllables) == vowel_count]

            if final_paths:
                # Sort so split with max total frequency of syllables becomes the top one
                final_paths = sorted(final_paths, key=lambda z: -z[1])

                # Take the split with max total frequencies of syllables as a final solution
                syllables = final_paths[0][0]
                return syllables

        sx = self.syllabler.syllabify(lword)
        if sx:
            syllables = sx.split('-')
            if len(syllables) == vowel_count:
                return syllables

        # -------------------------------------------------------------------
        # Fall-back algorithm for cases when beam search syllabifier fails.

        # Split by vowel phonemes
        syllables_phonemes = []
        current = []
        for p in phones:
            current.append(p)
            if p[-1].isdigit():
                syllables_phonemes.append(current)
                current = []
        if current:
            syllables_phonemes[-1].extend(current)

        # Approximate written syllables by slicing word into parts
        n = len(syllables_phonemes)
        avg_len = len(word) // n
        syllables = []
        i = 0
        for j in range(n - 1):
            syllables.append(word[i:i + avg_len])
            i += avg_len
        syllables.append(word[i:])
        return syllables

    def get_syllables(self, word, phones=None):
        if len(word) < 2 or re.match(r'^\W+$', word):
            return [word]

        lword = word.lower()

        if lword in self.word2syllables:
            sx = list(self.word2syllables[lword])
            if is_Aa(word):
                sx[0] = Aa(sx[0])
            return sx

        sx = self.syllabify_with_pronouncing(lword, phones)
        if is_Aa(word):
            sx[0] = Aa(sx[0])
        return sx

        # # Step 1: Get phoneme-based syllable count
        # phones = pronouncing.phones_for_word(lword)
        # if not phones:
        #     return [word]  # fallback
        #
        # syllable_count = pronouncing.syllable_count(phones[0])
        #
        # # Step 2: Use Pyphen to hyphenate the word (likely written syllables)
        # hyphenated = self.hyph_dic.inserted(word)
        # syllables = hyphenated.split('-')
        #
        # # Step 3: Adjust number of syllables to match phoneme count
        # if len(syllables) == syllable_count:
        #     return syllables
        # elif len(syllables) > syllable_count:
        #     # Merge extras from the end
        #     while len(syllables) > syllable_count:
        #         syllables[-2] += syllables[-1]
        #         syllables.pop()
        #     return syllables
        # else:
        #     # Not enough splits, just return the hyphenated guess
        #     return syllables

    # def render_stressed_form(self, word, phonemes):
    #     stressed_syllable_pos = -1
    #     vowel_count = 0
    #     for phoneme in phonemes:
    #         if phoneme[:2] in vowels:
    #             if phoneme[-1] == '1':
    #                 stressed_syllable_pos = vowel_count
    #                 break
    #             vowel_count += 1
    #
    #     if stressed_syllable_pos >= 0:
    #         syllables = self.get_syllables(word)
    #         sx = syllables[:stressed_syllable_pos] + [inject_stress_mark(syllables[stressed_syllable_pos])] + syllables[stressed_syllable_pos+1:]
    #         stressed_form = ''.join(sx)
    #         return stressed_form
    #     else:
    #         return word

    def get_last_stressed_vowel_syllable(self, phonemes):
        """Find the last stressed vowel syllable in a word's pronunciation"""
        last_stressed = None
        for i in range(len(phonemes)-1, -1, -1):
            phoneme = phonemes[i]
            if any(phoneme.startswith(v) for v in vowels):
                if '1' in phoneme:  # Primary stress
                    last_stressed = i
                    break
                #elif '2' in phoneme and last_stressed is None:  # Secondary stress
                #    last_stressed = i
                #    break

        return last_stressed

    def do_words_rhyme(self, word1, word2) -> RhymedWords:
        """Check if two words rhyme based on their pronunciation"""
        phonemes1 = self.get_phonemes(word1)[0]
        phonemes2 = self.get_phonemes(word2)[0]
        return self.do_pronunciation_rhyme(phonemes1, phonemes2)

    def do_pronunciation_rhyme(self, phonemes1, phonemes2) -> RhymedWords:
        if not phonemes1 or not phonemes2:
            return RhymedWords.unrhymed()

        idx1 = self.get_last_stressed_vowel_syllable(phonemes1)
        idx2 = self.get_last_stressed_vowel_syllable(phonemes2)

        if idx1 is None or idx2 is None:
            return RhymedWords.unrhymed()

        clausula1 = phonemes1[idx1:]
        clausula2 = phonemes2[idx2:]

        if clausula1 == clausula2:
            return RhymedWords(clausula1=clausula1, clausula2=clausula2, score=1.0)

        clausula1_str = ' '.join(clausula1)
        clausula2_str = ' '.join(clausula2)

        phonemes1_str = ' '.join(phonemes1)
        phonemes2_str = ' '.join(phonemes2)

        for rhyme_detector in self.rhyme_detectors:
            fit_score = rhyme_detector.fit(clausula1, phonemes1_str, clausula2, phonemes2_str)
            if fit_score > 0.0:
                return RhymedWords(clausula1=clausula1_str, clausula2=clausula2_str, score=fit_score)

        return RhymedWords.unrhymed()

    def extract_last_word(self, line):
        """Extract the last word from a line of poetry"""
        # Remove punctuation and split into words
        line = line.translate(str.maketrans('', '', string.punctuation))
        words = line.strip().split()
        return words[-1].lower() if words else None

    def analyze_poem_rhyme_scheme(self, poem_lines):
        """Analyze the rhyme scheme of a poem"""
        rhyme_scheme = []
        rhyme_map = {}
        current_char = ord('A')

        for i, line in enumerate(poem_lines):
            last_word = self.extract_last_word(line)
            if not last_word:
                rhyme_scheme.append(' ')
                continue

            # Check against previous lines
            best_fit = None
            best_score = 0.0
            best_matched_letter = None
            for j in [i-1, i-2, i-3]:
                if j >= 0:
                    prev_last_word = self.extract_last_word(poem_lines[j])
                    if prev_last_word:
                        fit = self.do_words_rhyme(last_word, prev_last_word)
                        if fit.score > best_score:
                            best_fit = fit
                            best_score = fit.score
                            best_matched_letter = rhyme_map[j]
                            if fit.score == 1.0:
                                break

            if best_fit is not None:
                rhyme_scheme.append(best_matched_letter)
                rhyme_map[i] = best_matched_letter
            else:
                rhyme_char = chr(current_char)
                rhyme_scheme.append(rhyme_char)
                rhyme_map[i] = rhyme_char
                current_char += 1

        return ''.join(rhyme_scheme)

    def build_rhyme_graph(self, rhyming_tails):
        line_rhymes = [RhymeGraphNode() for _ in rhyming_tails]
        for i1, tail1 in enumerate(rhyming_tails):
            for i2 in [i1+1, i1+2, i1+3]:
                if i2 < len(rhyming_tails):
                    tail2 = rhyming_tails[i2]

                    fit = self.do_pronunciation_rhyme(tail1, tail2)
                    if fit.score > 0.5:
                        line_rhymes[i1].offset_to_right = (i2-i1)
                        line_rhymes[i1].fit_to_right = fit
                        line_rhymes[i2].offset_to_left = (i1-i2)
                        line_rhymes[i2].fit_to_left = fit
                        break

        current_char = ord('A')
        for i, rhyme in enumerate(line_rhymes):
            if rhyme.offset_to_right > 0:
                if rhyme.rhyme_scheme_letter == '-':
                    rhyme.rhyme_scheme_letter = chr(current_char)
                    current_char += 1

                line_rhymes[i+rhyme.offset_to_right].rhyme_scheme_letter = rhyme.rhyme_scheme_letter

        rhyme_scheme = ''.join([rhyme.rhyme_scheme_letter for rhyme in line_rhymes])

        return line_rhymes, rhyme_scheme

    def parse_line(self, line: str):
        if len(line) == 0:
            return []

        nodes = []

        nodes.append(EnglishWord.build_start_node())

        for form in tokenize_slowly(line.strip()):
            pronunciations = self.get_phonemes(form)
            syllabifications = [self.get_syllables(form, pronunciation) for pronunciation in pronunciations]
            nodes.append(EnglishWord(form, pronunciations, syllabifications))

        for word1, word2 in zip(nodes, nodes[1:]):
            word1.next_word = word2

        return nodes

    def get_prefixes_for_meter(selfself, metre_signature):
        return [0] if len(metre_signature)==2 else [0, 1]

    def enumerate_meters(self, parsed_lines):
        num_syllables = []
        for parsed_line in parsed_lines:
            ns = sum(len(word.pronunciations[0].stress_signature) for word in parsed_line[1:])
            num_syllables.append(ns)

        max_num_syllabs = max(num_syllables)

        yield 'ямб', [(0, 1)]
        yield 'хорей', [(1, 0)]
        yield 'дактиль', [(1, 0, 0)]
        yield 'амфибрахий', [(0, 1, 0)]
        yield 'анапест', [(0, 0, 1)]

    def align(self, poem_lines: List[str]) -> EnglishPoemScansion:
        if '' in poem_lines:
            # There is at least two stanza.
            # Divide the poem lines into stanzas, analyze the stanza sequentially and combine the results.
            results = []

            stanzas = [list(group) for is_empty, group in itertools.groupby(poem_lines, lambda x: x == "") if not is_empty]
            for stanza_lines in stanzas:
                stanza_result = self.align_stanza(stanza_lines)
                results.append(stanza_result)

            return EnglishPoemScansion.build_from_stanzas(results)
        else:
            return self.align_stanza(poem_lines)

    def align_stanza(self, poem_lines: List[str]) -> EnglishPoemScansion:
        plines = [self.parse_line(line) for line in poem_lines]

        # Try every meter, estimate fitness and choose the best one.
        best_score = 0.0
        best_meter_name = None
        best_mappings = None
        best_rhyme_graph = None
        best_rhyme_scheme = None

        for metre_name, metre_signature in self.enumerate_meters(plines):
            line_mappings = []
            for ipline, pline in enumerate(plines):

                best_line_mapping = None

                for prefix in [0, 1]:
                    cursor = MetreMappingCursor(metre_signature[ipline % len(metre_signature)], prefix=prefix)
                    metre_mappings = cursor.map(pline, self)
                    metre_mapping1 = metre_mappings[0]
                    if best_line_mapping is None or best_line_mapping.score < metre_mapping1.score:
                        best_line_mapping = metre_mapping1

                line_mappings.append(best_line_mapping)

            meter_score = np.prod([m.get_score() for m in line_mappings])

            # Evaluate rhyming
            rhyming_tails = [self.extract_rhyming_tail(line_mapping) for line_mapping in line_mappings]
            rhyme_graph, rhyme_scheme = self.build_rhyme_graph(rhyming_tails)

            rhyme_score = 1.0
            for rhyme in rhyme_graph:
                if rhyme.fit_to_right is None:
                    if rhyme.fit_to_left is None:
                        rhyme_score *= 0.5
                else:
                    rhyme_score *= rhyme.fit_to_right.score

            total_score = meter_score * rhyme_score
            if total_score > best_score:
                best_meter_name = metre_name
                best_score = total_score
                best_mappings = line_mappings
                best_rhyme_graph = rhyme_graph
                best_rhyme_scheme = rhyme_scheme

        scansion = EnglishPoemScansion(best_mappings, best_meter_name, best_rhyme_graph, best_rhyme_scheme, best_score)
        return scansion

    def extract_rhyming_tail(self, line_mapping):
        tail_words = []
        for rword in line_mapping.word_mappings[::-1]:
            if len(rword.syllabic_mapping) > 0:
                tail_words.append(rword)
                if 'TP' in rword.syllabic_mapping:
                    break

        tail_phones = []
        for i, tail_word in enumerate(tail_words[::-1]):
            if i == 0:
                # У первого слова в клаузуле оставляем ударения as is
                tail_phones.extend(tail_word.word.phonemes)
            else:
                # У второго (и последующих - маловероятно, но всё таки) слова делаем все гласные безударными.
                for phone in tail_word.word.phonemes:
                    if phone[:2] in vowels:
                        tail_phones.append(phone[:2] + '0')
                    else:
                        tail_phones.append(phone)

        return tuple(tail_phones)



if __name__ == '__main__':
    # Download the CMU Pronouncing Dictionary data
    #nltk.download('cmudict')

    # Load the dictionary
    tool = EnglishPoetryScansion(model_dir="/home/inkoziev/polygon/text_generator/models/scansion_tool")

    sx = tool.get_syllables("Impregnable")
    print(sx)

    sx = tool.get_syllables("story")
    print(sx)

    sx = tool.get_syllables("I've")
    print(sx)

    sx = tool.get_syllables("Extremity")
    print(sx)

    sx = tool.get_syllables("chillest")
    print(sx)

    sx = tool.get_syllables("phantoms")
    print(sx)

    for word in ["I'm", "We've", "Don't", "won't", "doesn't", "They're",  "Dogs"]:
        print("{} ==> {}".format(word, tool.get_phonemes(word)))

    r = tool.do_words_rhyme(word1='feet', word2='fit')
    assert r.score > 0.0

    r = tool.do_words_rhyme(word1='door', word2='or')
    assert r.score > 0.0

    r = tool.do_words_rhyme(word1='teeth', word2='feet')
    assert r.score > 0.0
