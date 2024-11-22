import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import unicodedata
import numpy as np
from .modules import is_Vietnamese
from .modules import VocabBuilder
import os


class VietnameseTokenizer:
    def __init__(self, vocab_path: str = None, annotation_paths: list[str] = None):
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        elif annotation_paths:
            vocab_builder = VocabBuilder(annotation_paths)
            self.vocab = vocab_builder.vocab
            if vocab_path:
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        else:
            raise ValueError("Must provide 'vocab_path' or 'annotation_paths'.")

        if ' ' not in self.vocab['onset']:
            self.vocab['onset'][' '] = len(self.vocab['onset'])
        self.rev_onset_vocab = {v: k for k, v in self.vocab['onset'].items()}
        self.rev_rhyme_vocab = {v: k for k, v in self.vocab['rhyme'].items()}
        self.rev_tone_vocab = {v: k for k, v in self.vocab['tone'].items()}
        self.diacritics = {
            '<`>': '\u0300',
            '</>': '\u0301',
            '<?>' : '\u0309',
            '<~>': '\u0303',
            '<.>': '\u0323',
        }
        self.vowel_groups = {
            # Nguyên âm đôi và ba
            'ai': 'a',
            'ao': 'a',
            'au': 'a',
            'ay': 'a',
            'âu': 'â',
            'ây': 'â',
            'eo': 'e',
            'êu': 'ê',
            'ia': 'i',
            'iê': 'ê',
            'ie': 'e',
            'iu': 'i',
            'oa': 'a',
            'oe': 'o',
            'oi': 'o',
            'ôi': 'ô',
            'ơi': 'ơ',
            'oo': 'o',
            'ua': 'u',
            'uâ': 'â',
            'uê': 'ê',
            'ui': 'u',
            'uy': 'y',
            'ưa': 'ư',
            'ươ': 'ơ',
            'ưu': 'ư',
            'yê': 'ê',
            'yêu': 'ê',
            'iêu': 'ê',
            'uai': 'a',
            'uay': 'a',
            'uây': 'â',
            'uôi': 'ô',
            'ươi': 'ơ',
            'uyê': 'ê',
            'uyên': 'ê',
            'uyê': 'ê',
            'ươu': 'ơ',
            'oai': 'a',
            'oay': 'a',
            'oeo': 'e',
            'oac': 'a',
            'oan': 'a',
        }

    def find_tone_position(self, syllable):
        syllable_nfd = unicodedata.normalize('NFD', syllable)
        vowels = 'aăâeêiioôơuưy'
        letters = []
        idx_map = []

        idx = 0
        char_idx = 0
        while idx < len(syllable_nfd):
            c = syllable_nfd[idx]
            base = c
            combs = ''
            idx += 1
            while idx < len(syllable_nfd) and unicodedata.combining(syllable_nfd[idx]):
                combs += syllable_nfd[idx]
                idx += 1
            letter = unicodedata.normalize('NFC', base + combs)
            letters.append(letter)
            idx_map.append(char_idx)
            char_idx += len(letter)

        is_qu = False
        is_gi = False
        if letters[0].lower() == 'q' and len(letters) > 1 and letters[1].lower() == 'u':
            is_qu = True
        elif letters[0].lower() == 'g' and letters[1].lower() == 'i':
            # Kiểm tra nếu từ có hơn 2 chữ cái
            if len(letters) > 2:
                is_gi = True
            else:
                is_gi = False  # Trường hợp từ có 2 chữ cái, 'g' là âm đầu, 'i' là nguyên âm

        vowel_indices = []
        for i, letter in enumerate(letters):
            if letter.lower() in vowels:
                if (is_qu and i == 1 and letters[i].lower() == 'u') or \
                (is_gi and i == 1 and letters[i].lower() == 'i'):
                    continue
                vowel_indices.append(i)

        vowel_seq = ''.join([letters[i].lower() for i in vowel_indices])

        max_length = min(3, len(vowel_seq))
        for length in range(max_length, 0, -1):
            for start in range(len(vowel_seq) - length + 1):
                seq = vowel_seq[start:start+length]
                if seq in self.vowel_groups:
                    tone_vowel_char = self.vowel_groups[seq]
                    for idx in vowel_indices[start:start+length]:
                        if letters[idx].lower() == tone_vowel_char:
                            pos = idx_map[idx]
                            length = len(letters[idx])
                            return pos, length
                    idx = vowel_indices[start]
                    pos = idx_map[idx]
                    length = len(letters[idx])
                    return pos, length

        priority_vowels = ['ă', 'â', 'ê', 'ô', 'ơ', 'ư']
        for vowel in priority_vowels:
            for idx in vowel_indices:
                if letters[idx].lower() == vowel:
                    pos = idx_map[idx]
                    length = len(letters[idx])
                    return pos, length

        if vowel_indices:
            idx = vowel_indices[-1]
            pos = idx_map[idx]
            length = len(letters[idx])
            return pos, length

        return -1, 0



    def remove_existing_tone(self, syllable):
        syllable_nfd = unicodedata.normalize('NFD', syllable)
        tone_marks = ['\u0300', '\u0301', '\u0303', '\u0309', '\u0323']
        new_syllable_nfd = ''.join([c for c in syllable_nfd if c not in tone_marks])
        return unicodedata.normalize('NFC', new_syllable_nfd)

    def apply_tone(self, syllable, accent):
        syllable = self.remove_existing_tone(syllable)
        vowel_pos, vowel_len = self.find_tone_position(syllable)

        if vowel_pos == -1:
            return syllable

        vowel = syllable[vowel_pos:vowel_pos+vowel_len]
        vowel_nfd = unicodedata.normalize('NFD', vowel)
        base_char = ''
        diacritics_chars = []
        for c in vowel_nfd:
            if not unicodedata.combining(c):
                base_char += c
            else:
                diacritics_chars.append(c)
        diacritics_chars.append(accent)
        diacritic_order = ['\u031B', '\u0306', '\u0302', '\u0309', '\u0300', '\u0301', '\u0303', '\u0323']
        diacritics_chars.sort(key=lambda x: diacritic_order.index(x) if x in diacritic_order else 999)
        new_vowel_nfd = base_char + ''.join(diacritics_chars)
        new_vowel = unicodedata.normalize('NFC', new_vowel_nfd)
        syllable = syllable[:vowel_pos] + new_vowel + syllable[vowel_pos+vowel_len:]

        return syllable

    def process_text(self, text):
        tokens = re.split(r'(\s+)', text)
        for idx, token in enumerate(tokens):
            if not token.strip():
                continue
            for marker in self.diacritics:
                while marker in token:
                    idx_marker = token.find(marker)
                    accent = self.diacritics[marker]
                    token = token[:idx_marker] + token[idx_marker+len(marker):]
                    if token == 'gii':
                        token = 'gi'
                    elif token == 'giin':
                        token = 'gin'
                    token = self.apply_tone(token, accent)
            tokens[idx] = token
        return ''.join(tokens)

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\s]+', ' ', text)
        return text.strip()

    def encode(self, sentence: str, max_length) -> list[list[int]]:
        sentence = sentence.lower()
        tokens = re.findall(r'\S+|\s', sentence)
        phoneme_indices = []

        bos_idx = self.vocab['onset'].get('<bos>', self.vocab['onset']['none'])
        phoneme_indices.append([bos_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])

        for token in tokens:
            token_phonemes = []
            if token.isspace():
                for _ in token:
                    onset_idx = self.vocab['onset'].get('<_>', self.vocab['onset']['none'])
                    token_phonemes.append([onset_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])
            elif re.match(r'^[\W_]+$', token):
                for char in token:
                    onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['none'])
                    token_phonemes.append([onset_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])
            elif token == '<_>':
                token_phonemes.append([self.vocab['onset']['<_>'], self.vocab['rhyme']['none'], self.vocab['tone']['none']])
            else:
                try:
                    is_vietnamese, components = is_Vietnamese(token)
                    if is_vietnamese:
                        onset, rhyme, tone = components
                        onset_idx = self.vocab['onset'].get(onset, self.vocab['onset']['none'])
                        rhyme_idx = self.vocab['rhyme'].get(rhyme, self.vocab['rhyme']['none'])
                        tone_idx = self.vocab['tone'].get(tone, self.vocab['tone']['none'])
                        token_phonemes.append([onset_idx, rhyme_idx, tone_idx])
                    else:
                        for char in token:
                            onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['none'])
                            token_phonemes.append([onset_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])
                except Exception as e:
                    print(f"Error processing word '{token}': {e}")
                    token_phonemes.append([self.vocab['onset']['none'], self.vocab['rhyme']['none'], self.vocab['tone']['none']])

            if len(phoneme_indices) + len(token_phonemes) >= max_length:
                if not is_vietnamese:
                    remaining_length = max_length - len(phoneme_indices) - 1
                    token_phonemes = token_phonemes[:remaining_length]
                else:
                    break

            phoneme_indices.extend(token_phonemes)

        eos_idx = self.vocab['onset'].get('<eos>', self.vocab['onset']['none'])
        phoneme_indices.append([eos_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])

        if len(phoneme_indices) > max_length:
            phoneme_indices = phoneme_indices[:max_length]

        if len(phoneme_indices) < max_length:
            pad_length = max_length - len(phoneme_indices)
            onset_idx_pad = self.vocab['onset'].get('<pad>', self.vocab['onset']['none'])
            rhyme_idx_pad = self.vocab['rhyme'].get('<pad>', self.vocab['rhyme']['none'])
            tone_idx_pad = self.vocab['tone'].get('<pad>', self.vocab['tone']['none'])
            pad_indices = [[onset_idx_pad, rhyme_idx_pad, tone_idx_pad]] * pad_length
            phoneme_indices.extend(pad_indices)

        return phoneme_indices

    def batch_encode(self, sentences: list[str], max_length) -> list[list[list[int]]]:
        sentences = [sentence.lower() for sentence in sentences]
        return [self.encode(sentence, max_length) for sentence in sentences]

    def decode(self, phoneme_matrix: list[list[int]]) -> str:
        pad_onset_idx = self.vocab['onset'].get('<pad>', self.vocab['onset']['none'])
        bos_onset_idx = self.vocab['onset'].get('<bos>', self.vocab['onset']['none'])
        eos_onset_idx = self.vocab['onset'].get('<eos>', self.vocab['onset']['none'])

        sentence = []
        current_word = ""
        for phoneme in phoneme_matrix:
            try:
                onset_index, rhyme_index, tone_index = phoneme

                if onset_index in {pad_onset_idx, bos_onset_idx, eos_onset_idx}:
                    if onset_index == eos_onset_idx:
                        break
                    continue

                onset = self.rev_onset_vocab.get(onset_index, '')
                rhyme = self.rev_rhyme_vocab.get(rhyme_index, '')
                tone = self.rev_tone_vocab.get(tone_index, '')

                if onset == 'none':
                    onset = ''
                if rhyme == 'none':
                    rhyme = ''
                if tone == 'none':
                    tone = ''

                if onset == '<_>':
                    if current_word:
                        sentence.append(current_word)
                        current_word = ""
                    sentence.append(' ')
                    continue


                word = onset + rhyme

                if tone:
                    word = self.apply_tone(word, tone)

                current_word += word
            except Exception as e:
                print(f"Error processing phoneme '{phoneme}': {e}")
                if current_word:
                    sentence.append(current_word)
                    current_word = ""

        if current_word:
            sentence.append(current_word)

        final_sentence = ''.join(sentence)
        return self.process_text(final_sentence)

    def batch_decode(self, phoneme_matrices: list[list[list[int]]]) -> list[str]:
        return [self.decode(phoneme_matrix) for phoneme_matrix in phoneme_matrices]

    def __call__(self, sentences, max_length=30):
        if isinstance(sentences, str):
            return self.encode(sentences, max_length=max_length)
            return encoded
        elif isinstance(sentences, list):
            return self.batch_encode(sentences, max_length=max_length)
            

    def create_mask(self, phoneme_indices: list[list[int]]) -> list[int]:
        onset_idx_pad = self.vocab['onset'].get('<pad>', self.vocab['onset']['none'])
        rhyme_idx_pad = self.vocab['rhyme'].get('<pad>', self.vocab['rhyme']['none'])
        tone_idx_pad = self.vocab['tone'].get('<pad>', self.vocab['tone']['none'])
        
        mask = []
        for phoneme in phoneme_indices:
            if (phoneme[0] == onset_idx_pad and
                phoneme[1] == rhyme_idx_pad and
                phoneme[2] == tone_idx_pad):
                mask.append(1)
            else:
                mask.append(0)
        return mask
