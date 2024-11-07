import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import unicodedata
import numpy as np

# Import other components from the project
from word_processing import is_Vietnamese

class VietnameseTokenizer:
    def __init__(self, vocab: dict):
        self.vocab = vocab

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)  # Normalize to composed form
        text = re.sub(r'[\s]+', ' ', text)  # Replace multiple spaces with a single space
        return text.strip()

    def split_sentence_to_phonemes(self, sentence: str) -> list[list[int]]:
        words = sentence.split()
        phoneme_indices = []

        for idx, word in enumerate(words):
            # Handle punctuation and special characters separately
            if re.match(r'^[\W_]+$', word):
                for char in word:
                    onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['null'])
                    phoneme_indices.append([onset_idx, self.vocab['rhyme']['null'], self.vocab['tone']['null']])
                continue

            if word == '<_>':
                phoneme_indices.append([self.vocab['onset'][' '], self.vocab['rhyme']['null'], self.vocab['tone']['null']])
            else:
                try:
                    is_vietnamese, components = is_Vietnamese(word)
                    if is_vietnamese:
                        onset, rhyme, tone = components
                        onset_idx = self.vocab['onset'].get(onset, self.vocab['onset']['null'])
                        rhyme_idx = self.vocab['rhyme'].get(rhyme, self.vocab['rhyme']['null'])
                        tone_idx = self.vocab['tone'].get(tone, self.vocab['tone']['null'])
                        phoneme_indices.append([onset_idx, rhyme_idx, tone_idx])
                    else:
                        # For non-Vietnamese words, treat each character as onset, with rhyme and tone as 'none'
                        for char in word:
                            onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['null'])
                            rhyme_idx = self.vocab['rhyme']['null']
                            tone_idx = self.vocab['tone']['null']
                            phoneme_indices.append([onset_idx, rhyme_idx, tone_idx])
                except Exception as e:
                    print(f"Error processing word '{word}': {e}")
                    phoneme_indices.append([self.vocab['onset']['null'], self.vocab['rhyme']['null'], self.vocab['tone']['null']])

            # Add space between words if not the last word
            if idx < len(words) - 1:
                phoneme_indices.append([self.vocab['onset']['<_>'], self.vocab['rhyme']['null'], self.vocab['tone']['null']])

        return phoneme_indices

    def tokenize(self, text: str) -> torch.Tensor:
        text = self.normalize_text(text)
        phoneme_indices = self.split_sentence_to_phonemes(text)
        return torch.tensor(phoneme_indices, dtype=torch.long)

    def detokenize_phonemes(self, phoneme_tensor: torch.Tensor) -> str:
        onset_vocab = self.vocab['onset']
        rhyme_vocab = self.vocab['rhyme']
        tone_vocab = self.vocab['tone']
        
        # Reverse vocab dictionaries for easy lookup
        rev_onset_vocab = {v: k for k, v in onset_vocab.items()}
        rev_rhyme_vocab = {v: k for k, v in rhyme_vocab.items()}
        rev_tone_vocab = {v: k for k, v in tone_vocab.items()}

        sentence = []
        for phoneme in phoneme_tensor:
            try:
                onset_index, rhyme_index, tone_index = phoneme.tolist()
                onset = rev_onset_vocab.get(onset_index, '')
                rhyme = rev_rhyme_vocab.get(rhyme_index, '')
                tone = rev_tone_vocab.get(tone_index, '')

                # Construct word from onset, rhyme, and tone
                if onset == '<_>':
                    onset = ' '
                if rhyme == 'null':
                    rhyme = ''
                if tone == 'null':
                    tone = ''

                word = onset + rhyme

                # Place tone mark correctly on the vowel
                if tone:
                    decomposed = unicodedata.normalize('NFD', word)
                    vowels = 'aeiouyăâêôơư'
                    vowel_found = False
                    for i, char in enumerate(decomposed):
                        if char in vowels:  # Apply tone mark to the first vowel found
                            decomposed = decomposed[:i + 1] + tone + decomposed[i + 1:]
                            vowel_found = True
                            break
                    if not vowel_found:
                        decomposed += tone  # If no vowel found, append the tone mark
                    word = unicodedata.normalize('NFC', decomposed)
                
                sentence.append(word)
            except Exception as e:
                print(f"Error processing phoneme '{phoneme}': {e}")
                sentence.append('')
        
        return ' '.join(sentence).strip()


# Example test
if __name__ == "__main__":
    vocab = {
        "onset": {
            " ": 0,
            "ng": 1,
            "d": 2,
            "kh": 3,
            "đ": 4,
            "t": 5,
            "q": 6,
            "b": 7,
            "nh": 8,
            "?": 9,
            "1": 10,
            "0": 11,
            "ch": 12,
            "n": 13,
            "null": 14,
            "ph": 15,
            "th": 16,
            "x": 17,
            "tr": 18,
            "gi": 19,
            "l": 20,
            "c": 21,
            "m": 22,
            "ý": 23,
            "-": 24,
            "á": 25,
            "s": 26,
            "v": 27,
            "o": 28,
            "a": 29,
            "g": 30,
            "e": 31,
            "i": 32,
            "ệ": 33,
            "p": 34,
            "y": 35,
            "h": 36,
            "9": 37,
            "7": 38,
            "5": 39,
            "2": 40,
            "3": 41,
            "4": 42,
            "ngh": 43,
            "r": 44,
            ",": 45,
            "8": 46,
            "6": 47,
            "k": 48,
            "/": 49,
            "w": 50,
            ".": 51,
            "u": 52,
            "\"": 53,
            "ắ": 54,
            "ề": 55,
            "f": 56,
            "z": 57,
            "â": 58,
            "ọ": 59,
            "gh": 60,
            "%": 61,
            "à": 62,
            ":": 63,
            "@": 64,
            "(": 65,
            ")": 66,
            "ứ": 67,
            "&": 68,
            "ỹ": 69,
            "ố": 70,
            "ỗ": 71,
            "ế": 72,
            "+": 73,
            ";": 74,
            "ó": 75,
            "!": 76,
            "ô": 77,
            "ạ": 78,
            "ì": 79,
            "ê": 80,
            "ư": 81,
            "ơ": 82,
            "ầ": 83,
            "ò": 84,
            "ồ": 85,
            "ỉ": 86,
            "'": 87,
            "ừ": 88,
            "ả": 89,
            "j": 90,
            "=": 91,
            "é": 92,
            "ờ": 93,
            "ụ": 94,
            "ậ": 95,
            "ờ": 96,
            "ở": 97,
            "ợ": 98,
            ">": 99,
            "ồ": 100,
            "ú": 101,
            "́": 102,
            "ở": 103,
            "ị": 104,
            "ộ": 105,
            "è": 106,
            "^": 107,
            "ã": 108,
            "ù": 109,
            "ă": 110,
            "ặ": 111,
            "ễ": 112,
            "í": 113,
            "ẩ": 114,
            "ấ": 115,
            "ủ": 116,
            "ủ": 117,
            "ĩ": 118,
            "ị": 119,
            "ỏ": 120,
            "ẻ": 121,
            "ũ": 122,
            "[": 123,
            "]": 124,
            "ể": 125,
            "ở": 126,
            "ỳ": 127,
            "`": 128,
            "õ": 129,
            "<_>": 130
        },
        "rhyme": {
            "null": 0,
            "ưoi": 1,
            "ân": 2,
            "ông": 3,
            "uưc": 4,
            "u": 5,
            "âp": 6,
            "ua": 7,
            "ao": 8,
            "iêu": 9,
            "ơ": 10,
            "ay": 11,
            "ương": 12,
            "anh": 13,
            "uân": 14,
            "ung": 15,
            "ây": 16,
            "ang": 17,
            "i": 18,
            "inh": 19,
            "ong": 20,
            "ư": 21,
            "ô": 22,
            "uy": 23,
            "an": 24,
            "o": 25,
            "au": 26,
            "ai": 27,
            "uyên": 28,
            "am": 29,
            "ê": 30,
            "iêp": 31,
            "uc": 32,
            "ên": 33,
            "a": 34,
            "oa": 35,
            "iên": 36,
            "oai": 37,
            "ênh": 38,
            "ưa": 39,
            "at": 40,
            "ăt": 41,
            "âm": 42,
            "ôi": 43,
            "up": 44,
            "âu": 45,
            "ơi": 46,
            "ăp": 47,
            "e": 48,
            "oăc": 49,
            "ăng": 50,
            "oanh": 51,
            "y": 52,
            "ia": 53,
            "ich": 54,
            "ơn": 55,
            "oc": 56,
            "uôc": 57,
            "ât": 58,
            "ep": 59,
            "ưc": 60,
            "uan": 61,
            "on": 62,
            "oe": 63,
            "ôc": 64,
            "uât": 65,
            "oi": 66,
            "ăc": 67,
            "uôn": 68,
            "yêu": 69,
            "êu": 70,
            "ưng": 71,
            "ăm": 72,
            "ăn": 73,
            "uang": 74,
            "om": 75,
            "ac": 76,
            "iêm": 77,
            "ơt": 78,
            "ui": 79,
            "ach": 80,
            "âng": 81,
            "ôt": 82,
            "êt": 83,
            "ap": 84,
            "im": 85,
            "in": 86,
            "it": 87,
            "uynh": 88,
            "ơm": 89,
            "ut": 90,
            "ưu": 91,
            "iêt": 92,
            "em": 93,
            "uôi": 94,
            "oach": 95,
            "eo": 96,
            "iêc": 97,
            "yên": 98,
            "oat": 99,
            "uây": 100,
            "un": 101,
            "êm": 102,
            "iêng": 103,
            "ôn": 104,
            "oan": 105,
            "ưu": 106,
            "en": 107,
            "op": 108,
            "uê": 109,
            "êch": 110,
            "oang": 111,
            "uyết": 112,
            "ôm": 113,
            "ip": 114,
            "îp": 115,
            "ưt": 116,
            "yết": 117,
            "et": 118,
            "ươn": 119,
            "uai": 120,
            "ưi": 121,
            "ueo": 122,
            "uưng": 123,
            "ôp": 124,
            "iu": 125,
            "uơm": 126,
            "ơp": 127,
            "ươp": 128,
            "uat": 129,
            "âc": 130,
            "uôt": 131,
            "uăng": 132,
            "ươt": 133,
            "uanh": 134,
            "um": 135,
            "uyt": 136,
            "oac": 137,
            "ươm": 138,
            "uay": 139,
            "ot": 140,
            "ue": 141,
            "oăn": 142,
            "ym": 143,
            "uet": 144,
            "uăc": 145,
            "eu": 146,
            "yêm": 147,
            "oay": 148,
            "uơi": 149,
            "uen": 150,
            "oong": 151,
            "eng": 152,
            "uao": 153,
            "uyu": 154,
            "yu": 155,
            "ou": 156,
            "oen": 157,
            "ư": 158
        },
        "tone": {
            "<`>": 0,
            "null": 1,
            "<.>": 2,
            "</>": 3,
            "<?>": 4,
            "<~>": 5
        }
    }
    tokenizer = VietnameseTokenizer(vocab)
    sentence = "con mèo test"
    phoneme_indices = tokenizer.split_sentence_to_phonemes(sentence)
    for idx, phoneme in enumerate(phoneme_indices):
        print(f"Phoneme {idx + 1}: {phoneme}")

    # Tokenize the sentence to tensor
    phoneme_tensor = tokenizer.tokenize(sentence)
    print("Phoneme tensor:", phoneme_tensor)

    # Detokenize the tensor back to sentence
    detokenized_sentence = tokenizer.detokenize_phonemes(phoneme_tensor)
    print("Detokenized sentence:", detokenized_sentence)