import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import unicodedata
import numpy as np
from decode.word_processing import is_Vietnamese
from decode.vocab_builder import VocabBuilder
import os


class VietnameseTokenizer:
    def __init__(self, vocab_path: str = None, annotation_paths: list[str] = None):
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
        elif annotation_paths:
            # Sử dụng VocabBuilder để tạo từ điển
            vocab_builder = VocabBuilder(annotation_paths)
            self.vocab = vocab_builder.vocab
            # Lưu từ điển vào tệp vocab.json nếu không tồn tại
            if vocab_path:
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        else:
            raise ValueError("Cần cung cấp 'vocab_path' hoặc 'annotation_paths'.")

        # Đảm bảo ký tự khoảng trắng có trong từ điển onset
        if ' ' not in self.vocab['onset']:
            self.vocab['onset'][' '] = len(self.vocab['onset'])
        self.rev_onset_vocab = {v: k for k, v in self.vocab['onset'].items()}
        self.rev_rhyme_vocab = {v: k for k, v in self.vocab['rhyme'].items()}
        self.rev_tone_vocab = {v: k for k, v in self.vocab['tone'].items()}
        # Từ điển chứa các dấu đặc biệt, mã Unicode của dấu thanh và vị trí áp dụng
        self.diacritics = {
            '<`>': '\u0300',   # Dấu huyền
            '</>': '\u0301',   # Dấu sắc
            '<?>' : '\u0309',  # Dấu hỏi
            '<~>': '\u0303',   # Dấu ngã
            '<.>': '\u0323',   # Dấu nặng
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
            'ia': 'a',
            'iê': 'ê',
            'ie': 'e',
            'iu': 'u',
            'oa': 'a',
            'oe': 'e',
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
            'ye': 'e',
            'ya': 'a',
            'yo': 'o',
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
        }

    # Hàm xác định vị trí đặt dấu thanh trong âm tiết
    def find_tone_position(self,syllable):
        # Chuyển âm tiết sang dạng NFD để phân tách các ký tự tổ hợp
        syllable_nfd = unicodedata.normalize('NFD', syllable)
        vowels = 'aăâeêiioôơuưy'
        letters = []
        idx_map = []

        idx = 0
        char_idx = 0  # Chỉ số ký tự trong từ gốc
        while idx < len(syllable_nfd):
            c = syllable_nfd[idx]
            base = c
            combs = ''
            idx += 1
            # Thu thập các dấu kết hợp (diacritics)
            while idx < len(syllable_nfd) and unicodedata.combining(syllable_nfd[idx]):
                combs += syllable_nfd[idx]
                idx += 1
            # Kết hợp ký tự cơ bản với các dấu
            letter = unicodedata.normalize('NFC', base + combs)
            letters.append(letter)
            # Lưu vị trí của ký tự trong âm tiết gốc
            idx_map.append(char_idx)
            char_idx += len(letter)

        # Xử lý trường hợp đặc biệt với "qu"
        is_qu = False
        if letters[0].lower() == 'q' and len(letters) > 1 and letters[1].lower() == 'u':
            is_qu = True

        # Tạo danh sách các chỉ số nguyên âm trong âm tiết
        vowel_indices = []
        for i, letter in enumerate(letters):
            if letter.lower() in vowels:
                # Loại bỏ chữ "u" trong "qu" khỏi danh sách nguyên âm
                if is_qu and i == 1 and letters[i].lower() == 'u':
                    continue
                vowel_indices.append(i)

        # Tạo chuỗi các nguyên âm trong âm tiết
        vowel_seq = ''.join([letters[i].lower() for i in vowel_indices])

        # Tìm tổ hợp nguyên âm trong vowel_groups
        max_length = min(3, len(vowel_seq))
        for length in range(max_length, 0, -1):  # Kiểm tra các tổ hợp có độ dài từ 3 đến 1
            for start in range(len(vowel_seq) - length + 1):
                seq = vowel_seq[start:start+length]
                if seq in self.vowel_groups:
                    tone_vowel_char = self.vowel_groups[seq]
                    # Tìm vị trí của nguyên âm mang dấu trong tổ hợp
                    for idx in vowel_indices[start:start+length]:
                        if letters[idx].lower() == tone_vowel_char:
                            pos = idx_map[idx]
                            length = len(letters[idx])
                            return pos, length
                    # Nếu không tìm thấy nguyên âm chính, đặt vào nguyên âm đầu tiên của tổ hợp
                    idx = vowel_indices[start]
                    pos = idx_map[idx]
                    length = len(letters[idx])
                    return pos, length

        # Nếu không tìm thấy tổ hợp, áp dụng quy tắc chung
        priority_vowels = ['ă', 'â', 'ê', 'ô', 'ơ', 'ư']
        for vowel in priority_vowels:
            for idx in vowel_indices:
                if letters[idx].lower() == vowel:
                    pos = idx_map[idx]
                    length = len(letters[idx])
                    return pos, length

        if vowel_indices:
            # Nếu không có nguyên âm ưu tiên, đặt dấu vào nguyên âm cuối cùng
            idx = vowel_indices[-1]
            pos = idx_map[idx]
            length = len(letters[idx])
            return pos, length

        # Nếu không tìm thấy nguyên âm, trả về -1
        return -1, 0

    # Hàm loại bỏ dấu thanh hiện có trong âm tiết
    def remove_existing_tone(self,syllable):
        syllable_nfd = unicodedata.normalize('NFD', syllable)
        tone_marks = ['\u0300', '\u0301', '\u0303', '\u0309', '\u0323']
        new_syllable_nfd = ''.join([c for c in syllable_nfd if c not in tone_marks])
        return unicodedata.normalize('NFC', new_syllable_nfd)

    # Hàm áp dụng dấu thanh vào âm tiết
    def apply_tone(self,syllable, accent):
        # Loại bỏ dấu thanh hiện có
        syllable = self.remove_existing_tone(syllable)

        vowel_pos, vowel_len = self.find_tone_position(syllable)

        if vowel_pos == -1:
            # Không tìm thấy nguyên âm, không áp dụng dấu
            return syllable

        # Tách nguyên âm cần thêm dấu
        vowel = syllable[vowel_pos:vowel_pos+vowel_len]
        # Chuyển sang dạng NFD để xử lý dấu
        vowel_nfd = unicodedata.normalize('NFD', vowel)
        base_char = ''
        diacritics_chars = []
        for c in vowel_nfd:
            if not unicodedata.combining(c):
                base_char += c
            else:
                diacritics_chars.append(c)
        # Thêm dấu thanh mới
        diacritics_chars.append(accent)
        # Sắp xếp lại các dấu theo thứ tự chuẩn
        diacritic_order = ['\u031B', '\u0306', '\u0302', '\u0309', '\u0300', '\u0301', '\u0303', '\u0323']
        diacritics_chars.sort(key=lambda x: diacritic_order.index(x) if x in diacritic_order else 999)
        # Tạo nguyên âm mới với dấu thanh
        new_vowel_nfd = base_char + ''.join(diacritics_chars)
        # Chuyển về dạng NFC
        new_vowel = unicodedata.normalize('NFC', new_vowel_nfd)
        # Thay thế nguyên âm trong âm tiết
        syllable = syllable[:vowel_pos] + new_vowel + syllable[vowel_pos+vowel_len:]

        return syllable

    # Hàm xử lý văn bản, áp dụng dấu thanh dựa trên các dấu đặc biệt
    def process_text(self,text):
        # Sử dụng re để tách chuỗi mà vẫn giữ nguyên các khoảng trắng
        tokens = re.split(r'(\s+)', text)
        for idx, token in enumerate(tokens):
            if not token.strip():
                # Nếu token chỉ chứa khoảng trắng, bỏ qua
                continue
            # Xử lý các dấu đặc biệt trong token
            for marker in self.diacritics:
                while marker in token:
                    idx_marker = token.find(marker)
                    accent =self.diacritics[marker]
                    # Xóa dấu đặc biệt khỏi token
                    token = token[:idx_marker] + token[idx_marker+len(marker):]
                    if token == 'gii':
                        token = 'gi'
                    elif token == 'giin':
                        token = 'gin'
                    # Áp dụng dấu thanh vào token
                    token = self.apply_tone(token, accent)
            tokens[idx] = token
        return ''.join(tokens)
    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[\s]+', ' ', text)
        return text.strip()

    def encode(self, sentence: str, max_length: int = 30) -> list[list[int]]:
        # Sử dụng regex để tách câu thành các token, bao gồm cả khoảng trắng
        sentence = sentence.lower()
        tokens = re.findall(r'\S+|\s', sentence)
        phoneme_indices = []

        # Thêm token <bos> vào đầu câu
        bos_idx = self.vocab['onset'].get('<bos>', self.vocab['onset']['none'])
        phoneme_indices.append([bos_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])

        for token in tokens:
            if token.isspace():
                # Xử lý từng ký tự khoảng trắng
                for _ in token:
                    onset_idx = self.vocab['onset'].get('<_>', self.vocab['onset']['none'])
                    phoneme_indices.append([onset_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])
                continue  # Bỏ qua để xử lý token tiếp theo

            # Xử lý các ký tự đặc biệt
            if re.match(r'^[\W_]+$', token):
                for char in token:
                    onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['none'])
                    phoneme_indices.append([onset_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])
                continue

            if token == '<_>':
                phoneme_indices.append([self.vocab['onset']['<_>'], self.vocab['rhyme']['none'], self.vocab['tone']['none']])
            else:
                try:
                    is_vietnamese, components = is_Vietnamese(token)
                    if is_vietnamese:
                        onset, rhyme, tone = components
                        onset_idx = self.vocab['onset'].get(onset, self.vocab['onset']['none'])
                        rhyme_idx = self.vocab['rhyme'].get(rhyme, self.vocab['rhyme']['none'])
                        tone_idx = self.vocab['tone'].get(tone, self.vocab['tone']['none'])
                        phoneme_indices.append([onset_idx, rhyme_idx, tone_idx])
                    else:
                        # Đối với từ không phải tiếng Việt, xử lý từng ký tự
                        for char in token:
                            onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['none'])
                            phoneme_indices.append([onset_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])
                except Exception as e:
                    print(f"Lỗi khi xử lý từ '{token}': {e}")
                    phoneme_indices.append([self.vocab['onset']['none'], self.vocab['rhyme']['none'], self.vocab['tone']['none']])

        # Thêm token <eos> vào cuối câu
        eos_idx = self.vocab['onset'].get('<eos>', self.vocab['onset']['none'])
        phoneme_indices.append([eos_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])

        # Thêm các token '<pad>' để đủ độ dài max_length
        if len(phoneme_indices) < max_length:
            pad_length = max_length - len(phoneme_indices)
            onset_idx_pad = self.vocab['onset'].get('<pad>', self.vocab['onset']['none'])
            rhyme_idx_pad = self.vocab['rhyme'].get('<pad>', self.vocab['rhyme']['none'])
            tone_idx_pad = self.vocab['tone'].get('<pad>', self.vocab['tone']['none'])
            pad_indices = [[onset_idx_pad, rhyme_idx_pad, tone_idx_pad]] * pad_length
            phoneme_indices.extend(pad_indices)
        else:
            # Nếu chuỗi dài hơn max_length, cắt bớt
            phoneme_indices = phoneme_indices[:max_length]

        return phoneme_indices

    def encode_multiple(self, sentences: list[str], max_length: int = 30) -> list[list[list[int]]]:
        sentences = [sentence.lower() for sentence in sentences]
        return [self.encode(sentence, max_length) for sentence in sentences]

    def decode(self, phoneme_matrix: list[list[int]]) -> str:
        # Lấy các chỉ số của token '<pad>', '<bos>', và '<eos>' trong từ điển
        pad_onset_idx = self.vocab['onset'].get('<pad>', self.vocab['onset']['none'])
        bos_onset_idx = self.vocab['onset'].get('<bos>', self.vocab['onset']['none'])
        eos_onset_idx = self.vocab['onset'].get('<eos>', self.vocab['onset']['none'])

        sentence = []
        current_word = ""
        for phoneme in phoneme_matrix:
            try:
                onset_index, rhyme_index, tone_index = phoneme

                # Bỏ qua các token '<pad>', '<bos>', và '<eos>'
                if onset_index in {pad_onset_idx, bos_onset_idx, eos_onset_idx}:
                    continue

                onset = self.rev_onset_vocab.get(onset_index, '')
                rhyme = self.rev_rhyme_vocab.get(rhyme_index, '')
                tone = self.rev_tone_vocab.get(tone_index, '')

                # Xử lý 'none'
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

                if onset == ' ':
                    if current_word:
                        sentence.append(current_word)
                        current_word = ""
                    sentence.append(' ')
                    continue

                # Xây dựng từ từ onset và rhyme
                word = onset + rhyme

                # Áp dụng dấu thanh nếu cần
                if tone:
                    # Áp dụng dấu thanh vào từ
                    word = self.apply_tone(word, tone)

                current_word += word
            except Exception as e:
                print(f"Lỗi khi xử lý phoneme '{phoneme}': {e}")
                if current_word:
                    sentence.append(current_word)
                    current_word = ""

        if current_word:
            sentence.append(current_word)

        # Tái tạo lại câu
        final_sentence = ''.join(sentence)
        return self.process_text(final_sentence)

    def decode_multiple(self, phoneme_matrices: list[list[list[int]]]) -> list[str]:
        return [self.decode(phoneme_matrix) for phoneme_matrix in phoneme_matrices]
    def __call__(self, sentences):
        """
        Hàm này cho phép đối tượng VietnameseTokenizer được gọi như một hàm.
        Nếu đầu vào là một chuỗi, nó sẽ mã hóa và giải mã một câu.
        Nếu đầu vào là một danh sách các chuỗi, nó sẽ mã hóa và giải mã nhiều câu.
        """
        if isinstance(sentences, str):
            # Nếu chỉ là một câu, sử dụng encode
            encoded = self.encode(sentences)
            
            return encoded
        elif isinstance(sentences, list):
            # Nếu là danh sách các câu, sử dụng encode_multiple
            encoded = self.encode_multiple(sentences)
            
            return encoded
    def create_mask(self, phoneme_indices: list[list[int]]) -> list[int]:
        """
        Hàm này tạo một mask từ danh sách các phoneme_indices.
        Nếu phoneme là [3, 0, 0] (tương ứng token <pad>) thì gán 1, ngược lại gán 0.
        """
        # Lấy các chỉ số của token '<pad>' trong từ điển
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

if __name__ == "__main__":
    vocab_path = 'vocab.json'
    annotation_paths = ['openvivqa_dev_v2.json', 'openvivqa_test_v2.json', 'openvivqa_train_v2.json']

    # Khởi tạo VietnameseTokenizer, tự động tạo vocab nếu cần
    tokenizer = VietnameseTokenizer(vocab_path=vocab_path, annotation_paths=annotation_paths)

    sentences = ["N "]
    phoneme_indices_list = tokenizer.encode_multiple(sentences)

    for i, phoneme_indices in enumerate(phoneme_indices_list):
        print(f"Câu {i + 1}:")
        for idx, phoneme in enumerate(phoneme_indices):
            print(f"  Matrix {idx + 1}: {phoneme}")

    # Giải mã các ma trận về câu gốc
    phoneme_matrices = [phoneme_matrix for phoneme_matrix in phoneme_indices_list]
    detokenized_sentences = tokenizer.decode_multiple(phoneme_matrices)
    for i, detokenized_sentence in enumerate(detokenized_sentences):
        print(f"Câu giải mã {i + 1}: '{detokenized_sentence}'")

    # Kiểm tra hàm create_mask
    for i, phoneme_indices in enumerate(phoneme_indices_list):
        mask = tokenizer.create_mask(phoneme_indices)
        print(f"Mask cho câu {i + 1}: {mask}")
