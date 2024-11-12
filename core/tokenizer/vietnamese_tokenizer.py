import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import unicodedata
import numpy as np
from word_processing import is_Vietnamese
# Import other components from the project

class VietnameseTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

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

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)  # Chuẩn hóa về dạng NFC
        text = re.sub(r'[\s]+', ' ', text)  # Thay thế nhiều khoảng trắng bằng một khoảng trắng duy nhất
        return text.strip()

    # Hàm xác định vị trí đặt dấu thanh trong âm tiết
    def find_tone_position(self, syllable):
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
    def remove_existing_tone(self, syllable):
        syllable_nfd = unicodedata.normalize('NFD', syllable)
        tone_marks = ['\u0300', '\u0301', '\u0303', '\u0309', '\u0323']
        new_syllable_nfd = ''.join([c for c in syllable_nfd if c not in tone_marks])
        return unicodedata.normalize('NFC', new_syllable_nfd)

    # Hàm áp dụng dấu thanh vào âm tiết
    def apply_tone(self, syllable, accent):
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
    def process_text(self, text):
        # Tách từ trong câu
        words = text.split()
        for idx_w, word in enumerate(words):
            # Xử lý các dấu đặc biệt trong từ
            for marker in self.diacritics:
                while marker in word:
                    idx = word.find(marker)
                    accent = self.diacritics[marker]
                    # Xóa dấu đặc biệt khỏi từ
                    word = word[:idx] + word[idx+len(marker):]
                    if word == 'gii':
                        word = 'gi'
                    elif word == 'giin':
                        word = 'gin'
                    # Áp dụng dấu thanh vào từ
                    word = self.apply_tone(word, accent)
            words[idx_w] = word
        return ' '.join(words)

    def encode(self, sentence: str) -> list[list[int]]:
        sentence = self.process_text(sentence)  # Process text to apply diacritics
        words = sentence.split()
        phoneme_indices = []

        for idx, word in enumerate(words):
            # Xử lý dấu câu và ký tự đặc biệt riêng biệt
            if re.match(r'^[\W_]+$', word):
                for char in word:
                    onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['none'])
                    phoneme_indices.append([onset_idx, self.vocab['rhyme']['none'], self.vocab['tone']['none']])
                continue

            if word == '<_>':
                phoneme_indices.append([self.vocab['onset']['none'], self.vocab['rhyme']['none'], self.vocab['tone']['none']])
            else:
                try:
                    is_vietnamese_word, components = is_Vietnamese(word)
                    if is_vietnamese_word:
                        onset, rhyme, tone = components
                        onset_idx = self.vocab['onset'].get(onset, self.vocab['onset']['none'])
                        rhyme_idx = self.vocab['rhyme'].get(rhyme, self.vocab['rhyme']['none'])
                        tone_idx = self.vocab['tone'].get(tone, self.vocab['tone']['none'])
                        phoneme_indices.append([onset_idx, rhyme_idx, tone_idx])
                    else:
                        # Đối với từ không phải tiếng Việt, mỗi ký tự được xem như là onset, với rhyme và tone là 'none'
                        for char in word:
                            onset_idx = self.vocab['onset'].get(char, self.vocab['onset']['none'])
                            rhyme_idx = self.vocab['rhyme']['none']
                            tone_idx = self.vocab['tone']['none']
                            phoneme_indices.append([onset_idx, rhyme_idx, tone_idx])
                except Exception as e:
                    print(f"Lỗi khi xử lý từ '{word}': {e}")
                    phoneme_indices.append([self.vocab['onset']['none'], self.vocab['rhyme']['none'], self.vocab['tone']['none']])

            # Thêm khoảng trắng giữa các từ nếu không phải từ cuối
            if idx < len(words) - 1:
                phoneme_indices.append([self.vocab['onset']['<_>'], self.vocab['rhyme']['none'], self.vocab['tone']['none']])

        return phoneme_indices

    def encode_multiple(self, sentences: list[str]) -> list[list[list[int]]]:
        return [self.encode(sentence) for sentence in sentences]

    def decode(self, phoneme_tensor: torch.Tensor) -> str:
        onset_vocab = self.vocab['onset']
        rhyme_vocab = self.vocab['rhyme']
        tone_vocab = self.vocab['tone']

        # Đảo ngược từ điển vocab để tra cứu dễ dàng
        rev_onset_vocab = {v: k for k, v in onset_vocab.items()}
        rev_rhyme_vocab = {v: k for k, v in rhyme_vocab.items()}
        rev_tone_vocab = {v: k for k, v in tone_vocab.items()}

        sentence = []
        current_word = ""
        for phoneme in phoneme_tensor:
            try:
                onset_index, rhyme_index, tone_index = phoneme.tolist()
                onset = rev_onset_vocab.get(onset_index, '')
                rhyme = rev_rhyme_vocab.get(rhyme_index, '')
                tone = rev_tone_vocab.get(tone_index, '')

                # Kiểm tra và xử lý giá trị 'none'
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
                    continue

                word = onset + rhyme

                # Đặt dấu thanh đúng vị trí trên nguyên âm
                if tone:
                    word = self.apply_tone(word, tone)

                current_word += word
            except Exception as e:
                print(f"Lỗi khi xử lý phoneme '{phoneme}': {e}")
                if current_word:
                    sentence.append(current_word)
                    current_word = ""

        if current_word:
            sentence.append(current_word)

        return self.process_text(' '.join(sentence).strip())

    def decode_multiple(self, phoneme_tensors: list[torch.Tensor]) -> list[str]:
        return [self.decode(phoneme_tensor) for phoneme_tensor in phoneme_tensors]

    def __call__(self, sentences):
        """
        Process input sentences by encoding and then decoding them.
        If 'sentences' is a list, it uses encode_multiple and decode_multiple.
        If 'sentences' is a single string, it uses encode and decode.
        Returns a tuple of (encoded, decoded) results.
        """
        if isinstance(sentences, list):
            # If input is a list of sentences, use encode_multiple and decode_multiple
            encoded = self.encode_multiple(sentences)
            # Convert each list of phoneme indices to a torch.Tensor for decoding
            decoded = self.decode_multiple([torch.tensor(e) for e in encoded])
        else:
            # If input is a single sentence, use encode and decode
            encoded = self.encode(sentences)
            decoded = self.decode(torch.tensor(encoded))
        return encoded, decoded
