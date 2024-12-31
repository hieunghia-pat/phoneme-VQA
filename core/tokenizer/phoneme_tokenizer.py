import torch

from decode.word_processing import is_Vietnamese, decompose_non_vietnamese_word, compose_word, split_phoneme

class PhonemeTokenizer:
    def __init__(self):
        
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.blank_token = None
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.blank_token]

        onsets = [
            'ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh', 
            'gi', 'gh', 'ch', 'q', 'đ', 'x', 'v', 't', 
            's', 'r', 'n', 'm', 'l', 'k', 'h', 'g', 'd', 
            'c', 'b'
        ]
        rhymes = [
            # a
            "a", "ac", "ach", "ai", 
            "am", "an", "ang", "anh", 
            "ao", "ap", "at", "ay", "au",
            # ă
            "ă", "ăc", "ăm", "ăn", "ăng", "ăp", "ăt",
            # â
            "â", "âc", "âm", "ân", "âng",
            "âp", "ât", "âu", "ây",
            # e
            "e", "ec", "em", "en",
            "eng", "eo", "ep", "et",
            # ê
            "ê", "êch", "êm", "ên", 
            "ênh", "êp", "êt", "êu",
            # i
            "i", "ia", "ich", "iêc", "iêm", "iên",
            "iêng", "iêp", "iêt", "iêu", "im", "in",
            "inh", "ip", "it", "iu",
            # o
            "o", "oa", "oac", "oach", "oai",
            "oam", "oan", "oang", "oanh",
            "oao", "oap", "oat", "oay",
            "oăc", "oăm", "oăn", "oăng",
            "oăt", "oc", "oe", "oen","oeo",
            "oet", "oi", "om", "on", "ong",
            "ooc", "oong", "op", "ot",
            # ô
            "ô", "ôc", "ôi",
            "ôm", "ôn", "ông",
            "ôp", "ôt",
            # ơ
            "ơ", "ơi", "ơm",
            "ơn", "ơp", "ơt",
            # u
            "u", "ua", "uân", "uâng", "uât",
            "uây", "uc", "uê", "uêch", "uênh",
            "ui", "um", "un", "ung", "uơ", "uôc",
            "uôi", "uôm", "uôn", "uông", "uôt",
            "up", "ut", "uy", "uya", "uych",
            "uyên", "uyêt", "uyn", "uynh",
            "uyp", "uyt", "uyu",
            "uach", "uai", "uan", "uang", "uanh", "uao", "uat", "uau", "uay",
            "uăc", "uăm", "uăn", "uăng", "uăp", "uăt", "uâc", "uât", "uoang",
            "ue", "uen", "ueo", "uet", "uên", "uêt", "uêu", "uơ", "uơi",
            
            # ư
            "ư", "ưa", "ưc", "ưi",
            "ưng", "ươc", "ươi",
            "ươm", "ươn", "ương",
            "ươp", "ươt", "ươu",
            "ưt", "ưu",
            # y
            "y", "yêm", "yên", 
            "yêng", "yêt", "yêu",
            # punctuations
            "?", ",", ".", "-","/", 
            "!", "@", "(", ")", ":", 
            "%", "\"", "*", "'", "+",
            "$",
            # numbers
            "0", "1", "2", "3", "4", 
            "5", "6", "7", "8", "9",
            # foreign letters
            "w", "f", "z", "j", "p"
        ]
        tones = ['<huyền>', '<sắc>', '<ngã>', '<hỏi>', '<nặng>']
        phonemes = self.special_tokens + onsets + rhymes + tones
        self.phoneme2idx = {
            phoneme: idx for idx, phoneme in enumerate(phonemes)
        }
        self.idx2phoneme = {idx: phoneme for phoneme, idx in self.phoneme2idx.items()}
        
        self.pad_idx = self.phoneme2idx[self.pad_token]
        self.bos_idx = self.phoneme2idx[self.bos_token]
        self.eos_idx = self.phoneme2idx[self.eos_token]
        self.blank_idx = self.phoneme2idx[self.blank_token]

    @property
    def size(self) -> int:
        return len(self.phoneme2idx)

    def encode(self, sentence: str, max_length: int) -> torch.Tensor:
        words = sentence.split()
        
        word_components = []
        word_indices = [] # mark the token belonging to a word
        word_index = 1
        for word in words:
            is_Vietnamese_word, components = is_Vietnamese(word)
            if is_Vietnamese_word:
                word_components.append(components)
                word_indices.append(word_index)
            else:
                characters = decompose_non_vietnamese_word(word)
                word_components.extend(characters)
                word_indices.extend([word_index]*len(characters))
            word_index += 1

        phoneme_script = []
        for word_component in word_components:
            onset, medial, nucleus, coda, tone = word_component
            rhyme = compose_word(None, medial, nucleus, coda, None)
            phoneme_script.append([
                self.phoneme2idx[onset] if onset else self.blank_idx, 
                self.phoneme2idx[rhyme] if rhyme else self.blank_idx, 
                self.phoneme2idx[tone] if tone else self.blank_idx
            ])
        
        bos_token = [self.bos_idx, self.blank_idx, self.blank_idx]
        eos_token = [self.eos_idx, self.blank_idx, self.blank_idx]
        phoneme_script = [bos_token] + phoneme_script + [eos_token]
        # index for bos token and eos token
        word_indices = [0] + word_indices + [max(word_indices)+1]

        if len(phoneme_script) < max_length:
            delta_length = max_length - len(phoneme_script)
            padding_values = [[self.pad_idx]*3] * delta_length
            phoneme_script.extend(padding_values)
            word_indices.extend([max(word_indices)+1] * delta_length)
        else:
            phoneme_script = phoneme_script[:max_length]
            word_indices = word_indices[:max_length]

        return phoneme_script, word_indices

    def batch_encode(self, sentences: list[str], max_length) -> torch.Tensor:
        sentences = [sentence.lower() for sentence in sentences]
        sentences = [self.encode(sentence, max_length) for sentence in sentences]

        return torch.tensor(sentences)
    
    def decode(self, tensor_sentence: torch.Tensor, word_phrases: torch.Tensor) -> str:
        '''
            tensorscript: (1, seq_len)
        '''
        words = tensor_sentence.squeeze(0).long().tolist()
        word_phrases = word_phrases.squeeze(0).long().tolist()
        sentence = []
        current_phrase = 0
        current_word = ""
        for word_components, phrase in zip(words, word_phrases):
            onset, rhyme, tone = [self.idx2phoneme[idx] for idx in word_components]
            if rhyme is not None:
                _, medial, nucleus, coda = split_phoneme((onset if onset else "") + rhyme)
            else:
                medial = None
                nucleus = None
                coda = None
            word = compose_word(onset, medial, nucleus, coda, tone)
            if current_phrase != phrase:
                # append the whole word into sentence
                sentence.append(current_word)
                # reset the current marks
                current_phrase = phrase
                current_word = word
            else:
                current_word += word

        sentence = [word for word in sentence if word not in self.special_tokens]
        sentence = " ".join(sentence)

        return sentence

    def batch_decode(self, phoneme_matrices: torch.Tensor, word_phrases: torch.Tensor) -> list[str]:
        return [self.decode(phoneme_matrix, word_phrase) for phoneme_matrix, word_phrase in zip(phoneme_matrices, word_phrases)]

    def __call__(self, sentences, max_length=30):
        if isinstance(sentences, str):
            return self.encode(sentences, max_length=max_length)
        elif isinstance(sentences, list):
            return self.batch_encode(sentences, max_length=max_length)

    def create_mask(self, phoneme_indices: list[list[int]]) -> list[int]:
        mask = []
        for phoneme in phoneme_indices:
            if (phoneme[0] == self.pad_idx and phoneme[1] == self.pad_idx and phoneme[2] == self.pad_idx):
                mask.append(1)
            else:
                mask.append(0)
        return mask
