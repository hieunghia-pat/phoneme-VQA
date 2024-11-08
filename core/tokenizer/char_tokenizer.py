import string
class CharTokenizer:
    VIETNAMESE_DIACRITIC_CHARACTERS = 'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    GENERAL_PRINTABLE = string.printable
    
    def __init__(self,
                pad_token = "<pad>",
                bos_token = "<bos>",
                eos_token = "<eos>",
                unk_token = "<unk>"):
        
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.special_tokens = [pad_token, bos_token, eos_token, unk_token]

        self.idx2str = list(self.VIETNAMESE_DIACRITIC_CHARACTERS) + list(self.GENERAL_PRINTABLE) + self.special_tokens
        
        self.str2idx = {
            j:i
            for i,j in enumerate(self.idx2str)
        }

        self.pad_id = self.str2idx[self.pad_token]
        self.bos_id = self.str2idx[self.bos_token]
        self.eos_id = self.str2idx[self.eos_token]
    
    def __call__(self, text, max_length = None, padding=True, add_special_tokens=True):
        if type(text)==list:
            return self.batch_encode(text, max_length, padding, add_special_tokens)

        return self.encode(text, max_length, padding, add_special_tokens)
    
    def __len__(self):
        return len(self.idx2str)
    
    def encode(self, text, max_length = None, padding=True, add_special_tokens = True):
        char_seq = list(text)
        length = len(char_seq) + 2

        for i in range(length-2):
            try:
                char_seq[i] = self.str2idx[char_seq[i]]
            except:
                char_seq[i] = self.str2idx[self.unk_token]

        #truncate if necessary
        if max_length is None:
            max_length = length

        if length > max_length:
            char_seq = char_seq[:max_length-2]
            length = max_length
        
        if add_special_tokens:
            return self.add_special_tokens(char_seq, length, max_length, padding)    

        return char_seq
    
    def add_special_tokens(self, encoding, length, max_len, padding):
        
        if padding:
            return [self.str2idx[self.bos_token]] + encoding + [self.str2idx[self.eos_token]] + [self.str2idx[self.pad_token]]*(max_len-length)

        return [self.str2idx[self.bos_token]] + encoding + [self.str2idx[self.eos_token]]
    
    def batch_encode(self, texts, max_length = None, padding = True, add_special_tokens = True):
        encodings = []

        for text in texts:
            encodings.append(self.encode(text, max_length, padding, add_special_tokens))
        return encodings
    
    def post_processing(self, out_ids):
        res = []
        for out in out_ids:
            try:
                res.append(out[1:out.index(self.str2idx[self.eos_token])])
            except:
                res.append(out)

        return res
    
    def decode(self, char_seq):
        char_seqs = self.post_processing([char_seq])

        return ["".join([self.idx2str[item] for item in char_seq if item not in self.special_tokens]) for char_seq in char_seqs]
    
    def batch_decode(self, char_seqs):

        char_seqs = self.post_processing(char_seqs)

        return ["".join([self.idx2str[item] for item in char_seq if item not in self.special_tokens]) for char_seq in char_seqs]