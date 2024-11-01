from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

def get_training_corpus(data, batch):
    data = [i for i in data]
    for i in range(0, len(data), batch):
        yield data[i : i + batch]

def create_BPEtokenizer(vocab_size, data, step, special_tokens, unk_token):
    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, unk_token=unk_token)
    tokenizer.train_from_iterator(get_training_corpus(data, step), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer

class BPE_Tokenizer:
    def __init__(self,
                data,
                step,
                max_vocab_size = 5000,
                pad_token = "<pad>",
                bos_token = "<bos>",
                eos_token = "<eos>",
                unk_token = "<unk>"):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
    
        self.special_tokens = [pad_token, bos_token, eos_token, unk_token]

        self.tokenizer = create_BPEtokenizer(max_vocab_size, data, step, self.special_tokens, self.unk_token)
    
    def __len__(self):
        return len(self.tokenizer.get_vocab())

    
