import torch
from torch import nn, Tensor
import math
import json

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):

        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)])

class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_file, embedding_dim=256, dropout_rate=0):
        super(PhonemeEmbedding, self).__init__()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.phonemes = json.load(f)

        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_embeddings()
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def init_embeddings(self):
        # Create embeddings for each type of phoneme: onset, rhyme, tone
        self.onset_embedding = nn.Embedding(len(self.phonemes['onset']), self.embedding_dim).to(self.device)
        self.rhyme_embedding = nn.Embedding(len(self.phonemes['rhyme']), self.embedding_dim).to(self.device)
        self.tone_embedding = nn.Embedding(len(self.phonemes['tone']), self.embedding_dim).to(self.device)


    def forward(self, phoneme_tensor):
        onset_indices = phoneme_tensor[:, :, 0]
        rhyme_indices = phoneme_tensor[:, :, 1]
        tone_indices = phoneme_tensor[:, :, 2]

        onset_emb = self.onset_embedding(onset_indices.to(self.device))
        rhyme_emb = self.rhyme_embedding(rhyme_indices.to(self.device))
        tone_emb = self.tone_embedding(tone_indices.to(self.device))

        # Kết hợp các embeddings để tạo thành embedding cho từ
        word_embeddings = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)

        # Áp dụng dropout
        word_embeddings = self.dropout(word_embeddings)

        # Trả về embedding cuối cùng
        return word_embeddings

class BaseDecoder(nn.Module):
    def __init__(self, 
                emb_size: int,
                num_layers: int,
                n_head: int,
                batch_first: bool=True,
                ):
        super(BaseDecoder, self).__init__()
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_size, nhead=n_head, batch_first = batch_first)
            ,num_layers=num_layers)
        
    def forward(self,
                tgt,
                memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,):

        return self.decoder(tgt=tgt,
                            memory=memory,
                            tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,)