import torch
from torch import nn, Tensor
import math
from embedding import PhonemeEmbedding 

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

# Thay thế TokenEmbedding bằng PhonemeEmbedding
class VN_Embedding(nn.Module):
    def __init__(self,
                 vocab_file: str,
                 embedding_dim: int,
                 dropout_rate: float = 0.1):
        super(VN_Embedding, self).__init__()
        self.phoneme_embedding = PhonemeEmbedding(vocab_file, embedding_dim, dropout_rate)

    def forward(self, phoneme_matrices):
        # Lấy sentence_embeddings từ PhonemeEmbedding
        sentence_embeddings = self.phoneme_embedding(phoneme_matrices)
        # Chúng ta có thể cần nhân với căn bậc hai của embedding_dim nếu cần thiết
        return sentence_embeddings 

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