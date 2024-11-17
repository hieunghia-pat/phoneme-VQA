from word_processing import is_Vietnamese
import json
import torch
import torch.nn as nn
from vietnamese_tokenizer import VietnameseTokenizer

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
        """
        phoneme_tensor: Tensor có kích thước (batch_size, seq_len, 3)
        """
        # Tách tensor thành các chỉ số onset, rhyme, tone
        onset_indices = phoneme_tensor[:, :, 0]
        rhyme_indices = phoneme_tensor[:, :, 1]
        tone_indices = phoneme_tensor[:, :, 2]

        # Lấy embeddings cho từng loại âm vị
        onset_emb = self.onset_embedding(onset_indices.to(self.device))
        rhyme_emb = self.rhyme_embedding(rhyme_indices.to(self.device))
        tone_emb = self.tone_embedding(tone_indices.to(self.device))

        # Kết hợp các embeddings để tạo thành embedding cho từ
        word_embeddings = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)

        # Áp dụng dropout
        word_embeddings = self.dropout(word_embeddings)

        # Trả về embedding cuối cùng
        return word_embeddings


        
    
 

            
