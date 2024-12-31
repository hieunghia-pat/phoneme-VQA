import torch
from torch import nn

class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim:int = 256):
        super(PhonemeEmbedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)

    def forward(self, phoneme_tensor: torch.Tensor):
        onset_indices = phoneme_tensor[:, :, 0]
        rhyme_indices = phoneme_tensor[:, :, 1]
        tone_indices = phoneme_tensor[:, :, 2]

        onset_emb = self.embedding(onset_indices)
        rhyme_emb = self.embedding(rhyme_indices)
        tone_emb = self.embedding(tone_indices)

        word_embeddings = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)

        return word_embeddings