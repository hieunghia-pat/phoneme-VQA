import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import unicodedata
import numpy as np

# Import other components from the project
from word_processing import is_Vietnamese
from embedding import PhonemeEmbedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions (batch_size, seq_len, d_model), but got {x.size()}")

        batch_size, seq_len, d_model = x.size()

        if d_model != self.d_model:
            raise ValueError(f"Input d_model ({d_model}) does not match the expected d_model ({self.d_model})")

        pe = self.pe[:, :seq_len, :].expand(batch_size, seq_len, d_model).to(x.device)  # (batch_size, seq_len, d_model)
        x = x + pe
        return x

class MultiTokensPredictor(nn.Module):
    def __init__(self, vocab_file, model_dim, embedding_dim=128, dropout_rate=0.1):
        super(MultiTokensPredictor, self).__init__()
        self.model_dim = model_dim

        # Phoneme Embedding Layer
        self.phoneme_embedding = PhonemeEmbedding(vocab_file, embedding_dim, dropout_rate)

        # Positional Encoding for Transformer
        self.positional_encoding = PositionalEncoding(model_dim)

        # Transformer Layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=8, dim_feedforward=2048, dropout=dropout_rate),
            num_layers=6
        )

        # Output heads for each phoneme component
        self.output_heads = nn.ModuleDict({
            key: nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, len(phoneme_list))
            ) for key, phoneme_list in json.load(open(vocab_file, 'r', encoding='utf-8')).items()
        })

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sentence):
        if isinstance(sentence, list):
            sentence = sentence[0]
        words = sentence.split()
        embeddings_list = []

        # Get phoneme embeddings for each word
        for word in words:
            is_vietnamese, phoneme_tuple = is_Vietnamese(word)
            if is_vietnamese:
                onset, rhyme, tone = phoneme_tuple
                onset_emb = self.phoneme_embedding.onset_embedding(torch.tensor([self.phoneme_embedding.phonemes['onset'].get(onset, 0)], device=self.phoneme_embedding.device))
                rhyme_emb = self.phoneme_embedding.rhyme_embedding(torch.tensor([self.phoneme_embedding.phonemes['rhyme'].get(rhyme, 0)], device=self.phoneme_embedding.device))
                tone_emb = self.phoneme_embedding.tone_embedding(torch.tensor([self.phoneme_embedding.phonemes['tone'].get(tone, 0)], device=self.phoneme_embedding.device))
                word_emb = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)
                word_emb = self.dropout(word_emb)
                embeddings_list.append(word_emb.squeeze(0))
            else:
                # Handle non-Vietnamese words by treating each character as a separate phoneme prediction
                for char in word:
                    onset_emb = self.phoneme_embedding.onset_embedding(torch.tensor([self.phoneme_embedding.phonemes['onset'].get(char, 0)], device=self.phoneme_embedding.device))
                    rhyme_emb = self.phoneme_embedding.rhyme_embedding(torch.tensor([self.phoneme_embedding.phonemes['none'].get('none', 0)], device=self.phoneme_embedding.device))
                    tone_emb = self.phoneme_embedding.tone_embedding(torch.tensor([self.phoneme_embedding.phonemes['none'].get('none', 0)], device=self.phoneme_embedding.device))
                    char_emb = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)
                    char_emb = self.dropout(char_emb)
                    embeddings_list.append(char_emb.squeeze(0))

        # Stack embeddings for all words/characters in the sentence
        embeddings = torch.stack(embeddings_list).unsqueeze(0)  # (1, num_tokens, embedding_dim)

        # Add positional encoding
        embedded_vectors = self.positional_encoding(embeddings)  # (batch_size=1, seq_len=num_tokens, d_model)

        # Transformer expects (seq_len, batch_size, feature_dim)
        embedded_vectors = embedded_vectors.permute(1, 0, 2)  # (seq_len, batch_size, d_model)

        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(embedded_vectors)  # (seq_len, batch_size, d_model)

        # Apply mean pooling over sequence length
        pooled_output = transformer_output.permute(1, 0, 2).squeeze(0)  # (num_tokens, d_model)

        # Predict phoneme components using independent heads for each token
        predictions_list = []
        for i, word in enumerate(words):
            is_vietnamese, _ = is_Vietnamese(word)
            if is_vietnamese:
                predictions = {}
                for key in self.output_heads.keys():
                    head_output = self.output_heads[key](pooled_output[i].unsqueeze(0))
                    predictions[key] = F.log_softmax(head_output, dim=-1).squeeze(0)
                predictions_list.append(predictions)
            else:
                for j, char in enumerate(word):
                    predictions = {}
                    onset = char
                    for key in self.output_heads.keys():
                        if key == 'onset':
                            idx = self.phoneme_embedding.phonemes['onset'].get(onset, 0)
                        else:
                            idx = self.phoneme_embedding.phonemes['none'].get('none', 0)
                        head_output = self.output_heads[key](pooled_output[i + j].unsqueeze(0))
                        predictions[key] = F.log_softmax(head_output, dim=-1).squeeze(0)
                    predictions_list.append(predictions)

        return predictions_list
