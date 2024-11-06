import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from word_processing import is_Vietnamese


class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_file, embedding_dim=128, dropout_rate=0.1):
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

    def forward(self, sentence):
        if isinstance(sentence, list):
            sentence = sentence[0]
        words = sentence.split()
        batch_size = len(words)

        # Prepare data structures
        details = []  # List to store detailed embeddings for each word

        # Process each word to determine if it's Vietnamese
        for idx, word in enumerate(words):
            is_vietnamese, phoneme_tuple = is_Vietnamese(word)
            if is_vietnamese:
                onset, rhyme, tone = phoneme_tuple

                # Ensure all phonemes are in the vocabulary (handle None case)
                onset_idx = self.phonemes['onset'].get(onset, self.phonemes['onset'].get('null', 0))
                rhyme_idx = self.phonemes['rhyme'].get(rhyme, self.phonemes['rhyme'].get('null', 0))
                tone_idx = self.phonemes['tone'].get(tone, self.phonemes['tone'].get('null', 0))

                # Get embeddings
                onset_emb = self.onset_embedding(torch.tensor([onset_idx], device=self.device))
                rhyme_emb = self.rhyme_embedding(torch.tensor([rhyme_idx], device=self.device))
                tone_emb = self.tone_embedding(torch.tensor([tone_idx], device=self.device))

                # Concatenate embeddings to form the word embedding
                word_emb = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)
                word_emb = self.dropout(word_emb)

                # Store detailed embeddings
                details.append({
                    'word': word,
                    'onset': onset,
                    'onset_embedding': onset_emb.squeeze(0),
                    'rhyme': rhyme,
                    'rhyme_embedding': rhyme_emb.squeeze(0),
                    'tone': tone,
                    'tone_embedding': tone_emb.squeeze(0),
                    'word_embedding': word_emb,
                    'source': 'vocab'
                })
            else:
                # Handle non-Vietnamese words by splitting each character and treating as separate phonemes
                chars = list(word)
                char_embeddings = []
                character_details = []

                # Process each character as if it were an onset, with no rhyme or tone
                for char in chars:
                    onset_idx = self.phonemes['onset'].get(char, self.phonemes['onset'].get('null', 0))
                    onset_emb = self.onset_embedding(torch.tensor([onset_idx], device=self.device))

                    # Use 'null' index for rhyme and tone
                    rhyme_emb = self.rhyme_embedding(torch.tensor([self.phonemes['rhyme']['null']], device=self.device))
                    tone_emb = self.tone_embedding(torch.tensor([self.phonemes['tone']['null']], device=self.device))

                    # Concatenate embeddings to form the character embedding (1x9)
                    char_emb = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)
                    char_emb = self.dropout(char_emb)

                    # Store character details
                    character_details.append({
                        'character': char,
                        'onset': char,
                        'rhyme': 'none',
                        'tone': 'none',
                        'onset_embedding': onset_emb.squeeze(0),
                        'rhyme_embedding': rhyme_emb.squeeze(0),
                        'tone_embedding': tone_emb.squeeze(0),
                        'embedding': char_emb.squeeze(0),
                        'source': 'non-vietnamese'
                    })

                    # Append the concatenated character embedding to char_embeddings
                    char_embeddings.append(char_emb)

                # Average or sum character embeddings to form the word embedding (1x9)
                word_emb = torch.mean(torch.stack(char_embeddings), dim=0)

                # Store detailed embeddings
                details.append({
                    'word': word,
                    'characters': character_details,
                    'word_embedding': word_emb,
                    'source': 'non-vietnamese'
                })

        # Handle spaces between words
        if len(words) > 1:
            for i in range(len(words) - 1):
                space_emb = self.onset_embedding(torch.tensor([self.phonemes['onset'][' ']], device=self.device))
                rhyme_emb = self.rhyme_embedding(torch.tensor([self.phonemes['rhyme']['null']], device=self.device))
                tone_emb = self.tone_embedding(torch.tensor([self.phonemes['tone']['null']], device=self.device))

                cat_space_emb = torch.cat((space_emb, rhyme_emb, tone_emb), dim=-1)

                details.insert(2 * i + 1, {
                    'word': '<_>',
                    'word_embedding': cat_space_emb,
                    'source': 'space'
                })

        return details, torch.cat([details[i]['word_embedding'] for i in range(len(details))])

# Example of using PhonemeEmbedding
vocab_file = 'vocab.json'
phoneme_embedding_model = PhonemeEmbedding(vocab_file, embedding_dim=128, dropout_rate=0.1)
sentence = 'ngu'
details, sent_embed = phoneme_embedding_model(sentence)
for detail in details:
    print(f"Word: {detail['word']}")
    if 'onset' in detail:
        print(f"  Onset: {detail['onset']} - Tensor: {detail['onset_embedding']}")
        print(f"  Rhyme: {detail['rhyme']} - Tensor: {detail['rhyme_embedding']}")
        print(f"  Tone: {detail['tone']} - Tensor: {detail['tone_embedding']}")
    if 'characters' in detail and detail['characters'] is not None:
        for char_detail in detail['characters']:
            print(f"  Character: {char_detail['character']}")
            print(f"    Onset: {char_detail['onset']} - Tensor: {char_detail['onset_embedding']}")
            print(f"    Rhyme: {char_detail['rhyme']} - Tensor: {char_detail['rhyme_embedding']}")
            print(f"    Tone: {char_detail['tone']} - Tensor: {char_detail['tone_embedding']}")
            print(f"    Embedding: {char_detail['embedding']} - Source: {char_detail['source']}")
    if 'word_embedding' in detail:
        print(f"  Word Embedding for '{detail['word']}': {detail['word_embedding']}.shape")
    print(f"  Source: {detail['source']}")
