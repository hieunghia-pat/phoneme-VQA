import torch
import json
from multi_token import MultiTokensPredictor
from word_processing import is_Vietnamese
# Define configuration parameters
vocab_file = 'vocab.json'  # Path to the vocabulary file
model_dim = 384  # Dimension of the model
embedding_dim = 128  # Dimension of the embedding
dropout_rate = 0.1  # Dropout rate

# Load the model
print("Loading MultiTokensPredictor model...")
multi_tokens_predictor = MultiTokensPredictor(vocab_file, model_dim, embedding_dim, dropout_rate)
multi_tokens_predictor.eval()  # Set model to evaluation mode

# Example input sentence
sentence = "test"

# Make predictions
print("Making predictions...")
predictions = multi_tokens_predictor(sentence)

# Print predictions
for i, prediction in enumerate(predictions):
    is_vietnamese, _ = is_Vietnamese(sentence.split()[i])
    if is_vietnamese:
        onset_idx = torch.argmax(prediction['onset']).item()
        rhyme_idx = torch.argmax(prediction['rhyme']).item()
        tone_idx = torch.argmax(prediction['tone']).item()
        onset = list(multi_tokens_predictor.phoneme_embedding.phonemes['onset'].keys())[onset_idx]
        rhyme = list(multi_tokens_predictor.phoneme_embedding.phonemes['rhyme'].keys())[rhyme_idx]
        tone = list(multi_tokens_predictor.phoneme_embedding.phonemes['tone'].keys())[tone_idx]
    else:
        onset_idx = torch.argmax(prediction['onset']).item()
        rhyme_idx = torch.argmax(prediction['none']).item()
        tone_idx = torch.argmax(prediction['none']).item()
        onset = list(multi_tokens_predictor.phoneme_embedding.phonemes['onset'].keys())[onset_idx]
        rhyme = list(multi_tokens_predictor.phoneme_embedding.phonemes['none'].keys())[rhyme_idx]
        tone = list(multi_tokens_predictor.phoneme_embedding.phonemes['none'].keys())[tone_idx]
    print(f"Character {i + 1}: onset: {onset}, rhyme: {rhyme}, tone: {tone}")

print("Execution finished.")