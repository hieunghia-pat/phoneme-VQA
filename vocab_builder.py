import json
import os
import unicodedata
import re
import numpy as np
import torch
import torch.nn as nn
import string
from word_processing import is_Vietnamese

class VocabBuilder:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.annotation_paths = self.get_annotation_files()
        self.vocab = {'onset': {' ': 0}, 'rhyme': {'null': 0}, 'tone': {}, 'none': {'none': 0}}
        self.word_sources = {'onset': {}, 'rhyme': {}, 'tone': {}}
        self.text_sources = {'rhyme': {}}
        self.word_counts = self.create_vocab()

    def get_annotation_files(self) -> list[str]:
        # Get all JSON files in the dataset directory
        annotation_files = [
            os.path.join(self.dataset_dir, filename) for filename in os.listdir(self.dataset_dir)
            if filename.endswith('.json')
        ]
        return annotation_files

    def create_vocab(self) -> dict[str, int]:
        for annotation_path in self.annotation_paths:
            with open(annotation_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)

            annotations = json_data.get("annotations", [])

            for ann in annotations:
                # Collect words from questions and answers
                for field in ["question", "answers"]:
                    if field in ann:
                        text = ann[field] if isinstance(ann[field], str) else ann[field][0]
                        for word in text.split():
                            word = word.lower()

                            # Check if the word is Vietnamese
                            is_viet, (onset, rhyme, tone) = is_Vietnamese(word)
                            if is_viet:
                                # Add onset to vocab
                                if onset not in self.vocab['onset']:
                                    self.vocab['onset'][onset] = len(self.vocab['onset'])
                                    self.word_sources['onset'][onset] = [word]
                                else:
                                    self.word_sources['onset'][onset].append(word)

                                # Add rhyme to vocab
                                if rhyme not in self.vocab['rhyme']:
                                    self.vocab['rhyme'][rhyme] = len(self.vocab['rhyme'])
                                    self.word_sources['rhyme'][rhyme] = [word]
                                    self.text_sources['rhyme'][rhyme] = [text]
                                else:
                                    self.word_sources['rhyme'][rhyme].append(word)
                                    self.text_sources['rhyme'][rhyme].append(text)

                                # Add tone to vocab
                                if tone not in self.vocab['tone']:
                                    self.vocab['tone'][tone] = len(self.vocab['tone'])
                                    self.word_sources['tone'][tone] = [word]
                                else:
                                    self.word_sources['tone'][tone].append(word)
                            else:
                                # Handle non-Vietnamese words by adding each character to onset
                                for char in word:
                                    if char not in self.vocab['onset']:
                                        self.vocab['onset'][char] = len(self.vocab['onset'])
                                        self.word_sources['onset'][char] = [word]
                                    elif word not in self.word_sources['onset'][char]:
                                        self.word_sources['onset'][char].append(word)

        return self.vocab

    def check_vocab(self):
        # Print the vocabulary and its size
        print("Vocabulary Size:", {key: len(value) for key, value in self.vocab.items()})
        for category, vocab_dict in self.vocab.items():
            print(f"Category: {category}")
            for word, idx in vocab_dict.items():
                print(f"  {word}: {idx}")

    def save_vocab(self, output_path: str):
        # Save the vocabulary to a JSON file
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=4)

    def find_word_source(self, category: str, key: str):
        # Find the source words for a given key in the specified category
        if category in self.word_sources and key in self.word_sources[category]:
            print(f"Words that contributed to {category} '{key}': {self.word_sources[category][key]}")
            if category == 'rhyme' and key in self.text_sources['rhyme']:
                print(f"Original texts that contained rhyme '{key}': {self.text_sources['rhyme'][key]}")
        else:
            print(f"{category.capitalize()} '{key}' not found in vocabulary.")

# Load data from the given dataset directory and create vocab
dataset_dir = 'dataset'
vocab_builder = VocabBuilder(dataset_dir)
vocab = vocab_builder.vocab
vocab_builder.check_vocab()

# Save the vocabulary to a file
vocab_builder.save_vocab('vocab.json')
