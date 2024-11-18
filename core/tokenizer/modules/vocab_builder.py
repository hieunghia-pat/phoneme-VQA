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
    def __init__(self, annotation_paths: list[str]):
        self.annotation_paths = annotation_paths
        self.vocab = {'onset': {'none': 0, ' ': 1, '<_>': 2}, 'rhyme': {'none': 0}, 'tone': {'none': 0},'none':{'none': 0}}
        self.word_sources = {'onset': {}, 'rhyme': {}, 'tone': {}}
        self.text_sources = {'rhyme': {}}
        self.word_counts = self.create_vocab()

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
                                onset = onset.lower() if onset else 'none'
                                if onset not in self.vocab['onset']:
                                    self.vocab['onset'][onset] = len(self.vocab['onset'])
                                    self.word_sources['onset'][onset] = [word]
                                else:
                                    if onset not in self.word_sources['onset']:
                                        self.word_sources['onset'][onset] = [word]
                                    else:
                                        self.word_sources['onset'][onset].append(word)

                                # Add rhyme to vocab
                                rhyme = rhyme.lower() if rhyme else 'none'
                                if rhyme not in self.vocab['rhyme']:
                                    self.vocab['rhyme'][rhyme] = len(self.vocab['rhyme'])
                                    self.word_sources['rhyme'][rhyme] = [word]
                                    self.text_sources['rhyme'][rhyme] = [text]
                                else:
                                    if rhyme not in self.word_sources['rhyme']:
                                        self.word_sources['rhyme'][rhyme] = [word]
                                    else:
                                        self.word_sources['rhyme'][rhyme].append(word)

                                    if rhyme not in self.text_sources['rhyme']:
                                        self.text_sources['rhyme'][rhyme] = [text]
                                    else:
                                        self.text_sources['rhyme'][rhyme].append(text)

                                # Add tone to vocab
                                tone = tone.lower() if tone else 'none'
                                if tone not in self.vocab['tone']:
                                    self.vocab['tone'][tone] = len(self.vocab['tone'])
                                    self.word_sources['tone'][tone] = [word]
                                else:
                                    if tone not in self.word_sources['tone']:
                                        self.word_sources['tone'][tone] = [word]
                                    else:
                                        self.word_sources['tone'][tone].append(word)
                            else:
                                # Handle non-Vietnamese words by adding each character to onset
                                for char in word.lower():
                                    if char.islower() and char not in self.vocab['onset']:
                                        self.vocab['onset'][char] = len(self.vocab['onset'])
                                        self.word_sources['onset'][char] = [word]
                                    elif char in self.word_sources['onset'] and word not in self.word_sources['onset'][char]:
                                        self.word_sources['onset'][char].append(word)

                                # Check for missing characters in onset from string.printable
                                for printable_char in string.ascii_lowercase + string.digits + string.punctuation + ' ' + '<_>':
                                    if printable_char not in self.vocab['onset']:
                                        self.vocab['onset'][printable_char] = len(self.vocab['onset'])
                                        self.word_sources['onset'][printable_char] = []

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

# Load data from the given JSON files and create vocab
annotation_paths = ['openvivqa_dev_v2.json', 'openvivqa_test_v2.json', 'openvivqa_train_v2.json']
vocab_builder = VocabBuilder(annotation_paths)
vocab = vocab_builder.vocab
vocab_builder.check_vocab()

# Save the vocabulary to a file
vocab_builder.save_vocab('vocab.json')
