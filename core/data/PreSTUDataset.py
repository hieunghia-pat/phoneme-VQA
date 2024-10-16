from typing_extensions import override
from .base_dataset import BaseDataset
from logger.logger import get_logger
import torch
import os
import numpy as np
import pandas as pd


log = get_logger(__name__)

class PreSTUDataset(BaseDataset):
    def __init__(self, 
                qa_df,
                ocr_df,
                tokenizer,
                base_img_path,
                max_ocr_element = 50, # to limit input lengths
                max_ocr_length = 100, # to make lengths consistent
                max_input_length = 30, # max_input_length <=> max_ques_length
                max_output_length = 20,
                truncation=True,
                transform = None):
        super().__init__(qa_df, ocr_df, tokenizer, max_input_length, max_output_length, truncation)

        self.base_img_path = base_img_path
        self.max_ocr_length = max_ocr_length
        self.transform = transform
        self.truncation = truncation
        self.max_ocr_element = max_ocr_element

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'bboxes', 'texts']], on='image_id', how='inner')

        self.data_processing(dataframe)

    def __getitem__(self, index):
        
        img_path = os.path.join(self.base_img_path, str(self.data['image_id'][index])+'.npy')

        img = torch.from_numpy(np.load(open(img_path,"rb"), allow_pickle=True).tolist()['image'])

        return {
            'input_ids': torch.tensor([self.data['input_ids'][index]], dtype=torch.int64).squeeze(0),
            'src_attention_mask': torch.tensor([self.data['src_attention_mask'][index]], dtype=torch.int64).squeeze(0),
            'label_ids': torch.tensor([self.data['label_ids'][index]], dtype=torch.int64).squeeze(0),
            'label_attention_mask': torch.tensor([self.data['label_attention_mask'][index]], dtype=torch.int64).squeeze(0),
            'pixel_values': img.squeeze(0),
        }

    @override
    def init_storage(self):
        self.feature = ["input_ids", 
                        "src_attention_mask", 
                        "label_ids", 
                        "label_attention_mask", 
                        "pixel_values",
                        ]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []
    
    
    def data_processing(self, dataframe):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            input_ids, src_attention_mask = self.create_properties(dataframe['question'][i], dataframe['texts'][i])

            answer_encoding = self.tokenizer("<pad> " + dataframe['answer'][i].strip(),
                                                padding='max_length',
                                                max_length = self.max_output_length,
                                                truncation = self.truncation)

            self.data['label_ids'].append(answer_encoding['input_ids'])
            self.data['label_attention_mask'].append(answer_encoding['attention_mask'])

            self.data['input_ids'].append(input_ids)
            self.data['src_attention_mask'].append(src_attention_mask)


            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")


    def create_features(self, ques, ocr_texts):
        ocr_texts = ocr_texts[:self.max_ocr_element]

        ques_special_tokens_count = 2 
        ocr_special_tokens_count = 1

        ques_encoding = self.tokenizer(ques.strip(),
                                        max_length = self.max_input_length - ques_special_tokens_count,
                                        truncation = self.truncation,
                                        add_special_tokens=False)
        
        ques_ids = ques_encoding['input_ids']

        
        ocr_encoding = self.tokenizer(ocr_texts, 
                                    is_split_into_words=True,
                                    add_special_tokens=False)
        
        try:
            ocr_dist_ids = self.tokenizer(ocr_texts, is_split_into_words=False,
                            add_special_tokens=False).input_ids
            ocr_ids = ocr_encoding['input_ids']           
        except:
            ocr_dist_ids = []
            ocr_ids = []

        ocr_word_ids = []

        for i, e in enumerate(ocr_dist_ids):
            ocr_word_ids += [i]*len(e)
        
        
        tokenized_ocr_ids = ocr_ids[:(self.max_ocr_length - ocr_special_tokens_count)]
        
        valid_length = ocr_special_tokens_count + ques_special_tokens_count + len(ques_ids) + len(tokenized_ocr_ids)

        input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]\
             + tokenized_ocr_ids + [self.tokenizer.eos_token_id]\
             + [self.tokenizer.pad_token_id]*(self.max_input_length + self.max_ocr_length - valid_length)

        src_attention_mask = [1]*(valid_length) + [0]*(self.max_input_length + self.max_ocr_length - valid_length) 

        return input_ids, src_attention_mask