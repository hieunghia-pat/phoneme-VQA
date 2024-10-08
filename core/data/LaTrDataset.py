from typing_extensions import override
from .base_dataset import BaseDataset
from logger.logger import get_logger
import torch
import os
import math
import numpy as np


log = get_logger(__name__)

class LaTrDataset(BaseDataset):
    def __init__(self, 
                qa_df,
                ocr_df,
                base_img_path,
                max_ocr,
                tokenizer,
                max_input_length = 180,
                max_output_length = 128,
                truncation=True,
                pad_token_box=[0, 0, 0, 0, 0, 0],
                eos_token_box=[0, 0, 1000, 1000, 1000, 1000]):
        super().__init__(qa_df, ocr_df, tokenizer, max_input_length, max_output_length, truncation)

        self.base_img_path = base_img_path
        self.max_ocr = max_ocr
        self.pad_token_box = pad_token_box
        self.eos_token_box = eos_token_box

    def __getitem__(self, index):
        return {
            'input_ids': self.data[index]['input_ids'],
            'src_attention_mask': self.data[index]['src_attention_mask'],
            'label_ids': self.data[index]['label_ids'],
            'label_attention_mask': self.data[index]['label_attention_mask'],
        }

    def __getitem__(self, index):
        
        img_path = os.path.join(self.base_img_path, self.data['filename'][index].split(".")[0]+'.npy')

        img = torch.from_numpy(np.load(open(img_path,"rb"), allow_pickle=True).tolist()['image'])

        return {
            'input_ids': torch.tensor([self.data['input_ids'][index]], dtype=torch.int64).squeeze(0),
            'coordinates': torch.tensor([self.data['coordinates'][index]], dtype=torch.float64).squeeze(0),
            'src_attention_mask': torch.tensor([self.data['src_attention_mask'][index]], dtype=torch.int64).squeeze(0),
            'label_ids': torch.tensor([self.data['label_ids'][index]], dtype=torch.int64).squeeze(0),
            'label_attention_mask': torch.tensor([self.data['label_attention_mask'][index]], dtype=torch.int64).squeeze(0),
            'pixel_values': img.squeeze(0),
            'tokenized_ocr': torch.tensor([self.data['tokenized_ocr'][index]], dtype=torch.int64).squeeze(0),
            'ocr_attention_mask': torch.tensor([self.data['ocr_attention_mask'][index]], dtype=torch.int64).squeeze(0),
        }

    @override
    def init_storage(self):
        self.feature = ["input_ids", 
                        "src_attention_mask", 
                        "label_ids", 
                        "label_attention_mask", 
                        "pixel_values",
                        "coordinates",
                        "tokenized_ocr",
                        "ocr_attention_mask"
                        ]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []
    
    
    def data_processing(self, dataframe):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            input_ids, tokenized_ocr, coordinates, attention_mask, ocr_attention_mask = self.create_features(dataframe['question'][i], dataframe['texts'][i], dataframe['bboxes'][i])

            answer_encoding = self.tokenizer("<pad>" + dataframe['answer'][i].strip(),
                                                padding='max_length',
                                                max_length = self.max_output_length,
                                                truncation = True)

            self.data['label_ids'].append(answer_encoding['input_ids'])
            self.data['label_attention_mask'].append(answer_encoding['attention_mask'])

            self.data['input_ids'].append(input_ids)
            self.data['tokenized_ocr'].append(tokenized_ocr)
            self.data['coordinates'].append(coordinates)
            self.data['src_attention_mask'].append(attention_mask)
            self.data['ocr_attention_mask'].append(ocr_attention_mask)


            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")


    def create_features(self, ques, ocr_texts, bounding_box):
        bounding_box = [
                    [bounding_box[i][0],
                     bounding_box[i][1],
                     bounding_box[i][2],
                     bounding_box[i][3],
                     bounding_box[i][2]-bounding_box[i][0],
                     bounding_box[i][3]-bounding_box[i][1]
                     ] for i in range(len(bounding_box))
                ]

        ques_encoding = self.tokenizer(ques, add_special_tokens=False)

        ques_ids = ques_encoding['input_ids']
        #ques_mask = ques_encoding['attention_mask']


        ocr_encoding = self.tokenizer(ocr_texts, is_split_into_words=True,
                         add_special_tokens=False)

        ocr_dist_ids = self.tokenizer(ocr_texts, is_split_into_words=False,
                         add_special_tokens=False).input_ids

        ocr_ids = ocr_encoding['input_ids']
        #ocr_mask = ocr_encoding['attention_mask']

        ocr_word_ids = []

        for i, e in enumerate(ocr_dist_ids):
            ocr_word_ids += [i]*len(e)

        bbox_according_to_ocr_ids = [bounding_box[i]
                                   for i in ocr_word_ids]

        max_input_length = len(ques_ids) + len(ocr_ids) + 3

        if max_input_length > self.max_input_length:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]
            
            tokenized_ocr = ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]

            coordinates = bbox_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.eos_token_box]

            attention_mask = [1]*len(input_ids)

            ocr_attention_mask = [1]*len(tokenized_ocr)
        else:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]
            
            tokenized_ocr = ocr_ids + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)

            coordinates = bbox_according_to_ocr_ids + [self.eos_token_box] + [self.pad_token_box]*(self.max_input_length - max_input_length)

            attention_mask = [1]*len(input_ids) 
            
            ocr_attention_mask = [1]*len(bbox_according_to_ocr_ids) + [0]*(self.max_input_length - max_input_length)


        return input_ids, tokenized_ocr, coordinates, attention_mask, ocr_attention_mask