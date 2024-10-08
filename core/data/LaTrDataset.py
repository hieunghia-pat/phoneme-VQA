from typing_extensions import override
from .base_dataset import BaseDataset
from logger.logger import get_logger
import torch
import os
import numpy as np
import pandas as pd


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
                transform = None,
                pad_token_box=[0, 0, 0, 0, 0, 0],
                eos_token_box=[0, 0, 1000, 1000, 1000, 1000]):
        super().__init__(qa_df, ocr_df, tokenizer, max_input_length, max_output_length, truncation)

        self.base_img_path = base_img_path
        self.max_ocr = max_ocr
        self.pad_token_box = pad_token_box
        self.eos_token_box = eos_token_box

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'bboxes', 'texts']], on='image_id', how='inner')

        self.data_processing(dataframe)

    def __getitem__(self, index):
        
        img_path = os.path.join(self.base_img_path, str(self.data['image_id'][index])+'.npy')

        img = torch.from_numpy(np.load(open(img_path,"rb"), allow_pickle=True).tolist()['image'])

        return {
            'input_ids': torch.tensor([self.data['input_ids'][index]], dtype=torch.int64).squeeze(0),
            'coordinates': torch.tensor([self.data['coordinates'][index]], dtype=torch.int64).squeeze(0),
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
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            input_ids, tokenized_ocr, coordinates, attention_mask, ocr_attention_mask = self.create_features(dataframe['question'][i], dataframe['texts'][i], dataframe['bboxes'][i])

            answer_encoding = self.tokenizer("<pad> " + dataframe['answer'][i].strip(),
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

        ques_encoding = self.tokenizer("<pad> " + ques.strip(),
                                        padding='max_length',
                                        max_length = self.max_input_length,
                                        truncation = True)

        
        ocr_encoding = self.tokenizer(ocr_texts, is_split_into_words=True,
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
        
        special_tokens_count = 1
        bbox_according_to_ocr_ids = [bounding_box[i]
                                   for i in ocr_word_ids[:(self.max_ocr - special_tokens_count)]]

        
        tokenized_ocr = ocr_ids[:len(bbox_according_to_ocr_ids)] + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_ocr - len(bbox_according_to_ocr_ids) - special_tokens_count)

        coordinates = bbox_according_to_ocr_ids + [self.eos_token_box] + [self.pad_token_box]*(self.max_ocr - len(bbox_according_to_ocr_ids) - special_tokens_count)

        ocr_attention_mask = [1]*len(bbox_according_to_ocr_ids) + [0]*(self.max_ocr - len(bbox_according_to_ocr_ids) - special_tokens_count)
        


        return ques_encoding['input_ids'], tokenized_ocr, coordinates, ques_encoding['attention_mask'], ocr_attention_mask