from typing_extensions import override
from .base_dataset import BaseDataset
from logger.logger import get_logger
import torch
import os
import numpy as np
import pandas as pd


log = get_logger(__name__)

class CustomizedSaLDataset(BaseDataset):
    def __init__(self, 
                qa_df,
                ocr_df,
                obj_df,
                tokenizer,
                decode_tokenizer,
                base_ocr_feature_path,
                base_obj_feature_path,
                ocr_hidden,
                obj_hidden,
                max_ocr_element = 50, # to limit input lengths
                max_ocr_length = 150, # to make lengths consistent (inluding context tokens)
                max_obj_element = 25, # to limit input lengths
                max_obj_length = 50, # to make lengths consistent
                max_input_length = 30,
                max_output_length = 128,
                truncation=True,
                transform = None,
                context_token = "<c>",
                pad_token_box=[0, 0, 0, 0],
                eos_token_box=[0.9999, 0.9999, 0.9999, 0.9999]):
        super().__init__(qa_df, ocr_df, tokenizer, max_input_length, max_output_length, truncation)

        self.decode_tokenizer = decode_tokenizer
        self.base_ocr_feature_path = base_ocr_feature_path
        self.base_obj_feature_path = base_obj_feature_path

        self.ocr_hidden = ocr_hidden
        self.max_ocr_length = max_ocr_length
        self.max_ocr_element = max_ocr_element
        self.obj_hidden = obj_hidden
        self.max_obj_element = max_obj_element
        self.max_obj_length = max_obj_length

        self.pad_token_box = pad_token_box
        self.eos_token_box = eos_token_box
        self.context_token = context_token
        self.context_token_id = self.tokenizer(self.context_token).input_ids[0]

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'bboxes', 'texts']], on='image_id', how='inner')
        dataframe = pd.merge(dataframe, obj_df[['image_id', 'obj_bboxes', 'obj_labels']], on='image_id', how='inner')

        self.data_processing(dataframe)

    def __getitem__(self, index):
        
        obj_path = os.path.join(self.base_obj_feature_path, str(self.data['image_id'][index])+'.npy')
        obj_f = np.load(open(obj_path,"rb"), allow_pickle=True).tolist()['region_features']
        
        ocr_path = os.path.join(self.base_ocr_feature_path, str(self.data['image_id'][index])+'.npy')
        ocr_f = np.load(open(ocr_path,"rb"), allow_pickle=True).tolist()


        obj_features_according_to_obj_ids = [obj_f[i]
                                   for i in self.data['obj_word_ids'][index][:(self.max_obj_length - 1)]]
        obj_features = torch.stack(obj_features_according_to_obj_ids\
             + [torch.zeros(self.obj_hidden)]*(self.max_obj_length - len(obj_features_according_to_obj_ids)))

        ocr_features_according_to_ocr_ids = [torch.from_numpy(np.concatenate([ocr_f['det_features'][i], ocr_f['rec_features'][i]], axis=-1))
                                   for i in self.data['ocr_word_ids'][index][:(self.max_ocr_length - 1)]]
        ocr_features = torch.stack(ocr_features_according_to_ocr_ids\
             + [torch.zeros(self.ocr_hidden)]*(self.max_ocr_length - len(ocr_features_according_to_ocr_ids)))


        label_ids = torch.tensor([self.data['label_ids'][index]], dtype=torch.int64).squeeze(0)
        label_attention_mask = self._create_padding_mask(label_ids, self.decode_tokenizer.pad_id)

        return {
            'input_ids': torch.tensor([self.data['input_ids'][index]], dtype=torch.int64).squeeze(0),
            'src_attention_mask': torch.tensor([self.data['src_attention_mask'][index]], dtype=torch.float).squeeze(0),
            'label_ids': label_ids,
            'label_attention_mask': label_attention_mask,
            'tokenized_ocr': torch.tensor([self.data['tokenized_ocr'][index]], dtype=torch.int64).squeeze(0),
            'ocr_attention_mask': torch.tensor([self.data['ocr_attention_mask'][index]], dtype=torch.float).squeeze(0),
            'ocr_coordinates': torch.tensor([self.data['ocr_coordinates'][index]], dtype=torch.float).squeeze(0),
            'obj_attention_mask': torch.tensor([self.data['obj_attention_mask'][index]], dtype=torch.float).squeeze(0),
            'obj_coordinates': torch.tensor([self.data['obj_coordinates'][index]], dtype=torch.float).squeeze(0),
            'tokenized_obj': torch.tensor([self.data['tokenized_obj'][index]], dtype=torch.int64).squeeze(0),
            'ocr_features': ocr_features,
            'obj_features': obj_features,
        }

    @override
    def init_storage(self):
        self.feature = ['input_ids',
                        'src_attention_mask',
                        'label_ids',
                        'label_attention_mask',
                        'tokenized_ocr',
                        'ocr_attention_mask',
                        'ocr_coordinates',
                        'obj_attention_mask',
                        'obj_coordinates',
                        'ocr_features',
                        'obj_features',
                        'tokenized_obj',
                        'ocr_word_ids',
                        'obj_word_ids'
                        ]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []
    
    
    def data_processing(self, dataframe):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            tokenized_ocr, ocr_coordinates, ocr_attention_mask, ocr_word_ids = self.create_ocr_properties(dataframe['texts'][i], dataframe['bboxes'][i])

            tokenized_obj, obj_coordinates, obj_attention_mask, obj_word_ids = self.create_obj_properties(dataframe['obj_labels'][i], dataframe['obj_bboxes'][i])

            ques_encoding = self.tokenizer("<pad> " + dataframe['question'][i].strip(),
                                        padding='max_length',
                                        max_length = self.max_input_length,
                                        truncation = True)
            
            answer_encoding = self.decode_tokenizer(dataframe['answer'][i].strip(),
                                                max_length = self.max_output_length,
                                                )
            
            self.data['input_ids'].append(ques_encoding['input_ids'])
            self.data['src_attention_mask'].append(ques_encoding['attention_mask'])
            self.data['label_ids'].append(answer_encoding)

            self.data['tokenized_ocr'].append(tokenized_ocr)
            self.data['ocr_coordinates'].append(ocr_coordinates)
            self.data['ocr_attention_mask'].append(ocr_attention_mask)
            self.data['ocr_word_ids'].append(ocr_word_ids)

            self.data['tokenized_obj'].append(tokenized_obj)
            self.data['obj_coordinates'].append(obj_coordinates)
            self.data['obj_attention_mask'].append(obj_attention_mask)
            self.data['obj_word_ids'].append(obj_word_ids)


            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")
    
    def _create_padding_mask(self, ids, pad_token_id):

        return ids == pad_token_id


    def create_ocr_properties(self, ocr_texts, bounding_box):
        ocr_texts = ocr_texts[:self.max_ocr_element]
        bounding_box = bounding_box[:self.max_ocr_element]
        
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
        TSS_ocr_ids = []

        for i, e in enumerate(ocr_dist_ids):
            TSS_ocr_ids += e + [self.context_token_id]
            ocr_word_ids += [i]*(len(e)+1)
            
        
        special_tokens_count = 1
        
        bbox_according_to_ocr_ids = [bounding_box[i]
                                   for i in ocr_word_ids[:(self.max_ocr_length - special_tokens_count)]]
        
        
        tokenized_ocr = TSS_ocr_ids[:len(bbox_according_to_ocr_ids)] + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_ocr_length - len(bbox_according_to_ocr_ids) - special_tokens_count)

        coordinates = bbox_according_to_ocr_ids + [self.eos_token_box] + [self.pad_token_box]*(self.max_ocr_length - len(bbox_according_to_ocr_ids) - special_tokens_count)

        ocr_attention_mask = [1]*(len(bbox_according_to_ocr_ids)+1) + [0]*(self.max_ocr_length - len(bbox_according_to_ocr_ids) - special_tokens_count)
        
        return tokenized_ocr, coordinates, ocr_attention_mask, ocr_word_ids

    
    def create_obj_properties(self, obj_labels, bounding_box):
        obj_texts = obj_labels[:self.max_obj_element]
        bounding_box = bounding_box[:self.max_obj_element]
        
        obj_encoding = self.tokenizer(obj_texts, is_split_into_words=True,
                         add_special_tokens=False)
        try:
            obj_dist_ids = self.tokenizer(obj_texts, is_split_into_words=False,
                            add_special_tokens=False).input_ids
            obj_ids = obj_encoding['input_ids']           
        except:
            obj_dist_ids = []
            obj_ids = []

        obj_word_ids = []
        

        for i, e in enumerate(obj_dist_ids):
            obj_word_ids += [i]*len(e)
            
        
        special_tokens_count = 1
        
        bbox_according_to_obj_ids = [bounding_box[i]
                                   for i in obj_word_ids[:(self.max_obj_length - special_tokens_count)]]
        
          
        tokenized_obj = obj_ids[:len(bbox_according_to_obj_ids)] + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_obj_length - len(bbox_according_to_obj_ids) - special_tokens_count)

        coordinates = bbox_according_to_obj_ids + [self.eos_token_box] + [self.pad_token_box]*(self.max_obj_length - len(bbox_according_to_obj_ids) - special_tokens_count)

        obj_attention_mask = [1]*(len(bbox_according_to_obj_ids)+1) + [0]*(self.max_obj_length - len(bbox_according_to_obj_ids) - special_tokens_count)
        
        return tokenized_obj, coordinates, obj_attention_mask, obj_word_ids
