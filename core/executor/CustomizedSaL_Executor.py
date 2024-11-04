import os
import sys
import json
import torch
import math
import pandas as pd
from torch.utils.data import DataLoader
from typing_extensions import override

from logger.logger import get_logger
from .base_executor import Base_Executor

from core.model import CustomizedSaL, CustomizedSaL_config
from core.data import textlayout_ocr_adapt, textlayout_obj_adapt, CustomizedSaLDataset
from core.tokenizer import *

from timeit import default_timer as timer

import evaluation

from transformers import AutoTokenizer, AutoConfig
import itertools

log = get_logger(__name__)


class CustomizedSaL_Executor(Base_Executor):
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        super().__init__(config, mode, evaltype, predicttype)
        log.info("---Initializing Executor---")

    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE)
                ocr_attention_mask = batch['ocr_attention_mask'].to(self.config.DEVICE)
                tokenized_ocr = batch['tokenized_ocr'].to(self.config.DEVICE)
                ocr_coordinates = batch['ocr_coordinates'].to(self.config.DEVICE)
                ocr_features = batch['ocr_features'].to(self.config.DEVICE)
                tokenized_obj = batch['tokenized_obj'].to(self.config.DEVICE)
                obj_attention_mask = batch['obj_attention_mask'].to(self.config.DEVICE)
                obj_coordinates = batch['obj_coordinates'].to(self.config.DEVICE)
                obj_features = batch['obj_features'].to(self.config.DEVICE)

                pred = self.model.generate( input_ids,
                                            src_attention_mask,
                                            tokenized_ocr,
                                            ocr_attention_mask,
                                            ocr_coordinates,
                                            ocr_features,
                                            tokenized_obj,
                                            obj_attention_mask,
                                            obj_coordinates,
                                            obj_features,
                                            max_ocr=self.config.max_ocr_length,
                                            max_ques=self.config.max_q_length,
                                            max_length = max_length,
                                            isgreedy = self.config.isgreedy,
                                            num_beam = self.config.num_beam)

                decoded_preds += self.tokenizer.batch_decode(pred.tolist())

                log.info(f"|===| Inferring... {it+1} it |===|")

        return decoded_preds

    @override
    def _build_model(self):
        log.info(f"# Building model architecture ...")
        if self.config.MODEL_MOD_CONFIG_CLASS is not None:   
            self.model_config = self.build_class(self.config.MODEL_MOD_CONFIG_CLASS)().build(self.config, len(self.tokenizer))
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.backbone_name)

        self.model = self.build_class(self.config.MODEL_CLASS)(self.model_config, tgt_vocab_size=len(self.decode_tokenizer))
        self.model = self.model.to(self.config.DEVICE)
    
    
    def _create_data_utils(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)
        self.tokenizer.add_tokens([self.config.context_token])

        train_qa_df = pd.read_csv(self.config.qa_train_path)[["image_id", "question", "answer", "filename"]]
        val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        self.val_answer = list(val_qa_df["answer"])

        self._create_decode_tokenizer([train_qa_df, val_qa_df])

        ocr_df = textlayout_ocr_adapt(self.config.base_ocr_feature_path, h_scale=1, w_scale=1)
        obj_df = textlayout_obj_adapt(self.config.base_obj_feature_path, h_scale=1, w_scale=1)

        print("# Creating Datasets")
        
        self.train_data = CustomizedSaLDataset(   qa_df = train_qa_df,
                                        ocr_df = ocr_df,
                                        obj_df = obj_df,
                                        base_ocr_feature_path = self.config.base_ocr_feature_path,
                                        base_obj_feature_path = self.config.base_obj_feature_path,
                                        ocr_hidden = self.config.ocr_hidden,
                                        obj_hidden = self.config.obj_hidden,
                                        max_ocr_element = self.config.max_ocr_element,
                                        max_ocr_length = self.config.max_ocr_length,
                                        max_obj_element = self.config.max_obj_element,
                                        max_obj_length = self.config.max_obj_length,
                                        tokenizer = self.tokenizer,
                                        decode_tokenizer = self.decode_tokenizer,
                                        transform=None,
                                        max_input_length = self.config.max_q_length,
                                        max_output_length = self.config.max_a_length)

        self.val_data = CustomizedSaLDataset(     qa_df = val_qa_df,
                                        ocr_df = ocr_df,
                                        obj_df = obj_df,
                                        tokenizer = self.tokenizer,
                                        decode_tokenizer = self.decode_tokenizer,
                                        base_ocr_feature_path = self.config.base_ocr_feature_path,
                                        base_obj_feature_path = self.config.base_obj_feature_path,
                                        ocr_hidden = self.config.ocr_hidden,
                                        obj_hidden = self.config.obj_hidden,
                                        max_ocr_element = self.config.max_ocr_element,
                                        max_ocr_length = self.config.max_ocr_length,
                                        max_obj_element = self.config.max_obj_element,
                                        max_obj_length = self.config.max_obj_length,
                                        transform=None,
                                        max_input_length = self.config.max_q_length,
                                        max_output_length = self.config.max_a_length)

    def _init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)
        self.tokenizer.add_tokens([self.config.context_token])

        self.decode_tokenizer = self._create_decode_tokenizer()

        if self.mode == "eval":
            print("###Load eval data ...")
            val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.base_ocr_feature_path, h_scale=1, w_scale=1)
            obj_df = textlayout_obj_adapt(self.config.base_obj_feature_path, h_scale=1, w_scale=1)

            self.val_data = CustomizedSaLDataset(     qa_df = val_qa_df,
                                            ocr_df = ocr_df,
                                            obj_df = obj_df,
                                            tokenizer = self.tokenizer,
                                            decode_tokenizer = self.decode_tokenizer,
                                            base_ocr_feature_path = self.config.base_ocr_feature_path,
                                            base_obj_feature_path = self.config.base_obj_feature_path,
                                            ocr_hidden = self.config.ocr_hidden,
                                            obj_hidden = self.config.obj_hidden,
                                            max_ocr_element = self.config.max_ocr_element,
                                            max_ocr_length = self.config.max_ocr_length,
                                            max_obj_element = self.config.max_obj_element,
                                            max_obj_length = self.config.max_obj_length,
                                            transform=None,
                                            max_input_length = self.config.max_q_length,
                                            max_output_length = self.config.max_a_length)
            
            self.val_answer = list(val_qa_df["answer"])
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)
            self.valiter_length = math.ceil(len(self.val_data)/self.config.EVAL_BATCH_SIZE)
        elif self.mode == "predict":
            print("###Load predict data ...")
            predict_qa_df = pd.read_csv(self.config.qa_predict_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.base_ocr_feature_path, h_scale=1, w_scale=1)
            obj_df = textlayout_obj_adapt(self.config.base_obj_feature_path, h_scale=1, w_scale=1)

            self.predict_data = CustomizedSaLDataset(     qa_df = predict_qa_df,
                                                ocr_df = ocr_df,
                                                obj_df = obj_df,
                                                tokenizer = self.tokenizer,
                                                decode_tokenizer = self.decode_tokenizer,
                                                base_ocr_feature_path = self.config.base_ocr_feature_path,
                                                base_obj_feature_path = self.config.base_obj_feature_path,
                                                ocr_hidden = self.config.ocr_hidden,
                                                obj_hidden = self.config.obj_hidden,
                                                max_ocr_element = self.config.max_ocr_element,
                                                max_ocr_length = self.config.max_ocr_length,
                                                max_obj_element = self.config.max_obj_element,
                                                max_obj_length = self.config.max_obj_length,
                                                transform=None,
                                                max_input_length = self.config.max_q_length,
                                                max_output_length = self.config.max_a_length)
            
            if self.config.get_predict_score:
                self.predict_answer = list(predict_qa_df["answer"])
            else:
                self.predict_answer = None

            self.predictiter = DataLoader(dataset = self.predict_data, 
                                    batch_size=self.config.PREDICT_BATCH_SIZE)

    def _train_epoch(self, epoch):
        self.model.train()
        losses = 0
        
        for it, batch in enumerate(self.trainiter):
            label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
            labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)


            trg_input = labels[:, :-1]
            label_attention_mask = label_attention_mask[:, :-1]

            logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                label_ids = trg_input,
                                src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                label_attention_mask = label_attention_mask,
                                tokenized_ocr=batch['tokenized_ocr'].to(self.config.DEVICE),
                                ocr_attention_mask=batch['ocr_attention_mask'].to(self.config.DEVICE) ,
                                ocr_coordinates=batch['ocr_coordinates'].to(self.config.DEVICE),
                                ocr_features=batch['ocr_features'].to(self.config.DEVICE),
                                tokenized_obj=batch['tokenized_obj'].to(self.config.DEVICE),
                                obj_attention_mask=batch['obj_attention_mask'].to(self.config.DEVICE),
                                obj_coordinates=batch['obj_coordinates'].to(self.config.DEVICE),
                                obj_features=batch['obj_features'].to(self.config.DEVICE),
                                max_ocr=self.config.max_ocr_length,
                                max_ques=self.config.max_q_length,
                                )


            self.optim.zero_grad()

            trg_out = labels[:, 1:]

            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
            loss.backward()

            self.optim.step()

            self.scheduler.step()
            
            losses += loss.data.item()

            if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.trainiter_length:
                log.info(f"--TRAINING--|Epoch: {epoch}| Step: {it+1}/{self.trainiter_length} | Loss: {round(losses / (it + 1), 2)}")

        return losses / self.trainiter_length
    
    def _evaluate(self):
        self.model.eval()
        losses = 0
        
        with torch.no_grad():
            for it, batch in enumerate(self.valiter):
                label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)

                trg_input = labels[:, :-1]
                label_attention_mask = label_attention_mask[:, :-1]

                logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    label_ids = trg_input,
                                    src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    label_attention_mask = label_attention_mask,
                                    tokenized_ocr=batch['tokenized_ocr'].to(self.config.DEVICE),
                                    ocr_attention_mask=batch['ocr_attention_mask'].to(self.config.DEVICE) ,
                                    ocr_coordinates=batch['ocr_coordinates'].to(self.config.DEVICE),
                                    ocr_features=batch['ocr_features'].to(self.config.DEVICE),
                                    tokenized_obj=batch['tokenized_obj'].to(self.config.DEVICE),
                                    obj_attention_mask=batch['obj_attention_mask'].to(self.config.DEVICE),
                                    obj_coordinates=batch['obj_coordinates'].to(self.config.DEVICE),
                                    obj_features=batch['obj_features'].to(self.config.DEVICE),
                                    max_ocr=self.config.max_ocr_length,
                                    max_ques=self.config.max_q_length,)


                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                losses += loss.data.item()

                if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.valiter_length:
                    log.info(f"--VALIDATING--| Step: {it+1}/{self.valiter_length} | Loss: {round(losses / (it + 1), 2)}")


        return losses / self.valiter_length

    def _create_decode_tokenizer(self, frames=None):
        if "BPE" in self.config.DecodeTokenizer:
            if frames:
                data = self._prepare_bpe_frames(frames)
            
            self.decode_tokenizer = self.build_class(self.config.DecodeTokenizer)(data, 
                                                                                    self.config.bpe_step, 
                                                                                    self.vocab_save_path, 
                                                                                    self.config.max_vocab_size)

        else:
            self.decode_tokenizer = self.build_class(self.config.DecodeTokenizer)()

    def _prepare_bpe_frames(self, frames):
        data = []
        for f in frames:
            data += f["answer"].tolist()
        return data