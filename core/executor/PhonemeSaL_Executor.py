import os
import torch
import math
import pandas as pd
from torch.utils.data import DataLoader
from typing_extensions import override
from tqdm import tqdm

from logger.logger import get_logger
from .base_executor import Base_Executor

from core.data import textlayout_ocr_adapt, textlayout_obj_adapt, PhonemeSaLDataset
from core.tokenizer import PhonemeTokenizer

from transformers import AutoTokenizer, AutoConfig

log = get_logger(__name__)

class PhonemeSaL_Executor(Base_Executor):
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        super().__init__(config, mode, evaltype, predicttype)
        log.info("---Initializing Executor---")

    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        log.info("Inferring ...")

        with torch.no_grad():
            with tqdm(dataloader, desc="Inferring") as pb:
                for batch in dataloader:
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

                    pred = self.model.generate(
                        input_ids,
                        src_attention_mask,
                        tokenized_ocr,
                        ocr_attention_mask,
                        ocr_coordinates,
                        ocr_features,
                        tokenized_obj,
                        obj_attention_mask,
                        obj_coordinates,
                        obj_features,
                        start_symbol = self.decode_tokenizer.bos_idx,
                        end_symbol = self.decode_tokenizer.eos_idx,
                        max_ocr=self.config.max_ocr_length,
                        max_ques=self.config.max_q_length,
                        max_len = max_length
                    )

                    decoded_preds += self.decode_tokenizer.batch_decode(pred)

                pb.update()

        return decoded_preds

    @override
    def _build_model(self):
        log.info(f"# Building model architecture ...")
        if self.config.MODEL_MOD_CONFIG_CLASS is not None:   
            self.model_config = self.build_class(self.config.MODEL_MOD_CONFIG_CLASS)().build(self.config, len(self.tokenizer))
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.backbone_name)

        self.model = self.build_class(self.config.MODEL_CLASS)(self.model_config, self.decode_tokenizer.size)
        self.model = self.model.to(self.config.DEVICE)
    
    @override
    def _init_training_properties(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.LR, betas=self.config.BETAS, eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = self.optim, total_iters = self.config.warmup_step)

        self.SAVE = self.config.SAVE

        if os.path.isfile(os.path.join(self.config.SAVE_PATH, "last_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, "last_ckp.pth"))
            try:
                log.info(f"\t- Last train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Last train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])
            self.optim.load_state_dict(ckp['optimizer'])
            self.scheduler.load_state_dict(ckp['scheduler'])
            self.best_score = ckp['best_score']
    
    def _create_data_utils(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)
        self.tokenizer.add_tokens([self.config.context_token])

        train_qa_df = pd.read_csv(self.config.qa_train_path)[["image_id", "question", "answer", "filename"]]
        val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        self.val_answer = list(val_qa_df["answer"])

        self._create_decode_tokenizer()

        ocr_df = textlayout_ocr_adapt(self.config.base_ocr_feature_path, h_scale=1, w_scale=1)
        obj_df = textlayout_obj_adapt(self.config.base_obj_feature_path, h_scale=1, w_scale=1)

        log.info("# Creating Datasets")
        
        self.train_data = PhonemeSaLDataset(   
            qa_df = train_qa_df,
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
            max_output_length = self.config.max_a_length
        )

        self.val_data = PhonemeSaLDataset(
            qa_df = val_qa_df,
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
            max_output_length = self.config.max_a_length
        )

    def _init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)
        self.tokenizer.add_tokens([self.config.context_token])

        self._create_decode_tokenizer()

        if self.mode == "eval":
            log.info("###Load eval data ...")
            val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.base_ocr_feature_path, h_scale=1, w_scale=1)
            obj_df = textlayout_obj_adapt(self.config.base_obj_feature_path, h_scale=1, w_scale=1)

            self.val_data = PhonemeSaLDataset(
                qa_df = val_qa_df,
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
                max_output_length = self.config.max_a_length
            )
            
            self.val_answer = list(val_qa_df["answer"])
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)
            self.valiter_length = math.ceil(len(self.val_data)/self.config.EVAL_BATCH_SIZE)
        elif self.mode == "predict":
            log.info("###Load predict data ...")
            predict_qa_df = pd.read_csv(self.config.qa_predict_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.base_ocr_feature_path, h_scale=1, w_scale=1)
            obj_df = textlayout_obj_adapt(self.config.base_obj_feature_path, h_scale=1, w_scale=1)

            self.predict_data = PhonemeSaLDataset(     
                qa_df = predict_qa_df,
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
                max_output_length = self.config.max_a_length
            )
            
            if self.config.get_predict_score:
                self.predict_answer = list(predict_qa_df["answer"])
            else:
                self.predict_answer = None

            self.predictiter = DataLoader(dataset = self.predict_data, 
                                    batch_size=self.config.PREDICT_BATCH_SIZE)

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []

        if epoch <= self.config.NUM_FREEZE_EPOCH:
            for child in self.model.encoder.children():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for child in self.model.encoder.children():
                for param in child.parameters():
                    param.requires_grad = True

        with tqdm(self.trainiter, desc=f"Epoch {epoch} - Training") as pb:
            for batch in pb:
                label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)

                trg_input = labels[:, :-1]
                label_attention_mask = label_attention_mask[:, :-1]

                _, loss = self.model(
                    input_ids = batch['input_ids'].to(self.config.DEVICE),
                    label_ids = trg_input,
                    shifted_right_label_ids = labels[:, 1:],
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
                loss.backward()
                self.optim.step()

                self.scheduler.step()
                
                losses.append(loss.data.item())

                pb.set_postfix({
                    "loss": round(sum(losses) / len(losses), 2)
                })
                pb.update()

    def _create_decode_tokenizer(self):
        self.decode_tokenizer = PhonemeTokenizer()
