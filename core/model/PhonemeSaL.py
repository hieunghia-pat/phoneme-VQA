import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import T5LayerNorm

from .modules import (
    T52DEncoderModel, 
    RelativePositionBias1D, 
    SCPRelativePositionBias,
    RelativePositionBiasAggregated,
    SinusoidalPositionalEncoding,
    BaseDecoder 
)

class CustomizedSaL_config:
    def build(self, config, new_token_embedding_size):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"ocr_hidden" : config.ocr_hidden,
                                "obj_hidden" : config.obj_hidden,
                                "new_token_embedding_size": new_token_embedding_size,
                                "num_decoder_layers": config.num_decoder_layers,
                                "n_head": config.n_head})
        
        return model_config


class PhonemeSaL(nn.Module):
    def __init__(self, 
                config, 
                vocab_size, 
                obj_dropout=0.1, 
                ocr_dropout=0.1):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.encoder = T52DEncoderModel.from_pretrained(self.config._name_or_path)
        self.encoder.resize_token_embeddings(self.config.new_token_embedding_size)
        
        self.rel2Dbias = RelativePositionBiasAggregated(Relative1D=RelativePositionBias1D(num_heads = self.encoder.config.num_heads),
                                                        SCP=SCPRelativePositionBias(num_heads = self.encoder.config.num_heads))

        self.obj_dropout = nn.Dropout(obj_dropout)
        self.obj_feature_projector = nn.Linear(self.config.obj_hidden, self.encoder.config.d_model)
        self.obj_bbox_projector = nn.Linear(4, self.encoder.config.d_model)
        self.obj_feature_layer_norm = T5LayerNorm(self.encoder.config.d_model)

        self.ocr_dropout = nn.Dropout(ocr_dropout)
        self.ocr_feature_projector = nn.Linear(self.config.ocr_hidden, self.encoder.config.d_model)
        self.ocr_bbox_projector = nn.Linear(4, self.encoder.config.d_model)
        self.ocr_feature_layer_norm = T5LayerNorm(self.encoder.config.d_model)

        ###### CUSTOM COMPONENTS ######

        self.tgt_tok_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.encoder.config.d_model
        )

        self.positional_encoding = SinusoidalPositionalEncoding(self.encoder.config.d_model, dropout=0.1)

        self.decoder = BaseDecoder(
                            emb_size=self.encoder.config.d_model,
                            num_layers=self.config.num_decoder_layers,
                            n_head=self.config.n_head
                        )

        self.lm_head = nn.Linear(self.encoder.config.d_model, vocab_size)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,
                input_ids,
                src_attention_mask,
                label_ids,
                shifted_right_label_ids,
                label_attention_mask,
                tokenized_ocr,
                ocr_attention_mask,
                ocr_coordinates,
                ocr_features,
                tokenized_obj,
                obj_attention_mask,
                obj_coordinates,
                obj_features,
                max_ocr,
                max_ques) :

        obj_inputs_embeds = self._calculate_obj_embedding(
                tokenized_obj, obj_coordinates, obj_features)
        
        ocr_inputs_embeds = self._calculate_ocr_embedding(
                tokenized_ocr, ocr_coordinates, ocr_features)
        
        ques_inputs_embeds = self.encoder.shared(input_ids)

        multi_modal_feat = torch.cat([ques_inputs_embeds, ocr_inputs_embeds, obj_inputs_embeds], dim=1)

        input_attention_mask = torch.cat(
            [src_attention_mask, ocr_attention_mask, obj_attention_mask], dim=1)
        
        position_bias = self.rel2Dbias(multi_modal_feat, input_attention_mask, ocr_coordinates, max_ques, max_ocr)

        encoder_outputs = self.encoder(
                attention_mask=input_attention_mask,
                inputs_embeds=multi_modal_feat,
                position_bias=position_bias
            ).last_hidden_state

        decoder_outputs = self.decode(label_ids, 
                                        encoder_outputs, 
                                        input_attention_mask, 
                                        label_attention_mask)

        logits = self.lm_head(decoder_outputs)

        loss = self.loss_fn(logits.reshape((-1, self.vocab_size)), shifted_right_label_ids.reshape(-1))

        return logits, loss
    
    def decode(self, labels, encoder_outputs, encoder_attention_mask, label_attention_mask=None):
        square_subsequent_mask = self._create_square_subsequent_mask(labels.size(1), device=labels.device)
        label_embedding = self.tgt_tok_emb(labels)
        label_embedding = self.positional_encoding(label_embedding)

        return self.decoder(label_embedding,
                            encoder_outputs,
                            tgt_mask = square_subsequent_mask,
                            memory_key_padding_mask = encoder_attention_mask,
                            tgt_key_padding_mask = label_attention_mask)

    
    def generate(self, 
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
                    max_ocr,
                    max_ques,
                    start_symbol,
                    end_symbol,
                    max_len=100):
        
        bz = input_ids.size(0)
        DEVICE = input_ids.device

        obj_inputs_embeds = self._calculate_obj_embedding(
                tokenized_obj, obj_coordinates, obj_features)
        
        ocr_inputs_embeds = self._calculate_ocr_embedding(
                tokenized_ocr, ocr_coordinates, ocr_features)
        
        ques_inputs_embeds = self.encoder.shared(input_ids)

        multi_modal_feat = torch.cat([ques_inputs_embeds, ocr_inputs_embeds, obj_inputs_embeds], dim=1)

        input_attention_mask = torch.cat(
            [src_attention_mask, ocr_attention_mask, obj_attention_mask], dim=1)
        
        position_bias = self.rel2Dbias(multi_modal_feat, input_attention_mask, ocr_coordinates, max_ques, max_ocr)

        encoder_outputs = self.encoder(
                attention_mask=input_attention_mask,
                inputs_embeds=multi_modal_feat,
                position_bias=position_bias
            ).last_hidden_state

        ys = torch.tensor([start_symbol], dtype=torch.long).repeat(bz, 1).to(DEVICE)
        break_signal = torch.zeros_like(ys).fill_(0)
        for _ in range(max_len):
            encoder_outputs = encoder_outputs.to(DEVICE)

            out = self.decode(ys, encoder_outputs, input_attention_mask)

            onset_output = self.lm_head(out)  # (batch_size, seq_len, onset_vocab_size)
            next_word = torch.argmax(onset_output[:, -1], dim=-1)
            break_signal = torch.where(next_word == end_symbol, 1, break_signal)

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if torch.all(break_signal):
                break

        return ys
    
    
    def _calculate_obj_embedding(self, tokenized_obj, obj_coordinates, obj_features):
        return self.obj_feature_layer_norm(self.obj_feature_projector(obj_features))\
            + self.obj_feature_layer_norm(self.obj_bbox_projector(obj_coordinates))\
            + self.encoder.shared(tokenized_obj)
    
    def _calculate_ocr_embedding(self, tokenized_ocr, ocr_coordinates, ocr_features):
        return self.ocr_feature_layer_norm(self.ocr_feature_projector(ocr_features))\
            + self.ocr_feature_layer_norm(self.ocr_bbox_projector(ocr_coordinates))\
            + self.encoder.shared(tokenized_ocr)
    
    def _create_square_subsequent_mask(self, sz, device="cuda"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    