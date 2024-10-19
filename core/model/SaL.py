import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.t5.modeling_t5 import T5LayerNorm

from .modules import (
    T52dForConditionalGeneration, 
    RelativePositionBias1D, 
    SCPRelativePositionBias,
    RelativePositionBiasAggregated
)

class SaL_config:
    def build(self, config, new_token_embedding_size):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"ocr_hidden" : config.ocr_hidden,
                                "obj_hidden" : config.obj_hidden,
                                "new_token_embedding_size": new_token_embedding_size})
        
        return model_config


class SaL(nn.Module):
    def __init__(self, config, obj_dropout=0.1, ocr_dropout=0.1):
        super().__init__()

        self.config = config
        self.backbone = T52dForConditionalGeneration.from_pretrained(self.config._name_or_path)
        self.backbone.resize_token_embeddings(self.config.new_token_embedding_size)
        self.rel2Dbias = RelativePositionBiasAggregated(Relative1D=RelativePositionBias1D(num_heads = self.backbone.config.num_heads),
                                                        SCP=SCPRelativePositionBias(num_heads = self.backbone.config.num_heads))

        self.obj_dropout = nn.Dropout(obj_dropout)
        self.obj_feature_projector = nn.Linear(self.config.obj_hidden, self.backbone.config.d_model)
        self.obj_bbox_projector = nn.Linear(4, self.backbone.config.d_model)
        self.obj_feature_layer_norm = T5LayerNorm(self.backbone.config.d_model)

        self.ocr_dropout = nn.Dropout(ocr_dropout)
        self.ocr_feature_projector = nn.Linear(self.config.ocr_hidden, self.backbone.config.d_model)
        self.ocr_bbox_projector = nn.Linear(4, self.backbone.config.d_model)
        self.ocr_feature_layer_norm = T5LayerNorm(self.backbone.config.d_model)
        

    def forward(self,
                input_ids,
                src_attention_mask,
                label_ids,
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

        obj_inputs_embeds = self.calculate_obj_embedding(
                tokenized_obj, obj_coordinates, obj_features)
        
        ocr_inputs_embeds = self.calculate_ocr_embedding(
                tokenized_ocr, ocr_coordinates, ocr_features)
        
        ques_inputs_embeds = self.backbone.shared(input_ids)

        multi_modal_feat = torch.cat([ques_inputs_embeds, ocr_inputs_embeds, obj_inputs_embeds], dim=1)

        input_attention_mask = torch.cat(
            [src_attention_mask, ocr_attention_mask, obj_attention_mask], dim=1)
        
        position_bias = self.rel2Dbias(multi_modal_feat, input_attention_mask, ocr_coordinates, max_ques, max_ocr)

        encoder_outputs = self.backbone.encoder(
                attention_mask=input_attention_mask,
                inputs_embeds=multi_modal_feat,
                position_bias=position_bias
            ).last_hidden_state

        decoder_outputs = self.backbone.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.backbone.shared(label_ids),
            attention_mask = label_attention_mask
        ).last_hidden_state


        return self.backbone.lm_head(decoder_outputs)

    def calculate_obj_embedding(self, tokenized_obj, obj_coordinates, obj_features):
        return self.obj_feature_layer_norm(self.obj_feature_projector(obj_features))\
            + self.obj_feature_layer_norm(self.obj_bbox_projector(obj_coordinates))\
            + self.backbone.shared(tokenized_obj)
    
    def calculate_ocr_embedding(self, tokenized_ocr, ocr_coordinates, ocr_features):
        return self.ocr_feature_layer_norm(self.ocr_feature_projector(ocr_features))\
            + self.ocr_feature_layer_norm(self.ocr_bbox_projector(ocr_coordinates))\
            + self.backbone.shared(tokenized_ocr)


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
                max_length = 20):

        obj_inputs_embeds = self.calculate_obj_embedding(
                tokenized_obj, obj_coordinates, obj_features)
        
        ocr_inputs_embeds = self.calculate_ocr_embedding(
                tokenized_ocr, ocr_coordinates, ocr_features)
        
        ques_inputs_embeds = self.backbone.shared(input_ids)

        multi_modal_feat = torch.cat([ques_inputs_embeds, ocr_inputs_embeds, obj_inputs_embeds], dim=1)
        input_attention_mask = torch.cat(
            [src_attention_mask, ocr_attention_mask, obj_attention_mask], dim=1)

        position_bias = self.rel2Dbias(multi_modal_feat, input_attention_mask, ocr_coordinates, max_ques, max_ocr)

        return self.backbone.generate(inputs_embeds = multi_modal_feat,
                                        position_bias = position_bias, 
                                        max_length = max_length)