import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, ViTModel, AutoConfig

class PreSTU_config:
    def build(self, config):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"vit_model" : config.vit_model_name})
        
        return model_config

class PreSTU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)

        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.visual_projector = nn.Linear(self.vit.config.hidden_size, self.backbone.config.d_model)


    def forward(self,
                pixel_values,
                input_ids,
                labels,
                src_attention_mask,
                label_attention_mask):

        inputs_embeds, attention_mask = self.calculate_embedding(
                pixel_values, input_ids, src_attention_mask)

        encoder_outputs = self.backbone.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state

        decoder_outputs = self.backbone.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.backbone.shared(labels),
            attention_mask = label_attention_mask
        ).last_hidden_state


        return self.backbone.lm_head(decoder_outputs)

    def calculate_embedding(self, pixel_values, input_ids, src_attention_mask):
        img_feat = self.visual_projector(self.vit(pixel_values).last_hidden_state)
        language_ocr_feat = self.backbone.shared(input_ids)

        multi_modal_feat = torch.cat([img_feat, language_ocr_feat], axis=1)
        input_attention_mask = torch.cat(
            [torch.ones(img_feat.shape[:2]).to(img_feat.device), src_attention_mask], axis=1)

        return multi_modal_feat, input_attention_mask

    def generate(self,
                 pixel_values,
                 input_ids,
                 src_attention_mask,
                 max_length = 20):

        inputs_embeds, attention_mask = self.calculate_embedding(
                pixel_values, input_ids, src_attention_mask)

        return self.backbone.generate(inputs_embeds = inputs_embeds, max_length = max_length)