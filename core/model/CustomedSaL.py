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
    TokenEmbedding, 
    BaseDecoder 
)

class SaL_config:
    def build(self, config, new_token_embedding_size):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"ocr_hidden" : config.ocr_hidden,
                                "obj_hidden" : config.obj_hidden,
                                "new_token_embedding_size": new_token_embedding_size})
        
        return model_config


class SaL(nn.Module):
    def __init__(self, config, tgt_vocab_size, obj_dropout=0.1, ocr_dropout=0.1):
        super().__init__()

        self.config = config
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

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, self.encoder.config.d_model)

        self.positional_encoding = SinusoidalPositionalEncoding(
            self.encoder.config.d_model, dropout=0.1)

        self.decoder = BaseDecoder(emb_size = self.encoder.config.d_model, num_layers = self.config.num_decoder_layers)
        self.lm_head = nn.Linear(self.encoder.config.d_model, tgt_vocab_size)
        

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


        return self.encoder.lm_head(decoder_outputs)
    
    def decode(self, labels, encoder_outputs, encoder_attention_mask, label_attention_mask):
        square_subsequent_mask = self._create_square_subsequent_mask(labels)
        
        label_embedding = self.positional_encoding(
                                self.trg_tok_emb(labels))

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
                max_length = 20,
                isgreedy = True,
                num_beam = 2):

        
        if isgreedy:
            return self.greedy_generate(input_ids,
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
                                        max_length)

        return self.beam_generate(input_ids,
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
                                    max_length,
                                    num_beam)
    
    def greedy_generate(self, 
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
                    max_len=100):
        
        start_symbol = self.encoder.pad_token_id
        end_symbol = self.encoder.eos_token_id

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

        ys = torch.ones(bz, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

        for i in range(max_len):
            encoder_outputs = encoder_outputs.to(DEVICE)

            out = self.decoder(ys, encoder_outputs, input_attention_mask)

            prob = self.lm_head(out[:, -1])

            next_word = torch.argmax(prob, dim=-1).view(bz,-1)

            ys = torch.cat([ys, next_word], dim=1)

            if torch.any(ys == end_symbol, dim=1).sum() == bz:
                break

        return ys
    
    def beam_generate(self, 
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
                    max_len=100,
                    num_beam=2):
        start_symbol = self.encoder.pad_token_id
        end_symbol = self.encoder.eos_token_id

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
        
        ys = torch.ones(bz, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        
        encoder_outputs = encoder_outputs.to(DEVICE)
        out = self.decoder(ys, encoder_outputs, input_attention_mask)
        prob = self.lm_head(out[:, -1])

        values, indices = torch.topk(prob, num_beam, dim=-1)
        beams = [torch.cat([ys.clone().to(DEVICE), indices[:, i:i+1]], dim = 1) for i in range(num_beam)]
        beam_probs = [torch.log(values[:, i:i+1]) for i in range(num_beam)]

        done = [False]*num_beam
        eos_mask = [torch.ones(bz,1).type(torch.long).to(DEVICE)]*num_beam

        for _ in range(max_len-1):
            
            for b in range(num_beam):

                out = self.decoder(ys, encoder_outputs, input_attention_mask)
                prob = self.lm_head(out[:, -1])

                vals, inds =  torch.topk(prob, 1, dim=-1)

                eos_mask[b] *= (inds!=end_symbol)


                beams[b] = torch.cat([beams[b], inds], dim=1)

                if eos_mask[b].sum() == 0:
                    done[b] = True
                    continue
                
                beam_probs[b] += torch.log(vals)*eos_mask[b]

            if all(done):
                break
        
        beam_probs = torch.cat(beam_probs, dim=-1).cpu().detach()
        beams = torch.stack(beams, dim=1).cpu().detach()

        beam_idx = torch.argmax(beam_probs, dim=-1)

        chosen = beams[torch.tensor([i for i in range(bz)]), beam_idx.flatten(), :]
        
        return chosen
        
    
    def _calculate_obj_embedding(self, tokenized_obj, obj_coordinates, obj_features):
        return self.obj_feature_layer_norm(self.obj_feature_projector(obj_features))\
            + self.obj_feature_layer_norm(self.obj_bbox_projector(obj_coordinates))\
            + self.encoder.shared(tokenized_obj)
    
    def _calculate_ocr_embedding(self, tokenized_ocr, ocr_coordinates, ocr_features):
        return self.ocr_feature_layer_norm(self.ocr_feature_projector(ocr_features))\
            + self.ocr_feature_layer_norm(self.ocr_bbox_projector(ocr_coordinates))\
            + self.encoder.shared(tokenized_ocr)
    
    def _create_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=sz.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask