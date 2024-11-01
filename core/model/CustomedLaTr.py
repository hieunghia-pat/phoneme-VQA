import torch
import torch.nn as nn
from transformers import T5EncoderModel, ViTModel, AutoConfig
from .modules import SinusoidalPositionalEncoding, TokenEmbedding, BaseDecoder 

class LaTr_config:
    def build(self, config):
        model_config = AutoConfig.from_pretrained(config.encoder_name)

        model_config.update({"max_2d_position_embeddings" : config.max_2d_position_embeddings,
                                "vit_model" : config.vit_model_name,
                                "num_decoder_layers": config.num_decoder_layers})
        
        return model_config

class SpatialModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_left_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.top_left_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.width_emb = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.height_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)

    def forward(self, coordinates):
        top_left_x_feat = self.top_left_x(coordinates[:, :, 0])
        top_left_y_feat = self.top_left_y(coordinates[:, :, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:, :, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:, :, 3])
        width_feat = self.width_emb(coordinates[:, :, 4])
        height_feat = self.height_emb(coordinates[:, :, 5])

        layout_feature = top_left_x_feat + top_left_y_feat + \
            bottom_right_x_feat + bottom_right_y_feat + width_feat + height_feat
        return layout_feature


class LaTr(nn.Module):
    def __init__(self, config, tgt_vocab_size=300):
        super().__init__()

        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(self.config._name_or_path)

        self.spatial_feat_extractor = SpatialModule(config)
        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.visual_projector = nn.Linear(self.vit.config.hidden_size, self.encoder.config.d_model)

        #freeze ViT except the last dense layer
        for name, child in self.vit.named_children():
            for param in child.parameters():
                param.requires_grad = False
        
        ###### CUSTOM COMPONENTS ######

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, self.encoder.config.d_model)

        self.positional_encoding = SinusoidalPositionalEncoding(
            self.encoder.config.d_model, dropout=0.1)

        self.decoder = BaseDecoder(emb_size = self.encoder.config.d_model, num_layers = self.config.num_decoder_layers)
        self.lm_head = nn.Linear(self.encoder.config.d_model, tgt_vocab_size)

    def forward(self,
                pixel_values,
                coordinates,
                input_ids,
                labels,
                src_attention_mask,
                label_attention_mask,
                ocr_attention_mask,
                tokenized_ocr) :

        inputs_embeds, attention_mask = self._calculate_embedding(
                pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr)

        encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state

        decoder_outputs = self.decode(labels, 
                                        encoder_outputs, 
                                        attention_mask, 
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
                 pixel_values,
                 coordinates,
                 input_ids,
                 src_attention_mask,
                 ocr_attention_mask,
                 tokenized_ocr,
                 max_length = 20,
                 isgreedy = True):

        if isgreedy:
            return self.greedy_generate(pixel_values,
                                        coordinates,
                                        input_ids,
                                        src_attention_mask,
                                        ocr_attention_mask,
                                        tokenized_ocr,
                                        max_length)

        return self.beam_generate(pixel_values,
                                        coordinates,
                                        input_ids,
                                        src_attention_mask,
                                        ocr_attention_mask,
                                        tokenized_ocr,
                                        max_length)
    
    def greedy_generate(self, 
                    pixel_values,
                    coordinates,
                    input_ids,
                    src_attention_mask,
                    ocr_attention_mask,
                    tokenized_ocr,
                    max_len=100):
        
        start_symbol = self.encoder.pad_token_id
        end_symbol = self.encoder.eos_token_id

        bz = input_ids.size(0)
        DEVICE = input_ids.device

        inputs_embeds, attention_mask = self._calculate_embedding(
                pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr)

        encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state

        ys = torch.ones(bz, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

        for i in range(max_len):
            encoder_outputs = encoder_outputs.to(DEVICE)

            out = self.decoder(ys, encoder_outputs, attention_mask)

            prob = self.lm_head(out[:, -1])

            next_word = torch.argmax(prob, dim=-1).view(bz,-1)

            ys = torch.cat([ys, next_word], dim=1)

            if torch.any(ys == end_symbol, dim=1).sum() == bz:
                break

        return ys
    
    def beam_generate(self, 
                    pixel_values,
                    coordinates,
                    input_ids,
                    src_attention_mask,
                    ocr_attention_mask,
                    tokenized_ocr,
                    max_len=100,
                    num_beam=2):

        start_symbol = self.encoder.pad_token_id
        end_symbol = self.encoder.eos_token_id

        bz = input_ids.size(0)
        DEVICE = input_ids.device

        inputs_embeds, attention_mask = self._calculate_embedding(
                pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr)

        encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state
        
        ys = torch.ones(bz, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        
        encoder_outputs = encoder_outputs.to(DEVICE)
        out = self.decoder(ys, encoder_outputs, attention_mask)
        prob = self.lm_head(out[:, -1])

        values, indices = torch.topk(prob, num_beam, dim=-1)
        beams = [torch.cat([ys.clone().to(DEVICE), indices[:, i:i+1]], dim = 1) for i in range(num_beam)]
        beam_probs = [torch.log(values[:, i:i+1]) for i in range(num_beam)]

        done = [False]*num_beam
        eos_mask = [torch.ones(bz,1).type(torch.long).to(DEVICE)]*num_beam

        for _ in range(max_len-1):
            
            for b in range(num_beam):

                out = self.decoder(ys, encoder_outputs, attention_mask)
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
        
    def _calculate_embedding(self, pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr):
        img_feat = self.visual_projector(self.vit(pixel_values).last_hidden_state)
        spatial_feat = self.spatial_feat_extractor(coordinates)
        ocr_feat = self.encoder.shared(tokenized_ocr)
        language_feat = self.encoder.shared(input_ids)

        layout_feat = ocr_feat + spatial_feat

        multi_modal_feat = torch.cat([img_feat, layout_feat, language_feat], axis=1)
        input_attention_mask = torch.cat(
            [torch.ones(img_feat.shape[:2]).to(img_feat.device), ocr_attention_mask, src_attention_mask], axis=1)

        return multi_modal_feat, input_attention_mask

    def _create_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=sz.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask