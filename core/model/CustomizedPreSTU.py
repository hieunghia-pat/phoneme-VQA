import torch
import torch.nn as nn
from transformers import T5EncoderModel, ViTModel, AutoConfig
from .modules import SinusoidalPositionalEncoding, TokenEmbedding, BaseDecoder 

class CustomizedPreSTU_config:
    def build(self, config):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"vit_model" : config.vit_model_name,
                            "num_decoder_layers": config.num_decoder_layers,
                            "n_head": config.n_head})
                            
        return model_config

class CustomizedPreSTU(nn.Module):
    def __init__(self, config, tgt_vocab_size):
        super().__init__()

        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(self.config._name_or_path)

        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.visual_projector = nn.Linear(self.vit.config.hidden_size, self.encoder.config.d_model)

        ###### CUSTOM COMPONENTS ######

        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, self.encoder.config.d_model)

        self.positional_encoding = SinusoidalPositionalEncoding(
            self.encoder.config.d_model, dropout=0.1)

        self.decoder = BaseDecoder(emb_size = self.encoder.config.d_model, 
                                num_layers = self.config.num_decoder_layers,
                                n_head = self.config.n_head)
        self.lm_head = nn.Linear(self.encoder.config.d_model, tgt_vocab_size)


    def forward(self,
                pixel_values,
                input_ids,
                labels,
                src_attention_mask,
                label_attention_mask):

        inputs_embeds, attention_mask = self.calculate_embedding(
                pixel_values, input_ids, src_attention_mask)

        encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state

        decoder_outputs = self.decode(labels, 
                                        encoder_outputs, 
                                        attention_mask, 
                                        label_attention_mask)


        return self.lm_head(decoder_outputs)
    
    def decode(self, labels, encoder_outputs, encoder_attention_mask, label_attention_mask=None):
        square_subsequent_mask = self._create_square_subsequent_mask(labels.size(1), device=labels.device)
        
        label_embedding = self.positional_encoding(
                                self.tgt_tok_emb(labels))

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
                 start_symbol,
                 end_symbol,
                 max_length = 20,
                 isgreedy = True,
                 num_beam = 2):

        if isgreedy:
            return self.greedy_generate(pixel_values,
                                        coordinates,
                                        input_ids,
                                        src_attention_mask,
                                        ocr_attention_mask,
                                        tokenized_ocr,
                                        start_symbol,
                                        end_symbol,
                                        max_length)

        return self.beam_generate(pixel_values,
                                        coordinates,
                                        input_ids,
                                        src_attention_mask,
                                        ocr_attention_mask,
                                        tokenized_ocr,
                                        start_symbol,
                                        end_symbol,
                                        max_length,
                                        num_beam)
    
    def greedy_generate(self, 
                    pixel_values,
                    coordinates,
                    input_ids,
                    src_attention_mask,
                    ocr_attention_mask,
                    tokenized_ocr,
                    start_symbol,
                    end_symbol,
                    max_len=100):
        
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

            out = self.decode(ys, encoder_outputs, attention_mask)

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
                    start_symbol,
                    end_symbol,
                    max_len=100, 
                    num_beam=2):

        
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
        out = self.decode(ys, encoder_outputs, attention_mask)
        prob = self.lm_head(out[:, -1])

        values, indices = torch.topk(prob, num_beam, dim=-1)
        beams = [torch.cat([ys.clone().to(DEVICE), indices[:, i:i+1]], dim = 1) for i in range(num_beam)]
        beam_probs = [torch.log(values[:, i:i+1]) for i in range(num_beam)]

        done = [False]*num_beam
        eos_mask = [torch.ones(bz,1).type(torch.long).to(DEVICE)]*num_beam

        for _ in range(max_len-1):
            
            for b in range(num_beam):

                out = self.decode(ys, encoder_outputs, attention_mask)
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

    def _create_square_subsequent_mask(self, sz, device="cuda"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask