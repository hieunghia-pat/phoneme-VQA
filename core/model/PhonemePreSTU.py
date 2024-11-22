import torch
import torch.nn as nn
from transformers import T5EncoderModel, ViTModel, AutoConfig
from .modules import SinusoidalPositionalEncoding, PhonemeEmbedding, BaseDecoder 

class CustomizedPreSTU_config:
    def build(self, config):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"vit_model" : config.vit_model_name,
                            "num_decoder_layers": config.num_decoder_layers,
                            "n_head": config.n_head})
                            
        return model_config

class PhonemePreSTU(nn.Module):
    def __init__(self, 
                config, 
                onset_vocab_size,
                rhyme_vocab_size,
                tone_vocab_size):
        super().__init__()

        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(self.config._name_or_path)

        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.visual_projector = nn.Linear(self.vit.config.hidden_size, self.encoder.config.d_model)

        ###### CUSTOM COMPONENTS ######

        self.rhyme_tone_embed_dim = self.encoder.config.d_model // 3
        self.onset_embed_dim = int(self.encoder.config.d_model - self.rhyme_tone_embed_dim*2)

        self.tgt_tok_emb = PhonemeEmbedding(
            onset_vocab_size,
            rhyme_vocab_size,
            tone_vocab_size,
            self.onset_embed_dim,
            self.rhyme_tone_embed_dim
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            self.encoder.config.d_model, dropout=0.1)

        self.decoder = BaseDecoder(
                            emb_size=self.encoder.config.d_model,
                            num_layers=self.config.num_decoder_layers,
                            n_head=self.config.n_head
                        )

        # shared lm_head
        self.shared_lm_head = nn.Linear(
            self.encoder.config.d_model, self.encoder.config.d_model)

        # phoneme lm_head
        self.onset_lm_head = nn.Linear(self.onset_embed_dim, onset_vocab_size)
        self.rhyme_lm_head = nn.Linear(self.rhyme_tone_embed_dim, rhyme_vocab_size)
        self.tone_lm_head = nn.Linear(self.rhyme_tone_embed_dim, tone_vocab_size)


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


        decoder_outputs = self.shared_lm_head(decoder_outputs)


        onset_out = decoder_outputs[:, :, :self.onset_embed_dim]  # (batch_size, seq_len, d_model//3 + d_model%3)
        rhyme_out = decoder_outputs[:, :, self.onset_embed_dim:self.onset_embed_dim+self.rhyme_tone_embed_dim] # (batch_size, seq_len, d_model//3)
        tone_out = decoder_outputs[:, :, self.onset_embed_dim+self.rhyme_tone_embed_dim:] # (batch_size, seq_len, d_model//3)

        onset_output = self.onset_lm_head(onset_out)  # (batch_size, seq_len, onset_vocab_size)
        rhyme_output = self.rhyme_lm_head(rhyme_out)
        tone_output = self.tone_lm_head(tone_out)  

        return onset_output, rhyme_output, tone_output
    
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


        return self.greedy_generate(pixel_values,
                                    coordinates,
                                    input_ids,
                                    src_attention_mask,
                                    ocr_attention_mask,
                                    tokenized_ocr,
                                    start_symbol,
                                    end_symbol,
                                    max_length)

        
    
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

        ys = torch.tensor([[[start_symbol, 0, 0]]], dtype=torch.long).repeat(bz, 1, 1).to(DEVICE)

        for i in range(max_len):
            encoder_outputs = encoder_outputs.to(DEVICE)

            out = self.decode(ys, encoder_outputs, attention_mask)

            onset_out = out[:, :, :self.onset_embed_dim]  # (batch_size, seq_len, d_model//3 + d_model%3)
            rhyme_out = out[:, :, self.onset_embed_dim:self.onset_embed_dim+self.rhyme_tone_embed_dim] # (batch_size, seq_len, d_model//3)
            tone_out = out[:, :, self.onset_embed_dim+self.rhyme_tone_embed_dim:] # (batch_size, seq_len, d_model//3)   
            
            onset_output = self.onset_lm_head(onset_out)  # (batch_size, seq_len, onset_vocab_size)
            rhyme_output = self.rhyme_lm_head(rhyme_out)
            tone_output = self.tone_lm_head(tone_out)

            next_w_onset = torch.argmax(onset_output[:, -1], dim=-1)
            next_w_rhyme = torch.argmax(rhyme_output[:, -1], dim=-1)
            next_w_tone = torch.argmax(tone_output[:, -1], dim=-1)

            next_word = torch.stack([next_w_onset, next_w_rhyme, next_w_tone], dim=-1)

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if torch.any(ys[:,:,0] == end_symbol, dim=1).sum() == bz:
                break

        return ys
 
    
    def _calculate_embedding(self, pixel_values, input_ids, src_attention_mask):
        img_feat = self.visual_projector(self.vit(pixel_values).last_hidden_state)
        language_ocr_feat = self.encoder.shared(input_ids)

        multi_modal_feat = torch.cat([img_feat, language_ocr_feat], axis=1)
        input_attention_mask = torch.cat(
            [torch.ones(img_feat.shape[:2]).to(img_feat.device), src_attention_mask], axis=1)

        return multi_modal_feat, input_attention_mask

    def _create_square_subsequent_mask(self, sz, device="cuda"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask