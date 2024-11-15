import torch
import torch.nn as nn
from transformers import T5EncoderModel, ViTModel, AutoConfig
from modules import SinusoidalPositionalEncoding, VN_Embedding, BaseDecoder
import json

class PhonemeVocab:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.phonemes = json.load(f)

    def get_onset_length(self):
        return len(self.phonemes['onset'])

    def get_rhyme_length(self):
        return len(self.phonemes['rhyme'])

    def get_tone_length(self):
        return len(self.phonemes['tone'])

class CustomizedLaTr_config:
    def build(self, config):
        model_config = AutoConfig.from_pretrained(config.encoder_name)
        model_config.update({
            "max_2d_position_embeddings": config.max_2d_position_embeddings,
            "vit_model": config.vit_model_name,
            "num_decoder_layers": config.num_decoder_layers,
            "n_head": config.n_head
        })
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
        self.width_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.height_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)

    def forward(self, coordinates):
        top_left_x_feat = self.top_left_x(coordinates[:, :, 0])
        top_left_y_feat = self.top_left_y(coordinates[:, :, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:, :, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:, :, 3])
        width_feat = self.width_emb(coordinates[:, :, 4])
        height_feat = self.height_emb(coordinates[:, :, 5])

        layout_feature = (top_left_x_feat + top_left_y_feat +
                          bottom_right_x_feat + bottom_right_y_feat +
                          width_feat + height_feat)
        return layout_feature

class CustomizedLaTr(nn.Module):
    def __init__(self, config, vocab_file):
        super().__init__()

        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(
            self.config._name_or_path)

        self.spatial_feat_extractor = SpatialModule(config)
        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.visual_projector = nn.Linear(
            self.vit.config.hidden_size, self.encoder.config.d_model)

        # Đóng băng ViT trừ lớp dense cuối
        for name, child in self.vit.named_children():
            for param in child.parameters():
                param.requires_grad = False

        ###### THÀNH PHẦN TÙY CHỈNH ######

        self.tgt_tok_emb = VN_Embedding(
            vocab_file='vocab.json',
            embedding_dim=self.encoder.config.d_model,
            dropout_rate=0
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            self.encoder.config.d_model, dropout=0.1)

        self.decoder = BaseDecoder(
            emb_size=self.encoder.config.d_model,
            num_layers=self.config.num_decoder_layers,
            n_head=self.config.n_head
        )

        # Định nghĩa special_lm_head
        self.special_lm_head = nn.Linear(
            self.encoder.config.d_model, self.encoder.config.d_model)

        # Sử dụng PhonemeVocab để lấy độ dài từ vựng
        phoneme_vocab = PhonemeVocab(vocab_file)
        d_model_third = self.encoder.config.d_model // 3
        self.onset_lm_head = nn.Linear(
            d_model_third, phoneme_vocab.get_onset_length())
        self.rhyme_lm_head = nn.Linear(
            d_model_third, phoneme_vocab.get_rhyme_length())
        self.tone_lm_head = nn.Linear(
            d_model_third, phoneme_vocab.get_tone_length())

    def forward(self,
                pixel_values,
                coordinates,
                input_ids,
                labels,
                src_attention_mask,
                label_attention_mask,
                ocr_attention_mask,
                tokenized_ocr):

        inputs_embeds, attention_mask = self._calculate_embedding(
            pixel_values, coordinates, input_ids, ocr_attention_mask,
            src_attention_mask, tokenized_ocr)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        ).last_hidden_state

        decoder_outputs = self.decode(labels,
                                      encoder_outputs,
                                      attention_mask,
                                      label_attention_mask)

        # Truyền decoder_outputs qua special_lm_head
        decoder_outputs = self.special_lm_head(decoder_outputs)

        # Reshape decoder_outputs để có kích thước (batch_size, seq_len, 3, d_model//3)
        batch_size, sequence_length, d_model = decoder_outputs.size()
        decoder_outputs = decoder_outputs.view(batch_size, sequence_length, 3, d_model // 3)

        # Tách thành 3 vector
        onset_decode = decoder_outputs[:, :, 0, :].unsqueeze(2)  # (batch_size, seq_len, 1, d_model//3)
        rhyme_decode = decoder_outputs[:, :, 1, :].unsqueeze(2)
        tone_decode = decoder_outputs[:, :, 2, :].unsqueeze(2)

        # Truyền qua các lm_head tương ứng
        onset_output = self.onset_lm_head(onset_decode)  # (batch_size, seq_len, onset_vocab_size)
        rhyme_output = self.rhyme_lm_head(rhyme_decode)
        tone_output = self.tone_lm_head(tone_decode)

        return onset_output, rhyme_output, tone_output

    def decode(self, tgt, memory, attention_mask, tgt_attention_mask):
        tgt_emb = self.tgt_tok_emb(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)

        tgt_mask = self._create_square_subsequent_mask(
            tgt_emb.size(1)).to(tgt_emb.device)

        return self.decoder(tgt=tgt_emb,
                            memory=memory,
                            tgt_mask=tgt_mask,
                            memory_mask=None,
                            tgt_key_padding_mask=tgt_attention_mask,
                            memory_key_padding_mask=attention_mask)

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

        ys = torch.tensor([[[4, 0, 0]]], dtype=torch.long).repeat(bz, 1, 1).to(DEVICE)

        for i in range(max_len):
            encoder_outputs = encoder_outputs.to(DEVICE)

            out = self.decode(ys, encoder_outputs, attention_mask)

            # Reshape decoder_outputs để có kích thước (batch_size, seq_len, 3, d_model//3)
            batch_size, i, d_model = out.size()
            out = out.view(batch_size,i, 3, d_model // 3)

            # Tách thành 3 vector
            onset_decode = out[:, :, 0, :].unsqueeze(2)  # (batch_size, 1, 1, d_model//3)
            rhyme_decode = out[:, :, 1, :].unsqueeze(2)
            tone_decode = out[:, :, 2, :].unsqueeze(2)

            # Truyền qua các lm_head tương ứng
            onset_output = self.onset_lm_head(onset_decode)  # (batch_size, 1, onset_vocab_size)
            rhyme_output = self.rhyme_lm_head(rhyme_decode)
            tone_output = self.tone_lm_head(tone_decode)

            # Tính toán từ tiếp theo cho mỗi phần
            next_w_onset = torch.argmax(onset_output, dim=-1)
            next_w_rhyme = torch.argmax(rhyme_output, dim=-1)
            next_w_tone = torch.argmax(tone_output, dim=-1)

            # Kết hợp thành từ tiếp theo
            next_word = torch.cat([next_w_onset, next_w_rhyme, next_w_tone], dim=-1)

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
        """
        Phương thức beam_generate cho phép mô hình sinh output theo hướng âm vị học sử dụng thuật toán Beam Search.
        """

        bz = input_ids.size(0)
        DEVICE = input_ids.device

        inputs_embeds, attention_mask = self._calculate_embedding(
            pixel_values, coordinates, input_ids, ocr_attention_mask, src_attention_mask, tokenized_ocr)

        encoder_outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        ).last_hidden_state

        # Khởi tạo beam
        ys = torch.tensor([[[start_symbol, 0, 0]]], dtype=torch.long).repeat(bz, 1, 1).to(DEVICE)  # (batch_size, seq_len=1, 3)
        beam_scores = torch.zeros(bz, num_beam).to(DEVICE)
        beam_sequences = [ys.clone() for _ in range(num_beam)]  # Danh sách lưu trữ các beam

        # Biến đánh dấu kết thúc
        done = [False for _ in range(bz)]

        for step in range(max_len):
            all_candidates = []
            for beam_idx in range(num_beam):
                seqs = beam_sequences[beam_idx]  # (batch_size, seq_len, 3)

                # Giải mã
                out = self.decode(seqs, encoder_outputs, attention_mask)

                # Reshape decoder_outputs để có kích thước (batch_size, seq_len, 3, d_model//3)
                batch_size, seq_len, d_model = out.size()
                out = out.view(batch_size, seq_len, 3, d_model // 3)

                # Lấy token cuối cùng
                onset_decode = out[:, :, 0, :]  # (batch_size, d_model//3)
                rhyme_decode = out[:, :, 1, :]
                tone_decode = out[:, :, 2, :]

                # Truyền qua các lm_head tương ứng
                onset_output = self.onset_lm_head(onset_decode)  # (batch_size, onset_vocab_size)
                rhyme_output = self.rhyme_lm_head(rhyme_decode)
                tone_output = self.tone_lm_head(tone_decode)

                # Tính log_probs
                onset_log_probs = torch.log_softmax(onset_output, dim=-1)
                rhyme_log_probs = torch.log_softmax(rhyme_output, dim=-1)
                tone_log_probs = torch.log_softmax(tone_output, dim=-1)

                # Kết hợp các log_probs
                total_log_probs = onset_log_probs.unsqueeze(2).unsqueeze(3) + \
                                  rhyme_log_probs.unsqueeze(1).unsqueeze(3) + \
                                  tone_log_probs.unsqueeze(1).unsqueeze(2)  # (batch_size, onset_size, rhyme_size, tone_size)

                # Chuyển về dạng 2D
                total_log_probs = total_log_probs.view(batch_size, -1)  # (batch_size, onset_size * rhyme_size * tone_size)

                # Lấy topk
                topk_log_probs, topk_indices = torch.topk(total_log_probs, num_beam, dim=-1)  # (batch_size, num_beam)

                for batch_idx in range(bz):
                    if done[batch_idx]:
                        continue

                    for k in range(num_beam):
                        idx = topk_indices[batch_idx, k]
                        log_prob = topk_log_probs[batch_idx, k]

                        # Chuyển index về onset, rhyme, tone
                        onset_size = onset_log_probs.size(-1)
                        rhyme_size = rhyme_log_probs.size(-1)
                        tone_size = tone_log_probs.size(-1)

                        onset_idx = idx // (rhyme_size * tone_size)
                        rhyme_idx = (idx // tone_size) % rhyme_size
                        tone_idx = idx % tone_size

                        next_token = torch.tensor([[onset_idx, rhyme_idx, tone_idx]], dtype=torch.long).to(DEVICE)  # (1, 3)

                        # Kiểm tra end_symbol
                        if onset_idx.item() == end_symbol:
                            done[batch_idx] = True

                        # Cập nhật sequence và score
                        new_seq = torch.cat([beam_sequences[beam_idx][batch_idx], next_token], dim=0).unsqueeze(0)  # (1, seq_len+1, 3)
                        new_score = beam_scores[batch_idx, beam_idx] + log_prob

                        all_candidates.append((new_score, new_seq, batch_idx))

            # Sắp xếp theo score
            all_candidates.sort(key=lambda x: x[0], reverse=True)

            # Chọn topk beam mới
            beam_sequences = []
            beam_scores = torch.zeros(bz, num_beam).to(DEVICE)
            for idx, (score, seq, batch_idx) in enumerate(all_candidates[:num_beam * bz]):
                beam_idx = idx % num_beam
                if beam_idx >= len(beam_sequences):
                    beam_sequences.append(seq)
                else:
                    beam_sequences[beam_idx] = torch.cat([beam_sequences[beam_idx], seq], dim=0)
                beam_scores[batch_idx, beam_idx] = score

            # Kiểm tra điều kiện dừng
            if all(done):
                break

        # Chọn beam có score cao nhất
        final_sequences = []
        for batch_idx in range(bz):
            best_beam_idx = torch.argmax(beam_scores[batch_idx]).item()
            final_sequences.append(beam_sequences[best_beam_idx][batch_idx])

        # Chuyển về tensor
        final_sequences = torch.stack(final_sequences, dim=0)  # (batch_size, seq_len, 3)

        return final_sequences
        
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