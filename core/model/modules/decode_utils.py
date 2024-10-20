'''
import torch

def greedy_generate(self, 
                    src, 
                    start_symbol,
                    end_symbol, 
                    encoder,
                    decoder,
                    lm_head,
                    max_len=100, 
                    DEVICE = "cuda"):
        src = src.to(DEVICE)

        bz = src.size(0)

        memory, mask = encoder(src)

        ys = torch.ones(bz, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

        for i in range(max_len):
            memory = memory.to(DEVICE)

            out = decoder(ys, memory, mask)

            prob = lm_head(out[:, -1])

            next_word = torch.argmax(prob, dim=-1).view(bz,-1)

            ys = torch.cat([ys, next_word], dim=1)

            if torch.any(ys == end_symbol, dim=1).sum() == bz:
                break

        return ys
'''