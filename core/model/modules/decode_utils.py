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

'''
def beam_generate(self, src, start_symbol, num_beam = 2, max_len=100):
    assert num_beam > 0

    src = src.to(DEVICE)

    bz = src.size(0)

    memory, mask = self.encode(src)
    memory = memory.to(DEVICE)

    ys = torch.ones(bz, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    out = self.decode(ys, memory, mask)
    prob = self.generator(out[:, -1])

    values, indices =  torch.topk(prob, num_beam, dim=-1)

    beams = [torch.cat([ys.clone().to(DEVICE), indices[:, i:i+1]], dim = 1) for i in range(num_beam)]

    beam_probs = [torch.log(values[:, i:i+1]) for i in range(num_beam)]

    done = [False]*num_beam

    eos_mask = [torch.ones(bz,1).type(torch.long).to(DEVICE)]*num_beam

    for _ in range(max_len-1):
        
        for b in range(num_beam):

            out = self.decode(beams[b], memory, mask)

            prob = self.generator(out[:, -1])

            vals, inds =  torch.topk(prob, 1, dim=-1)

            eos_mask[b] *= (inds!=eos_token_id)


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


'''