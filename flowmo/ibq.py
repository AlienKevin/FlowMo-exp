import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import reduce

def compute_entropy_loss(
    logits,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss

class IndexPropagationQuantize(nn.Module):
    def __init__(self, codebook_size, e_dim, beta=0.25, use_entropy_loss=True,
                 remap=None, unknown_index="random", cosine_similarity=False,
                 entropy_temperature=0.1,
                 sample_minimization_weight=1.0, batch_maximization_weight=1.0):
        super().__init__()

        self.codebook_size = codebook_size
        self.e_dim = e_dim
        self.use_entropy_loss = use_entropy_loss
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_size, self.e_dim)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = codebook_size

        self.cosine_similarity = cosine_similarity
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # z: [b, d, t]
        # embed.weight: [n, d]

        logits = einsum('b d t, n d -> b n t', z, self.embedding.weight)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.softmax(logits, dim=1)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros

        dim = 1
        ind = soft_one_hot.max(dim, keepdim=True)[1]
        hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, ind, 1.0)
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

        z_q = einsum('b n t, n d -> b d t', one_hot, self.embedding.weight)
        z_q_2 = einsum('b n t, n d -> b d t', hard_one_hot, self.embedding.weight)

        commit_loss = torch.mean((z_q - z)**2)
        double_quant_loss = torch.mean((z_q_2.detach()-z)**2) + self.beta * \
                   torch.mean((z_q_2 - z.detach()) ** 2)
        quant_loss = commit_loss + double_quant_loss
        diff = quant_loss

        if self.use_entropy_loss:
            sample_entropy, avg_entropy, entropy_loss= compute_entropy_loss(logits=logits.permute(0, 2, 1).reshape(-1, self.codebook_size), temperature=self.entropy_temperature, sample_minimization_weight=self.sample_minimization_weight, batch_maximization_weight=self.batch_maximization_weight) # logits [b d t] -> [b * t, n]
            diff = (commit_loss, double_quant_loss, sample_entropy, avg_entropy, entropy_loss)

        ind = torch.flatten(ind)
        if self.remap is not None:
            ind = ind.reshape(z.shape[0], -1)
            ind = self.remap_to_used(ind)
            ind = ind.reshape(-1, 1)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, t, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q
