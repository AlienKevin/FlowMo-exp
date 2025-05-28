import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmo.vector_quantize import VectorQuantize
from flowmo.gptc import GPTC_models

class LARPQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stochastic Vector Quantizer (SVQ)
        self.vq = VectorQuantize(
            dim=self.config.model.codebook_size_for_entropy,
            codebook_size=2**self.config.model.codebook_size_for_entropy,
            codebook_dim=self.config.model.codebook_size_for_entropy,
            heads=1,
            separate_codebook_per_head=False,
            use_cosine_sim=True,
            commitment_weight=config.model.commit_loss_weight,
            codebook_diversity_loss_weight=0,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.03,
            straight_through=True,
            rotation_trick=False,
            learnable_codebook=False,
        )
        self.codebook_dim = self.vq._codebook.embed.shape[-1]
        self.codebook_size = self.vq.codebook_size

        # Continuous Autoregressive Prior Model
        # Expects config.prior to be an EasyDict with parameters for GPTCConfig
        # and other prior-specific settings.
        # Example: config.prior = edict(n_embd=256, n_head=8, n_layer=6, ...)
        prior_config = config.prior
        self.prior_model = GPTC_models[prior_config.model_name](n_ind=self.config.model.codebook_size_for_entropy)

        self.prior_loss_weight = prior_config.loss_weight
        self.use_mix_ss = prior_config.use_mix_ss
        self.mix_ss_max_ratio = prior_config.mix_ss_max_ratio

    def forward(self, x, return_loss_breakdown=True):
        # x: [batch_size, sequence_length, feature_dim] (continuous latents from encoder)
        batch_size, seq_len, _ = x.shape
        
        # 1. Stochastic Vector Quantization
        quantized_sg, indices, vq_loss, vq_loss_breakdown = self.vq(
            x,
            return_loss_breakdown=True
        )
        # `quantized_sg` is the straight-through gradient version: x + (codebook_vector - x).detach()
        # `indices` are the chosen codebook indices: [batch_size, sequence_length, num_heads (if any)]
        # Need to ensure indices are [B, T] if heads=1 or handle appropriately

        if self.vq.heads > 1 and self.vq.separate_codebook_per_head:
            # Assuming for now we want to work with a single sequence of indices
            # This might need adjustment based on how multi-head SVQ is intended to be used with AR prior
            indices_for_prior = indices[..., 0] # Take indices from the first head
        elif self.vq.heads > 1 and not self.vq.separate_codebook_per_head:
             indices_for_prior = indices # Should be [B, T*H] or [B,T,H] - check SVQ output
             if indices_for_prior.ndim > 2 : # if [B,T,H]
                 indices_for_prior = indices_for_prior.view(batch_size, -1) # Flatten to [B, T*H]
        else: # heads == 1
            indices_for_prior = indices # Should be [B, T]

        # 2. De-quantize for AR Prior Input (Z_hat = C_x)
        # The paper states: "de-quantization operation is performed via a straightforward index look-up, vˆ = Q−1(x) = Cx"
        # self.vq.codebook is [codebook_size, codebook_dim] or [heads, codebook_size, codebook_dim]
        
        codebook_for_lookup = self.vq.codebook
        if self.vq.heads > 1 and self.vq.separate_codebook_per_head:
            # If separate codebooks, use the first head's codebook for consistency with indices_for_prior
            codebook_for_lookup = codebook_for_lookup[0] # [codebook_size, codebook_dim]
        elif self.vq.heads > 1 and not self.vq.separate_codebook_per_head:
            # If shared codebook but multiple head outputs for indices, this implies each head chose an index from the *same* codebook
            # This case might be tricky if indices_for_prior is [B, T*H]. Let's assume indices_for_prior is [B, effective_seq_len]
            pass # codebook_for_lookup is already [codebook_size, codebook_dim]

        # Z_hat: [batch_size, seq_len_for_prior, codebook_dim]
        Z_hat = F.embedding(indices_for_prior, codebook_for_lookup)
        
        # 3. Autoregressive Prior Model
        # Prior input: Z_hat shifted by one (predict next token)
        # Prior target: actual next token indices
        prior_input = Z_hat[:, :-1, :] 
        prior_target_indices = indices_for_prior[:, 1:]

        # Get v_bar from prior model
        # The GPTC model's forward pass takes (x, targets=None).
        # We don't use its internal loss.
        v_bar_predicted, _ = self.prior_model(prior_input) # v_bar_predicted: [B, seq_len-1, codebook_dim]
        
        # 4. Calculate NLL loss (Lprior)
        # s = (v_bar_predicted @ self.vq.codebook.t()) / (torch.norm(v_bar_predicted, dim=-1, keepdim=True) * torch.norm(self.vq.codebook, dim=-1, keepdim=True).t())
        # Normalization for cosine similarity
        v_bar_norm = F.normalize(v_bar_predicted, p=2, dim=-1) # [B, seq_len-1, codebook_dim]
        codebook_norm = F.normalize(codebook_for_lookup, p=2, dim=-1) # [codebook_size, codebook_dim]
        
        # Cosine similarities (logits for softmax)
        # v_bar_norm is [B, T-1, D], codebook_norm.t() is [D, C]
        # logits_s will be [B, T-1, C]
        logits_s = torch.einsum('btd,cd->btc', v_bar_norm, codebook_norm)
        
        # Probabilities p = softmax(s)
        # NLL loss: F.cross_entropy expects logits [N, C] and targets [N]
        prior_nll_loss = F.cross_entropy(
            logits_s.reshape(-1, self.codebook_size), # [B*(T-1), C]
            prior_target_indices.reshape(-1)          # [B*(T-1)]
        )
        
        total_prior_loss = self.prior_loss_weight * prior_nll_loss

        # Scheduled Sampling (Simplified version for now)
        if self.use_mix_ss and self.training:
            # First pass loss is already computed (prior_nll_loss)
            
            # Mix predicted output with original input
            # Sample tokens from the first pass prediction
            probs_first_pass = F.softmax(logits_s, dim=-1) # [B, T-1, C]
            predicted_indices_first_pass = torch.multinomial(probs_first_pass.view(-1, self.codebook_size), 1).view_as(prior_target_indices) # [B, T-1]
            
            predicted_Z_hat_first_pass = F.embedding(predicted_indices_first_pass, codebook_for_lookup) # [B, T-1, D]
            
            # Mix:
            mask = torch.rand_like(predicted_Z_hat_first_pass) < self.mix_ss_max_ratio
            mixed_prior_input_token_embeddings = torch.where(mask, predicted_Z_hat_first_pass, prior_input) # prior_input is Z_hat[:, :-1, :]
            
            # Second forward pass with mixed input
            v_bar_predicted_ss, _ = self.prior_model(mixed_prior_input_token_embeddings)
            
            v_bar_norm_ss = F.normalize(v_bar_predicted_ss, p=2, dim=-1)
            logits_s_ss = torch.einsum('btd,cd->btc', v_bar_norm_ss, codebook_norm)
            
            prior_nll_loss_ss = F.cross_entropy(
                logits_s_ss.reshape(-1, self.codebook_size),
                prior_target_indices.reshape(-1) 
            )
            
            # Average loss from both rounds
            total_prior_loss = self.prior_loss_weight * (prior_nll_loss + prior_nll_loss_ss) / 2.0


        losses = {
            "quantizer_loss": vq_loss, # This is the total VQ loss (incl. commitment, etc.)
            "prior_loss": total_prior_loss,
        }
        if return_loss_breakdown and vq_loss_breakdown:
            losses["vq_commitment"] = vq_loss_breakdown.commitment
            losses["vq_codebook_diversity"] = vq_loss_breakdown.codebook_diversity
            losses["vq_orthogonal"] = vq_loss_breakdown.orthogonal_reg
            losses["vq_codebook_usage"] = torch.tensor(indices.unique().numel() / self.vq.codebook_size * 100)
        
        # The final "code" to be returned should be the output of SVQ that has gradients
        # which is `quantized_sg`
        return quantized_sg, indices, losses
