# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
import os
import time
import argparse
from omegaconf import OmegaConf
from einops import rearrange
# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

from flowmo import train_utils, models


def find_latest_checkpoint(exp_dir: str):
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint directory found at {ckpt_dir}")
    ckpts = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


class ARWrapper:
    """Wrapper around Qwen3ForCausalLM to match the expected interface."""
    
    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.prior_model = quantizer.prior_model
        self.codebook = quantizer.quantizer.embedding.weight
        self.proj_in = quantizer.prior_model_project_in
        self.proj_out = quantizer.prior_model_project_out
        self.cls_embedding = quantizer.prior_model_cls_embedding
        
    def __call__(self, x, cond_idx, input_pos):
        return self.forward(x, cond_idx, input_pos)
    
    def forward(self, x, cond_idx, input_pos):
        """
        Args:
            x: Token indices tensor of shape (batch_size, seq_len) or None for prefill
            cond_idx: Class conditioning indices of shape (batch_size,)
            input_pos: Position indices (ignored in this implementation)
        
        Returns:
            logits: Output logits of shape (batch_size, vocab_size)
            None: Placeholder for compatibility
        """
        device = self.codebook.device
        
        if x is None:
            # Prefill case - only class conditioning
            if cond_idx is not None:
                class_emb = self.cls_embedding(cond_idx, train=False)
                prompt_embedding = class_emb.to(torch.bfloat16)
                input_embeddings = prompt_embedding
            else:
                raise ValueError("Either x or cond_idx must be provided")
        else:
            # Decode case - tokens + optional class conditioning
            batch_size, seq_len = x.shape
            
            # Get token embeddings
            code_embs = self.codebook[x]
            code_embs_proj = self.proj_in(code_embs).to(torch.bfloat16)
            
            if cond_idx is not None:
                # Prepend class embedding
                class_emb = self.cls_embedding(cond_idx, train=False)
                prompt_embedding = class_emb.to(torch.bfloat16)
                input_embeddings = torch.cat([prompt_embedding, code_embs_proj], dim=1)
            else:
                input_embeddings = code_embs_proj
        
        # Forward through the model
        hidden_states = self.prior_model(
            inputs_embeds=input_embeddings, 
            use_cache=False, 
            output_hidden_states=True
        ).hidden_states[-1].to(torch.float)
        
        # Project and compute logits
        pred_embeddings = self.proj_out(hidden_states)
        logits = torch.matmul(pred_embeddings, self.codebook.T).float()
        
        return logits, None



def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, cond: torch.Tensor, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=cond, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=cond, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model, cond: torch.Tensor, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cond, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs


@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, num_classes=1000, **sampling_kwargs):
    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond
    T = 1

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    # with torch.device(device):
    #     max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
    #     model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    # if emb_masks is not None:
    #     assert emb_masks.shape[0] == max_batch_size
    #     assert emb_masks.shape[-1] == T
    #     if cfg_scale > 1.0:
    #         model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
    #     else:
    #         model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

    #     eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
    #     model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    # input_pos = torch.arange(0, T, device=device)
    # next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    # print(f'prefix next_token.shape: {next_token.shape}')
    # seq[:, T:T+1] = next_token

    next_token = torch.tensor([]).to(device)

    input_pos = torch.arange(0, T, device=device)
    generated_tokens, _ = decode_n_tokens(model, cond_combined, next_token, input_pos, max_new_tokens, cfg_scale, cfg_interval, **sampling_kwargs)
    print(f'generated_tokens[0].shape: {generated_tokens[0].shape}')
    seq[:, T:] = torch.cat(generated_tokens, dim=1)

    return seq[:, T:]


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_dir = os.path.join(args.results_dir, args.experiment_name)
    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    # Load config and merge with base
    config = OmegaConf.load(config_path)
    config = train_utils.restore_config(config)

    # Build model
    model = train_utils.build_model(config)
    model.eval()

    gpt_model = ARWrapper(model.quantizer)


    codebook = model.quantizer.quantizer.embedding.weight  # [V, D]
    vocab_size, e_dim = codebook.shape

    code_length = model.code_length  # e.g. 64
    context_dim = model.context_dim  # e.g. 56
    e_dim = model.config.model.codebook_size_for_entropy
    group_dim = model.config.model.codebook_size_for_entropy  # fg (e_dim)
    sub_tokens = context_dim // group_dim  # fh, should be integer (e.g. 4)
    seq_len = code_length * sub_tokens  # total number of tokens to generate (e.g. 256)


    # Labels to condition the model with (feel free to change):
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), seq_len, e_dim]

    t1 = time.time()
    index_sample = generate(
        gpt_model, c_indices, seq_len,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()

    z_q = model.quantizer.quantizer.get_codebook_entry(index_sample, shape=qzshape)
    # Rearrange to FlowMo code shape: (1, code_length, context_dim)
    print(f'z_q.shape: {z_q.shape}')
    quantized_code = rearrange(z_q, "b fg (t fh) -> b t (fg fh)", t=code_length, fh=sub_tokens)
    print(f'quantized_code.shape: {quantized_code.shape}')

    with torch.no_grad():
        img_size = config.data.image_size
        dummy_image = torch.zeros((len(class_labels), 3, img_size, img_size), device=device)
        samples = model.reconstruct(dummy_image, dtype=torch.float32, code=quantized_code)

    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample_{}.png".format(args.experiment_name), nrow=4, normalize=True, value_range=(-1, 1))
    print(f"image is saved to sample_{args.experiment_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results", help="Directory where experiment logs are stored.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment folder to load.")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Path to checkpoint to load; use 'auto' for latest inside experiment.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=62,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)
