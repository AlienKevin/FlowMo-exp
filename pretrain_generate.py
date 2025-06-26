import argparse
import os
import torch
import json
from omegaconf import OmegaConf
import torchvision
from einops import rearrange
from transformers import GPT2LMHeadModel, LogitsProcessor
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from flowmo import train_utils


class FilterLogitsProcessor(LogitsProcessor):
    def __init__(self, filter_vocab_size: int):
        self.filter_vocab_size = filter_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.filter_vocab_size:] = -float("Inf")
        return scores


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


def load_sft_model(ckpt_path, device):
    """Loads a saved SFT model."""
    model = GPT2LMHeadModel.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model, 
    class_idxs, 
    num_visual_tokens, 
    num_class_tokens, 
    cfg_scale=1.0, 
    temperature=1.0, 
    top_k=0, 
    top_p=1.0
):
    """Autoregressively generates visual tokens with CFG."""
    device = next(model.parameters()).device
    batch_size = len(class_idxs)
    
    cond_prompts = [[num_visual_tokens + c] for c in class_idxs]
    uncond_prompts = [[num_visual_tokens + num_class_tokens]] * batch_size
    
    cond_tokens = torch.tensor(cond_prompts, device=device)
    uncond_tokens = torch.tensor(uncond_prompts, device=device)
    
    seq_len = model.config.n_ctx - 1
    
    logits_processor = FilterLogitsProcessor(num_visual_tokens)

    for _ in range(seq_len):
        if cfg_scale > 1.0:
            input_ids = torch.cat([cond_tokens, uncond_tokens])
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            
            logits_cond, logits_uncond = logits.chunk(2)
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
        else:
            input_ids = cond_tokens
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]

        logits = logits_processor(None, logits)
        
        logits = logits / max(temperature, 1e-5)
        
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        cond_tokens = torch.cat([cond_tokens, next_token], dim=1)
        if cfg_scale > 1.0:
            uncond_tokens = torch.cat([uncond_tokens, next_token.clone()], dim=1)

    return cond_tokens[:, 1:]


def save_image(tensor_image: torch.Tensor, path: str):
    """Converts a [-1,1] ranged CHW tensor to a PNG file."""
    tensor_image = tensor_image.detach().cpu().clamp(-1, 1)
    tensor_image = (tensor_image + 1) / 2
    pil_image = T.ToPILImage()(tensor_image)
    pil_image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate images from a pretrained SFT model.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the SFT model checkpoint directory.")
    parser.add_argument("--class-idxs", nargs='+', type=int, required=True, help="List of class indices for conditional generation.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Filename for the generated image.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-Free Guidance scale. Use > 1.0 for CFG.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k filtering.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) filtering.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # These are based on pretrain.py and need to be consistent
    num_visual_tokens = 512

    # Recreate class_to_idx to get num_class_tokens
    with open('encoded_tokens_dogs_flowmo_lo_c2i_larp_ibq_rand_sg_128x128_pretrain.json', 'r') as f:
        data = json.load(f)
    items = [{'image_name': k} for k in data.keys()]
    class_data = {}
    for item in items:
        class_id = item['image_name'].split('_')[0]
        if class_id not in class_data:
            class_data[class_id] = []
        class_data[class_id].append(item)
    unique_classes = sorted(class_data.keys())
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    num_class_tokens = len(class_to_idx)
    
    model = load_sft_model(args.ckpt_path, device)
    
    decoder_model_name = "dogs_flowmo_lo_c2i_larp_ibq_rand_sg_128x128_pretrain"
    decoder_ckpth_iteration = 150000
    config_path = f'results/{decoder_model_name}/config.yaml'
    decoder_config = OmegaConf.load(config_path)
    checkpoint_path = f"results/{decoder_model_name}/checkpoints/{decoder_ckpth_iteration:08d}.pth"
    
    decoder_model = train_utils.build_model(decoder_config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    decoder_model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
    decoder_model.eval()
    decoder_model.to(device)

    print("Generating visual tokens...")
    visual_tokens = generate(
        model,
        args.class_idxs,
        num_visual_tokens,
        num_class_tokens,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    print("Decoding tokens to image...")
    code_length = decoder_config.model.code_length
    context_dim = decoder_config.model.context_dim
    codebook_size_for_entropy = decoder_config.model.codebook_size_for_entropy
    fh = context_dim // codebook_size_for_entropy
    seq_len = code_length * fh

    indices = visual_tokens
    total_images_in_batch = indices.shape[0]
    shape = (total_images_in_batch, seq_len, codebook_size_for_entropy)
    
    quantized = decoder_model.quantizer.quantizer.get_codebook_entry(indices, shape)
    code = rearrange(quantized, "b fg (t fh) -> b t (fg fh)", t=code_length, fh=fh)

    reconstructed_images = decoder_model.reconstruct(images=torch.zeros(total_images_in_batch, 3, decoder_config.data.image_size, decoder_config.data.image_size, device=device), code=code)
    
    torchvision.utils.save_image(reconstructed_images, args.output, nrow=len(args.class_idxs), normalize=True)
    print(f"Images saved to {args.output}")


if __name__ == "__main__":
    main()
