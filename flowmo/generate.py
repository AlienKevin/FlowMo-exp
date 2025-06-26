import argparse
import os
import torch
from omegaconf import OmegaConf
from einops import rearrange
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F

# FlowMo imports
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
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


@torch.no_grad()
def sample_code(model: models.FlowMo, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0, random_indices: bool = False, class_idx: int = None, cfg_scale: float = 1.0, num_classes: int = 1000):
    """Autoregressively samples quantized code tokens using the prior model.
    
    Args:
        model: FlowMo model with quantizer and prior
        temperature: Temperature for sampling. If 0, uses greedy decoding. If > 0, uses stochastic sampling.
        random_indices: If True, generates completely random indices instead of using the prior model
        class_idx: If not None, conditions generation on this class index.
        cfg_scale: Classifier-Free Guidance scale. If > 1.0, mixes conditional and unconditional logits.
        num_classes: Number of classes for conditional generation. Used for CFG.
    """
    device = next(model.parameters()).device
    quantizer = model.quantizer  # LARPQuantizer
    base_quantizer = quantizer.quantizer  # IndexPropagationQuantize

    codebook = base_quantizer.embedding.weight  # [V, D]
    vocab_size, e_dim = codebook.shape

    code_length = model.code_length  # e.g. 64
    context_dim = model.context_dim  # e.g. 56
    group_dim = model.config.model.codebook_size_for_entropy  # fg (e_dim)
    sub_tokens = context_dim // group_dim  # fh, should be integer (e.g. 4)
    seq_len = code_length * sub_tokens  # total number of tokens to generate (e.g. 256)

    if random_indices:
        # Generate completely random indices
        generated_indices = torch.randint(0, vocab_size, (seq_len,), device=device).tolist()
    else:
        # Storage for generated indices
        generated_indices = []
        generated_probs = []

        do_cfg = cfg_scale > 1.0 and class_idx is not None
        
        prompt_embedding = None
        if class_idx is not None:
            if not hasattr(quantizer, 'prior_model_cls_embedding'):
                raise AttributeError("Quantizer does not have 'prior_model_cls_embedding' attribute for conditioning.")
            
            class_emb = quantizer.prior_model_cls_embedding(torch.tensor([class_idx], device=device), train=False) # (1, 1, D_class)
            prompt_embedding = class_emb.to(torch.bfloat16)

        uncond_prompt_embedding = None
        if do_cfg:
            uncond_class_emb = quantizer.prior_model_cls_embedding(torch.tensor([num_classes], device=device), train=False)
            uncond_prompt_embedding = uncond_class_emb.to(torch.bfloat16)

        for _ in range(seq_len):
            # Build input embeddings tensor
            code_embeddings = codebook[torch.tensor(generated_indices, device=device)].unsqueeze(0) if generated_indices else None

            # Qwen prior
            proj_in = quantizer.prior_model_project_in
            proj_out = quantizer.prior_model_project_out
            code_embeddings_proj = proj_in(code_embeddings).to(torch.bfloat16) if code_embeddings is not None else None

            if do_cfg:
                prompts = torch.cat([prompt_embedding, uncond_prompt_embedding], dim=0)

                if code_embeddings_proj is not None:
                    code_embeddings_batch = code_embeddings_proj.repeat(2, 1, 1)
                    input_embeddings = torch.cat([prompts, code_embeddings_batch], dim=1)
                else:
                    input_embeddings = prompts
                
                hidden = quantizer.prior_model(inputs_embeds=input_embeddings, output_hidden_states=True).hidden_states[-1].to(torch.float)
                pred_embedding = proj_out(hidden[:, -1, :])

                logits_batch = torch.matmul(pred_embedding, codebook.T)
                logits_cond, logits_uncond = logits_batch.chunk(2)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                input_parts = []
                if prompt_embedding is not None:
                    input_parts.append(prompt_embedding)
                if code_embeddings_proj is not None:
                    input_parts.append(code_embeddings_proj)
                input_embeddings = torch.cat(input_parts, dim=1)
                
                hidden = quantizer.prior_model(inputs_embeds=input_embeddings, output_hidden_states=True).hidden_states[-1].to(torch.float)
                pred_embedding = proj_out(hidden[:, -1, :])  # (1, D)

                # Sample next token based on temperature
                logits = torch.matmul(pred_embedding, codebook.T)  # (1, V)
            
            logits = logits.float()

            next_idx, next_probs = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, sample_logits=temperature > 0)
            next_idx = next_idx.item()

            generated_indices.append(next_idx)
            generated_probs.append(next_probs)

    # Convert indices back to quantized embeddings (1, e_dim, seq_len)
    indices_tensor = torch.tensor(generated_indices, device=device).long()

    print(f'indices_tensor: {indices_tensor.tolist()}')

    z_q = base_quantizer.get_codebook_entry(indices_tensor, shape=(1, seq_len, e_dim))

    # Rearrange to FlowMo code shape: (1, code_length, context_dim)
    print(f'z_q.shape: {z_q.shape}')
    quantized_code = rearrange(z_q, "b fg (t fh) -> b t (fg fh)", t=code_length, fh=sub_tokens)
    print(f'quantized_code.shape: {quantized_code.shape}')

    # Calculate perplexity of generated tokens
    if generated_probs is not None and len(generated_probs) > 0:
        # Stack all probability tensors
        all_probs = torch.stack(generated_probs, dim=0).squeeze(1)  # Shape: [seq_len, vocab_size]
        generated_tokens_tensor = torch.tensor(generated_indices, device=device)  # Shape: [seq_len]
        
        # Get the probabilities of the actual generated tokens
        token_probs = all_probs[torch.arange(len(generated_indices)), generated_tokens_tensor]

        # Print probability of each selected token
        print(f'Token probabilities:')
        for token_idx in range(len(generated_indices)):
            token_id = generated_indices[token_idx]
            token_prob = token_probs[token_idx].item()
            print(f'  Token {token_idx}: ID={token_id}, Prob={token_prob:.6f}')
        
        # Count tokens in top 10 for each position
        top_10_count = 0
        for token_idx in range(len(generated_indices)):
            # Get top 10 probabilities for this position
            top_10_probs, top_10_indices = torch.topk(all_probs[token_idx], k=10)
            selected_token = generated_tokens_tensor[token_idx]
            
            # Check if selected token is in top 10
            if selected_token in top_10_indices:
                top_10_count += 1
        
        print(f'{top_10_count}/{len(generated_indices)} tokens in top 10 ({100 * top_10_count / len(generated_indices):.2f}%)')
        
        # Calculate log probabilities and perplexity
        log_probs = torch.log(token_probs + 1e-10)  # Add small epsilon to avoid log(0)
        avg_log_prob = log_probs.mean()
        perplexity = torch.exp(-avg_log_prob)
        
        print(f'Generated tokens perplexity: {perplexity.item():.4f}')
    else:
        print('No probabilities available for perplexity calculation')

    return quantized_code  # shape: (1, code_length, context_dim)


@torch.no_grad()
def calculate_gt_likelihood(model: models.FlowMo, gt_indices: torch.Tensor, class_idx: int = None):
    """Calculates the negative log-likelihood of ground truth tokens using the prior model."""
    device = next(model.parameters()).device
    quantizer = model.quantizer
    base_quantizer = quantizer.quantizer
    codebook = base_quantizer.embedding.weight
    proj_in = quantizer.prior_model_project_in
    proj_out = quantizer.prior_model_project_out
    
    gt_indices_batch = gt_indices.unsqueeze(0)
    seq_len = gt_indices_batch.shape[1]

    targets = gt_indices_batch

    # Prepare input for the prior model
    prompt_embedding = None
    if not hasattr(quantizer, 'prior_model_cls_embedding'):
        raise AttributeError("Quantizer does not have 'prior_model_cls_embedding' for conditioning.")
    
    class_emb = quantizer.prior_model_cls_embedding(torch.tensor([class_idx], device=device), train=False)
    prompt_embedding = class_emb.to(torch.bfloat16)
    
    # Input is [class_emb, token_0, ..., token_{N-2}]
    # We predict all tokens from 0 to N-1
    code_embs = codebook[gt_indices_batch[:, :-1]]
    code_embs_proj = proj_in(code_embs).to(torch.bfloat16)
    model_input = torch.cat([prompt_embedding, code_embs_proj], dim=1)
    
    # Single forward pass
    hidden_states = quantizer.prior_model(inputs_embeds=model_input, use_cache=False, output_hidden_states=True).hidden_states[-1].to(torch.float)
    pred_embeddings = proj_out(hidden_states)
    
    logits = torch.matmul(pred_embeddings, codebook.T)
    
    # Calculate per-token loss
    per_token_nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')
    nll_loss = per_token_nll.sum()
    
    avg_nll = nll_loss / seq_len
    perplexity = torch.exp(avg_nll)
    
    per_token_prob = torch.exp(-per_token_nll)
    avg_prob = per_token_prob.mean()

    print(f"Ground Truth Likelihood Calculation:")
    print(f"  Total Negative Log-Likelihood: {nll_loss.item():.4f}")
    print(f"  Average NLL per token: {avg_nll.item():.4f}")
    print(f"  Perplexity: {perplexity.item():.4f}")
    print(f"  Average GT Token Probability: {avg_prob.item():.4f}")
    # Calculate how many GT tokens are in top-10 predictions
    top10_indices = torch.topk(logits.reshape(-1, logits.size(-1)), k=10, dim=-1).indices
    gt_tokens_flat = targets.reshape(-1)
    
    gt_in_top10_count = 0
    for i, gt_token in enumerate(gt_tokens_flat):
        if gt_token in top10_indices[i]:
            gt_in_top10_count += 1
    
    print(f"  GT tokens in top-10 predictions: {gt_in_top10_count}/{seq_len} ({100.0 * gt_in_top10_count / seq_len:.1f}%)")

    print("\nPer-token Ground Truth Probabilities:")
    for i, prob in enumerate(per_token_prob):
        print(f"  Token {i:3d} (ID: {gt_indices[i]:4d}): {prob.item():.4f}")


def save_image(tensor_image: torch.Tensor, path: str):
    """Converts a [-1,1] ranged CHW tensor to a PNG file."""
    tensor_image = tensor_image.detach().cpu().clamp(-1, 1)
    tensor_image = (tensor_image + 1) / 2  # [0,1]
    pil_image = T.ToPILImage()(tensor_image)
    pil_image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate an image using a trained FlowMo LARP_IBQ model.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory where experiment logs are stored.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment folder to load.")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Path to checkpoint to load; use 'auto' for latest inside experiment.")
    parser.add_argument("--output", type=str, default="generation.png", help="Filename for the generated image.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Use 0 for greedy decoding, >0 for stochastic sampling (default: 1.0).")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--random-indices", action="store_true", help="Generate completely random indices instead of using the prior model")
    parser.add_argument("--class-idx", type=int, help="Class index for conditional generation.")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Classifier-Free Guidance scale. Use > 1.0 for CFG.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes for CFG.")
    parser.add_argument("--force-gt-from-image", type=str, help="Path to an image to force ground truth indices from and calculate likelihood.")
    args = parser.parse_args()

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

    # Figure out checkpoint
    if args.checkpoint == "auto":
        ckpt = find_latest_checkpoint(exp_dir)
    else:
        ckpt = args.checkpoint
    train_utils.restore_from_ckpt(model, None, ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.force_gt_from_image:
        img_path = args.force_gt_from_image
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found at {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        
        img_size = config.data.image_size
        transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
        ])
        image_tensor = transform(img).unsqueeze(0).to(device)
        # Normalize to [-1, 1] range as expected by the model
        image_tensor = (image_tensor * 2.0) - 1.0
        
        # Save the preprocessed image
        input_basename = os.path.basename(img_path)
        input_name, _ = os.path.splitext(input_basename)
        preprocessed_filename = f"{input_name}_preprocessed.jpeg"
        
        save_image(image_tensor[0], preprocessed_filename)
        print(f"Preprocessed image saved to: {preprocessed_filename}")

        with torch.no_grad():
            code, _ = model.encode(image_tensor)
            
            b, t, f = code.shape
            code_reshaped = rearrange(
                code,
                "b t (fg fh) -> b fg (t fh)",
                fg=model.config.model.codebook_size_for_entropy,
            )
            
            _, _, indices = model.quantizer.quantizer(code_reshaped)
            gt_indices = indices.squeeze(0)
        
        print(f'Ground truth indices: {gt_indices}')

        calculate_gt_likelihood(model, gt_indices, class_idx=args.class_idx)

        # Reconstruct and save the image from ground truth indices
        print("\nReconstructing image from ground truth indices to verify tokenizer...")
        
        quantizer = model.quantizer
        base_quantizer = quantizer.quantizer
        
        code_length = model.code_length
        context_dim = model.context_dim
        e_dim = model.config.model.codebook_size_for_entropy
        sub_tokens = context_dim // e_dim
        seq_len = code_length * sub_tokens

        # Get quantized code from ground truth indices
        z_q = base_quantizer.get_codebook_entry(gt_indices, shape=(1, seq_len, e_dim))
        quantized_code_from_gt = rearrange(z_q, "b fg (t fh) -> b t (fg fh)", t=code_length, fh=sub_tokens).to(device)

        # Reconstruct image
        with torch.no_grad():
            img_size = config.data.image_size
            dummy_image = torch.zeros((1, 3, img_size, img_size), device=device)
            reconstructed_image = model.reconstruct(dummy_image, dtype=torch.float32, code=quantized_code_from_gt)
        
        # Save reconstructed image
        input_basename = os.path.basename(img_path)
        input_name, _ = os.path.splitext(input_basename)
        output_filename = f"{input_name}_reconstructed.jpeg"
        
        save_image(reconstructed_image[0], output_filename)
        print(f"Reconstructed image saved to: {output_filename}")
        
        return

    # Sample code tokens with specified temperature
    quantized_code = sample_code(model, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, random_indices=args.random_indices, class_idx=args.class_idx, cfg_scale=args.cfg_scale, num_classes=args.num_classes).to(device)

    # Dummy image to provide spatial dimensions (content ignored when code provided)
    img_size = config.data.image_size
    dummy = torch.zeros((1, 3, img_size, img_size), device=device)

    # Run reconstruction / decoding pipeline
    with torch.no_grad():
        generated = model.reconstruct(dummy, dtype=torch.float32, code=quantized_code)  # (1,3,H,W)

    save_image(generated[0], args.output)
    print(f"Image saved to {args.output} (temperature: {args.temperature}, cfg_scale: {args.cfg_scale})")


if __name__ == "__main__":
    main()
