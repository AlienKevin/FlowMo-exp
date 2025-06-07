import argparse
import os
import torch
from omegaconf import OmegaConf
from einops import rearrange
import torchvision.transforms as T

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


@torch.no_grad()
def sample_code(model: models.FlowMo, temperature: float = 0.0, random_indices: bool = False, class_idx: int = None, cfg_scale: float = 1.0, num_classes: int = 1000):
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
        
        # Autoregressively generate the remaining tokens
        num_to_gen = seq_len if class_idx is not None else seq_len - 1

        for _ in range(num_to_gen):
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
            
            if temperature > 0:
                # Stochastic sampling with temperature
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
            else:
                # Greedy decoding (temperature = 0)
                next_idx = torch.argmax(logits, dim=-1).item()
                
            generated_indices.append(next_idx)

    # Convert indices back to quantized embeddings (1, e_dim, seq_len)
    indices_tensor = torch.tensor(generated_indices, device=device).long()
    z_q = base_quantizer.get_codebook_entry(indices_tensor, shape=(1, seq_len, e_dim))

    # Rearrange to FlowMo code shape: (1, code_length, context_dim)
    quantized_code = rearrange(z_q, "b fg (t fh) -> b t (fg fh)", t=code_length, fh=sub_tokens)
    return quantized_code  # shape: (1, code_length, context_dim)


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
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling. Use 0 for greedy decoding, >0 for stochastic sampling (default: 1.0).")
    parser.add_argument("--random-indices", action="store_true", help="Generate completely random indices instead of using the prior model")
    parser.add_argument("--class-idx", type=int, help="Class index for conditional generation.")
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Classifier-Free Guidance scale. Use > 1.0 for CFG.")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes for CFG.")
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

    # Sample code tokens with specified temperature
    quantized_code = sample_code(model, temperature=args.temperature, random_indices=args.random_indices, class_idx=args.class_idx, cfg_scale=args.cfg_scale, num_classes=args.num_classes).to(device)

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
