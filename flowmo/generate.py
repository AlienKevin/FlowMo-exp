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
def greedy_sample_code(model: models.FlowMo):
    """Autoregressively samples quantized code tokens using the prior model with greedy decoding."""
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

    # Storage for generated indices
    generated_indices = []

    # Initialise first token randomly for diversity (could also be fixed to 0)
    first_idx = torch.randint(0, vocab_size, (1,), device=device).item()
    generated_indices.append(first_idx)

    # Convenience projections (for Qwen-style priors)
    use_qwen = hasattr(quantizer, 'prior_model_project_in') and hasattr(quantizer, 'prior_model_project_out')

    for _ in range(1, seq_len):
        # Build input embeddings tensor (1, current_len, e_dim)
        input_embeddings = codebook[torch.tensor(generated_indices, device=device)].unsqueeze(0)  # (1, t, D)

        if use_qwen:
            # Qwen prior
            proj_in = quantizer.prior_model_project_in
            proj_out = quantizer.prior_model_project_out
            hidden = quantizer.prior_model(inputs_embeds=proj_in(input_embeddings), output_hidden_states=True).hidden_states[-1]
            pred_embedding = proj_out(hidden[:, -1, :])  # (1, D)
        else:
            # GPT-C style prior
            pred_sequence = quantizer.prior_model(input_embeddings)  # (1, t, D)
            pred_embedding = pred_sequence[:, -1, :]  # (1, D)

        # Greedy decoding: choose codebook entry with maximum dot-product similarity
        logits = torch.matmul(pred_embedding, codebook.T)  # (1, V)
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

    # Sample code tokens greedily
    quantized_code = greedy_sample_code(model).to(device)

    # Dummy image to provide spatial dimensions (content ignored when code provided)
    img_size = config.data.image_size
    dummy = torch.zeros((1, 3, img_size, img_size), device=device)

    # Run reconstruction / decoding pipeline
    with torch.no_grad():
        generated = model.reconstruct(dummy, dtype=torch.float32, code=quantized_code)  # (1,3,H,W)

    save_image(generated[0], args.output)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
