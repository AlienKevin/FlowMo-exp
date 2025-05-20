import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
import torch
from flowmo import train_utils, models
from omegaconf import OmegaConf

# Load the processed dataset
model_name = 'flowmo_lfq_qwen_hi_all_sg_50xlr_bce_0.01_pretrain'
# model_name = 'flowmo_lfq_qwen_hi_targets_sg_50xlr_bce_0.006_pretrain'
# model_name = 'flowmo_hi'
dataset_path = f"../infomec/outputs/processed_shapes3d_{model_name}"
processed_shapes3d = load_from_disk(dataset_path)

# Set up the FlowMo model
config = OmegaConf.load('flowmo/configs/base.yaml')
config.data.batch_size = 4
config.data.num_workers = 0

# Configure model based on the selected model
zoo = {
    "flowmo_lfq_qwen_hi_all_sg_50xlr_bce_0.01_pretrain": {
        "context_dim": 56, 
        "code_length": 256,
        "codebook_size_for_entropy": 14,
        "patch_size": 8, 
        "mup_width": 4, 
        "ckpt_path": "results/flowmo_lfq_qwen_hi_all_sg_50xlr_bce_0.01_pretrain/checkpoints/00200000.pth"
    },
    "flowmo_lfq_qwen_hi_targets_sg_50xlr_bce_0.006_pretrain": {
        "context_dim": 56, 
        "code_length": 256,
        "codebook_size_for_entropy": 14,
        "patch_size": 8, 
        "mup_width": 4, 
        "ckpt_path": "results/flowmo_lfq_qwen_hi_targets_sg_50xlr_bce_0.006_pretrain/checkpoints/00200000.pth"
    },
    "flowmo_hi": {
        "context_dim": 56, 
        "code_length": 256,
        "codebook_size_for_entropy": 14,
        "patch_size": 4, 
        "mup_width": 6, 
        "ckpt_path": "flowmo_hi.pth"
    }
}

# Load model configuration
if 'ckpt_path' in zoo[model_name]:
    state_dict = torch.load(zoo[model_name]['ckpt_path'], map_location='cuda')
config.model.context_dim = zoo[model_name]['context_dim']
if 'patch_size' in zoo[model_name]:
    config.model.patch_size = zoo[model_name]['patch_size']
if 'mup_width' in zoo[model_name]:
    config.model.mup_width = zoo[model_name]['mup_width']
if 'code_length' in zoo[model_name]:
    config.model.code_length = zoo[model_name]['code_length']
if 'quantization_type' in zoo[model_name]:
    config.model.quantization_type = zoo[model_name]['quantization_type']
config.model.codebook_size_for_entropy = 1  # don't need this at test time.

# Initialize the model
flowmo_model = train_utils.build_model(config)
# Filter out keys starting with 'qwen_model'
filtered_state_dict = {k: v for k, v in state_dict['model_ema_state_dict'].items() if not k.startswith('qwen_model')}
# Load the filtered state dict, ignoring missing keys (like the qwen_model ones)
flowmo_model.load_state_dict(filtered_state_dict, strict=False)
flowmo_model.eval()
flowmo_model.requires_grad_(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flowmo_model.to(device)

print(f"Dataset info: {processed_shapes3d}")
print(f"Number of examples: {len(processed_shapes3d)}")
print(f"Features: {processed_shapes3d.features}")

fig, axes = plt.subplots(5, 2, figsize=(18, 15))
fig.suptitle(f'Original vs Reconstructed Images (Model: {model_name})', fontsize=16)

def long_to_binary_tensor(x: torch.Tensor, num_bits: int = 64) -> torch.Tensor:
    """
    Converts a tensor of long integers into a binary bit tensor.

    Args:
        x (torch.Tensor): Input tensor of dtype torch.long.
        num_bits (int): Number of bits to represent each number (default: 64 for torch.long).

    Returns:
        torch.Tensor: Binary tensor of shape (x.shape[0], num_bits) with 0/1 values.
    """
    if x.dtype != torch.long:
        raise ValueError("Input tensor must be of dtype torch.long")
    
    # Create a bit mask of shape (num_bits,)
    bit_mask = 1 << torch.arange(num_bits - 1, -1, -1, dtype=torch.long, device=x.device)
    
    # Expand x to (len(x), num_bits) and apply bitwise AND then right shift
    binary_tensor = (x.unsqueeze(1) & bit_mask) > 0
    return binary_tensor.float()

for i in range(5):
    example = processed_shapes3d[i]
    
    # Original image
    axes[i, 0].imshow(example['image'])
    axes[i, 0].set_title(f"Original - Shape: {example['label_shape']}, Scale: {example['label_scale']}, Hue: {example['label_object_hue']}")
    axes[i, 0].axis('off')
    
    # # Stored reconstructed image
    toks = example['token_indices']
    # FlowMo reconstruction from tokens
    token_tensor = torch.tensor(toks, device=device)
    with torch.no_grad():
        # Reconstruct using FlowMo's reconstruct method with the stored tokens
        # Convert token indices to binary representation
        context_dim = config.model.context_dim
        code = (long_to_binary_tensor(x=token_tensor, num_bits=context_dim) * 2 - 1).unsqueeze(0)
        # print(token_tensor)
        # print(token_tensor.shape)
        # print(code)
        # print(code.shape)
        flowmo_recon = flowmo_model.reconstruct(
            images=torch.zeros(1, 3, 256, 256, device=device),  # Dummy input, will be ignored
            code=code
        )
        # Scale back to [0, 1] for display
        flowmo_recon = (flowmo_recon / 2.0) + 0.5
        flowmo_recon = torch.clamp(flowmo_recon, 0.0, 1.0)
    
    # Convert to numpy for display
    flowmo_recon_np = (flowmo_recon * 255.0).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    axes[i, 1].imshow(flowmo_recon_np)
    axes[i, 1].set_title(f"FlowMo Reconstruction - Tokens: {len(toks)}")
    axes[i, 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"visualization_{model_name}.png")
plt.show()

# Print token information for the first example
print(f"Token indices for first example (shape: {np.array(processed_shapes3d[0]['token_indices']).shape}):")
print(processed_shapes3d[0]['token_indices'][:10], "...")  # Show first 10 tokens
