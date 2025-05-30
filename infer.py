import torch
from omegaconf import OmegaConf
from flowmo import train_utils
import mediapy, einops

# Set up the data.
config = OmegaConf.load('flowmo/configs/base.yaml')
config.data.batch_size = 4
config.data.num_workers = 0
config.data.image_size = 64

torch.manual_seed(3)
val_dataloader = train_utils.load_dataset(config, 'val', shuffle_val=True)
batch = next(train_utils.wrap_dataloader([next(iter(val_dataloader))]))

images = batch['image']
# mediapy.show_image(einops.rearrange(images.cpu()/2+.5, "b c h w -> h (b w) c"), vmin=0, vmax=1)
print('Loaded images')

# Choose your model
model_name = "flowmo_hi_larp_no_prior_loss_64x64_pretrain"

# The low BPP model has 18 bits per token, the high bitrate model has 56 bits per token.
zoo = {
    "flowmo_hi_larp_qwen3_0.6b_rand_64x64_pretrain":
        {"prior.model_name": "Qwen3-0.6B", "context_dim": 56, "codebook_size_for_entropy": 14, "patch_size": 4, "mup_width": 4, "code_length": 64, "quantization_type": "larp", "ckpt_iteration": 100000},
    "flowmo_hi_larp_qwen3_0.6b_64x64_pretrain":
        {"prior.model_name": "Qwen3-0.6B", "context_dim": 56, "codebook_size_for_entropy": 14, "patch_size": 4, "mup_width": 4, "code_length": 64, "quantization_type": "larp", "ckpt_iteration": 60000},
    "flowmo_hi_larp_no_prior_loss_64x64_pretrain":
        {"prior.model_name": "gptc-S", "context_dim": 56, "codebook_size_for_entropy": 14, "patch_size": 4, "mup_width": 4, "code_length": 64, "quantization_type": "larp", "ckpt_iteration": 200000},
    "flowmo_hi_lfq_64x64_pretrain": {"context_dim": 56, "patch_size": 4, "mup_width": 4, "code_length": 64, "ckpt_iteration": 200000},
}
state_dict = torch.load(f"results/{model_name}/checkpoints/{zoo[model_name]['ckpt_iteration']:08d}.pth", map_location='cuda')
config.model.context_dim = zoo[model_name]['context_dim']
if 'patch_size' in zoo[model_name]:
    config.model.patch_size = zoo[model_name]['patch_size']
if 'mup_width' in zoo[model_name]:
    config.model.mup_width = zoo[model_name]['mup_width']
if 'code_length' in zoo[model_name]:
    config.model.code_length = zoo[model_name]['code_length']
if 'quantization_type' in zoo[model_name]:
    config.model.quantization_type = zoo[model_name]['quantization_type']
if 'prior.model_name' in zoo[model_name]:
    config.prior.model_name = zoo[model_name]['prior.model_name']
if 'codebook_size_for_entropy' in zoo[model_name]:
    config.model.codebook_size_for_entropy = zoo[model_name]['codebook_size_for_entropy']
else:
    config.model.codebook_size_for_entropy = 1  # don't need this at test time.

model = train_utils.build_model(config)

missing_keys, unexpected_keys = model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
print(f'Missing keys: {missing_keys}')
print(f'Unexpected keys: {unexpected_keys}')

# Reconstruct the images. If you want to use your own images, FlowMo accepts
# images in [-1, 1] in bchw format.

model.train()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    images_rec, _ = model.reconstruct(images)

# Show the original and reconstructed.
image_name = f"{model_name}_{zoo[model_name]['ckpt_iteration']:08d}.png"
mediapy.write_image(image_name, einops.rearrange(images_rec.cpu().numpy()/2+.5, "b c h w -> h (b w) c"))
