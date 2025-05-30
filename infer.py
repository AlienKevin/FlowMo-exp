import torch
from omegaconf import OmegaConf
from flowmo import train_utils
import mediapy, einops
from tqdm import tqdm

# Choose your model
model_name = "flowmo_hi_lfq_64x64_pretrain"

zoo = {
    "flowmo_hi_larp_qwen3_0.6b_rand_64x64_pretrain": 100000,
    "flowmo_hi_larp_qwen3_0.6b_64x64_pretrain": 60000,
    "flowmo_hi_larp_no_prior_loss_64x64_pretrain": 200000,
    "flowmo_hi_lfq_64x64_pretrain": 200000,
}

# Set up the data.
config = OmegaConf.load(f'results/{model_name}/config.yaml')
config.data.batch_size = 4
config.data.num_workers = 0
config.data.image_size = 64

torch.manual_seed(3)
val_dataloader = train_utils.load_dataset(config, 'val', shuffle_val=True)
batch = next(train_utils.wrap_dataloader([next(iter(val_dataloader))]))

images = batch['image']
# mediapy.show_image(einops.rearrange(images.cpu()/2+.5, "b c h w -> h (b w) c"), vmin=0, vmax=1)
print('Loaded images')

import os
samples_dir = f"samples/{model_name}"
os.makedirs(samples_dir, exist_ok=True)

for ckpth_iteration in tqdm(range(20000, 200001, 20000)):
    state_dict = torch.load(f"results/{model_name}/checkpoints/{ckpth_iteration:08d}.pth", map_location='cuda')

    model = train_utils.build_model(config)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
    print(f'Missing keys: {missing_keys}')
    print(f'Unexpected keys: {unexpected_keys}')

    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        images_rec, _ = model.reconstruct(images)

    image_name = f"{samples_dir}/{ckpth_iteration:08d}.png"
    mediapy.write_image(image_name, einops.rearrange(images_rec.cpu().numpy()/2+.5, "b c h w -> h (b w) c"))
