import torch
from omegaconf import OmegaConf
from flowmo import train_utils
import mediapy, einops
from tqdm import tqdm

# Choose your model
model_name = "dogs_flowmo_lo_c2i_larp_sg_ibq_128x128_pretrain"

zoo = {
    "flowmo_hi_larp_qwen3_0.6b_rand_64x64_pretrain": 100000,
    "flowmo_hi_larp_qwen3_0.6b_64x64_pretrain": 60000,
    "flowmo_hi_larp_no_prior_loss_64x64_pretrain": 200000,
    "flowmo_hi_lfq_64x64_pretrain": 200000,
    "flowmo_hi_c2i_larp_ibq_prior_0.001_multiplier_10_64x64_pretrain": 360000,
    "flowmo_hi_c2i_larp_ibq_rand_prior_0.001_multiplier_10_64x64_pretrain": 360000,
    "flowmo_hi_c2i_larp_ibq_sg_prior_0.001_multiplier_10_64x64_pretrain": 360000,
    "flowmo_hi_c2i_larp_ibq_rand_sg_prior_0.001_multiplier_10_64x64_pretrain": 380000,
    "flowmo_lo_c2i_larp_ibq_rand_128x128_pretrain": 120000,
    "flowmo_lo_ibq_128x128_pretrain": 150000,
    "dogs_flowmo_lo_c2i_larp_ibq_128x128_pretrain": 150000,
    "dogs_flowmo_lo_c2i_larp_ibq_rand_128x128_pretrain": 150000,
    "dogs_flowmo_lo_c2i_larp_ibq_rand_sg_128x128_pretrain": 150000,
    "dogs_flowmo_lo_c2i_larp_sg_ibq_128x128_pretrain": 150000,
}

# Set up the data.
config = OmegaConf.load(f'results/{model_name}/config.yaml')
config.data.batch_size = 4
config.data.num_workers = 0
config.data.image_size = 128

# torch.manual_seed(3)
# val_dataloader = train_utils.load_dataset(config, 'val', shuffle_val=True)
# batch = next(train_utils.wrap_dataloader([next(iter(val_dataloader))]))

import os
from PIL import Image
import torchvision.transforms as transforms

# Load all images from ./dog_pics/
dog_pics_dir = "./dog_pics/"
image_files = [f for f in os.listdir(dog_pics_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Create transform to match the expected format
transform = transforms.Compose([
    transforms.Resize((config.data.image_size, config.data.image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)  # Scale from [0, 1] to [-1, 1]
])

# Load and process images
images_list = []
for img_file in image_files:
    img_path = os.path.join(dog_pics_dir, img_file)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    images_list.append(img_tensor)

# Stack into batch
if images_list:
    images = torch.stack(images_list).cuda()
    batch = {'image': images}
else:
    raise ValueError("No images found in ./dog_pics/ directory")


images = batch['image']
# mediapy.show_image(einops.rearrange(images.cpu()/2+.5, "b c h w -> h (b w) c"), vmin=0, vmax=1)
print('Loaded images')

import os
samples_dir = f"samples/{model_name}"
os.makedirs(samples_dir, exist_ok=True)

for ckpth_iteration in tqdm([zoo[model_name]]):
    state_dict = torch.load(f"results/{model_name}/checkpoints/{ckpth_iteration:08d}.pth", map_location='cuda')

    model = train_utils.build_model(config)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
    print(f'Missing keys: {missing_keys}')
    print(f'Unexpected keys: {unexpected_keys}')

    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        images_rec, _, _ = model.reconstruct(images)

    image_name = f"{samples_dir}/{ckpth_iteration:08d}.png"
    mediapy.write_image(image_name, einops.rearrange(images_rec.cpu().numpy()/2+.5, "b c h w -> h (b w) c"))
