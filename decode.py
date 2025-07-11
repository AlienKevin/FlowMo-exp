import torch
import json
import os
from omegaconf import OmegaConf
from flowmo import train_utils
from tqdm import tqdm
import torchvision
import einops

def decode_images():
    # Choose your model
    model_name = "flowmo_lo"
    encoded_file = f"encoded_tokens_{model_name}.json"
    output_dir = "decoded_image_samples"

    zoo = {
        "flowmo_lo": 1325000,
    }

    # Set up the config
    config_path = f'results/{model_name}/config.yaml'
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return
        
    config = OmegaConf.load(config_path)
    
    # Load the model
    ckpth_iteration = zoo[model_name]
    checkpoint_path = f"results/{model_name}/checkpoints/{ckpth_iteration:08d}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found at {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location='cuda')
    
    model = train_utils.build_model(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
    print(f'Missing keys: {missing_keys}')
    print(f'Unexpected keys: {unexpected_keys}')
    
    model.eval()
    model.cuda()
    
    # Load the encoded tokens
    if not os.path.exists(encoded_file):
        print(f"Encoded tokens file not found at {encoded_file}")
        return
    with open(encoded_file, 'r') as f:
        encoded_tokens = json.load(f)
    
    print(f"Decoding {len(encoded_tokens)} images...")
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for image_name, token_ids_list in tqdm(encoded_tokens.items(), desc="Decoding images"):
            class_id = image_name.split('_')[0]
            class_dir = os.path.join(output_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            
            token_ids = torch.LongTensor(token_ids_list).cuda().unsqueeze(0)
            
            # Convert token_ids to code
            code_length = model.code_length
            
            # Decode token ids to get the quantized vectors (-1 or 1).
            # The shape will be (batch_size, num_tokens, quantizer_dim)
            decoded_code = model.quantizer.decode(token_ids)
            
            # Reshape to match the structure before the final rearrange in _quantize
            # The shape of `quantized` in _quantize after quantizer call is (b, d, ...)
            # so we permute (b, n, d) to (b, d, n)
            decoded_code = decoded_code.permute(0, 2, 1)

            # Rearrange back to (b, code_length, context_dim)
            # 'b d (t fh) -> b t (d fh)'
            # b = batch size (1)
            # d = quantizer_dim
            # t = code_length
            # fh = (num_tokens / code_length) = (context_dim / quantizer_dim)
            code = einops.rearrange(
                decoded_code,
                'b d (t fh) -> b t (d fh)',
                t=code_length
            )

            # Reconstruct image from code
            reconstructed_image = model.reconstruct(images=torch.zeros(1, 3, config.data.image_size, config.data.image_size).cuda(), code=code)
            
            # Save the image
            output_path = os.path.join(class_dir, image_name)
            
            # Post-process and save
            reconstructed_image = (reconstructed_image + 1.0) / 2.0  # Denormalize from [-1, 1] to [0, 1]
            reconstructed_image.clamp_(0, 1)
            torchvision.utils.save_image(reconstructed_image, output_path)

    print(f"Decoded images saved to {output_dir}")

if __name__ == "__main__":
    decode_images()
