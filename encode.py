import torch
import json
import os
from omegaconf import OmegaConf
from flowmo import train_utils
from flowmo.data import IndexedTarDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def encode_imagenet():
    # Choose your model
    model_name = "flowmo_lo"
    
    zoo = {
        "flowmo_lo": 1325000,
    }
    
    # Set up the config
    config = OmegaConf.load(f'results/{model_name}/config.yaml')
    config.data.batch_size = 256  # Adjust batch size as needed
    config.data.num_workers = 4
    
    # Load the model
    ckpth_iteration = zoo[model_name]
    state_dict = torch.load(f"results/{model_name}/checkpoints/{ckpth_iteration:08d}.pth", map_location='cuda')
    
    model = train_utils.build_model(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
    print(f'Missing keys: {missing_keys}')
    print(f'Unexpected keys: {unexpected_keys}')
    
    model.eval()
    model.cuda()
    model = torch.compile(model)
    
    # Set up the dataset
    dataset = IndexedTarDataset(
        imagenet_tar=config.data.imagenet_train_tar,
        imagenet_index=config.data.imagenet_train_index,
        size=config.data.image_size,
        random_crop=False,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # Dictionary to store encoded tokens
    encoded_tokens = {}
    
    print(f"Encoding {len(dataset)} images...")
    
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding batches")):
            images = batch['image'].cuda()
            images = images.permute(0, 3, 1, 2)
            # print(f'images.size(): {images.size()}')
            
            # Get the image names for this batch
            start_idx = batch_idx * config.data.batch_size
            end_idx = min(start_idx + config.data.batch_size, len(dataset))
            
            # Encode the images to get token IDs
            bs, c, h, w = images.shape
            x = images.cuda()
            prequantized_code = model.encode(x)[0].cuda()
            dummy_cond = 0
            dummy_caption = []
            code, token_ids, _ = model._quantize(prequantized_code, dummy_cond, dummy_caption)
            # print(f'code.size(): {code.size()}')
            # print(f'token_ids.size(): {token_ids.size()}')
            token_ids = token_ids.view((bs, -1))
            # print(f'token_ids.size(): {token_ids.size()}')
            
            # Store the token IDs for each image in the batch
            for i in range(token_ids.size(0)):
                if start_idx + i < len(dataset):
                    image_info = dataset.index[start_idx + i]
                    image_name = image_info['name'].split('/')[-1]  # Get just the JPEG filename
                    encoded_tokens[image_name] = token_ids[i].cpu().numpy().tolist()

            if (batch_idx % 1000 == 0) or (batch_idx >= len(dataloader) - 1):
                # Save the encoded tokens to JSON
                output_file = f"encoded_tokens_{model_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(encoded_tokens, f)
    
    print(f"Encoded tokens saved to {output_file}")
    print(f"Total images encoded: {len(encoded_tokens)}")

if __name__ == "__main__":
    encode_imagenet()
