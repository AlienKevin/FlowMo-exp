import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from omegaconf import OmegaConf
from flowmo import train_utils

def load_shapes3d_dataset(num_samples=10000):
    """
    Load the shapes3d dataset from huggingface and filter it to keep only rows
    where variables other than label_scale, label_shape, label_object_hue are 0.
    
    Returns:
        Filtered dataset
    """
    # Load the dataset from huggingface
    dataset = load_dataset("eurecom-ds/shapes3d")
    
    # # Define the columns we want to keep (others will be filtered to 0)
    # keep_columns = ['label_scale', 'label_shape', 'label_object_hue']
    
    # # Filter the dataset
    # def filter_fn(example):
    #     # Check if all variables not in keep_columns are 0
    #     for key in example:
    #         if key.startswith('label_') and key not in keep_columns and example[key] != 0:
    #             return False
    #     return True
    
    # # Apply the filter
    # filtered_dataset = dataset.filter(filter_fn, num_proc=8)

    # Get the train split
    train_dataset = dataset['train']
    
    # Take only the first num_samples
    filtered_dataset = train_dataset.select(range(min(num_samples, len(train_dataset))))
    
    print(f"Original dataset size: {len(train_dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    return filtered_dataset


# https://docs.pytorch.org/rl/0.6/_modules/torchrl/data/map/hash.html#BinaryToDecimal
class BinaryToDecimal(torch.nn.Module):
    """A Module to convert binaries encoded tensors to decimals.

    This is a utility class that allow to convert a binary encoding tensor (e.g. `1001`) to
    its decimal value (e.g. `9`)

    Args:
        num_bits (int): the number of bits to use for the bases table.
            The number of bits must be lower or equal to the input length and the input length
            must be divisible by ``num_bits``. If ``num_bits`` is lower than the number of
            bits in the input, the end result will be aggregated on the last dimension using
            :func:`~torch.sum`.
        device (torch.device): the device where inputs and outputs are to be expected.
        dtype (torch.dtype): the output dtype.
        convert_to_binary (bool, optional): if ``True``, the input to the ``forward``
            method will be cast to a binary input using :func:`~torch.heavyside`.
            Defaults to ``False``.

    Examples:
        >>> binary_to_decimal = BinaryToDecimal(
        ...    num_bits=4, device="cpu", dtype=torch.int32, convert_to_binary=True
        ... )
        >>> binary = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 10, 0]])
        >>> decimal = binary_to_decimal(binary)
        >>> assert decimal.shape == (2,)
        >>> assert (decimal == torch.Tensor([3, 2])).all()
    """

    def __init__(
        self,
        num_bits: int,
        device: torch.device,
        dtype: torch.dtype,
        convert_to_binary: bool = False,
    ):
        super().__init__()
        self.convert_to_binary = convert_to_binary
        self.bases = 2 ** torch.arange(num_bits - 1, -1, -1, device=device, dtype=dtype)
        self.num_bits = num_bits
        self.zero_tensor = torch.zeros((1,), device=device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        num_features = features.shape[-1]
        assert self.num_bits == num_features

        binary_features = (
            torch.heaviside(features, self.zero_tensor.to(features.dtype))
            if self.convert_to_binary
            else features
        )
        feature_parts = binary_features.reshape(shape=(-1, self.num_bits))
        digits = torch.vmap(torch.dot, (None, 0))(
            self.bases, feature_parts.to(self.bases.dtype)
        )
        # Reshape back to match input shape minus the last dimension
        return digits.reshape(shape=features.shape[:-1])


def tokenize_and_reconstruct(dataset):
    """
    Process a dataset of images through the FlowMo tokenizer and create a new dataset
    with original images, reconstructed images, token indices, and all original labels.
    
    Args:
        dataset: HuggingFace dataset containing images
        
    Returns:
        New dataset with added reconstructed images and tokens, preserving original data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    def process_batch(examples):
        batch_size = len(examples['image'])
        batch_token_indices = []

        binary_to_decimal = BinaryToDecimal(
            num_bits=config.model.context_dim, device='cpu', dtype=torch.int64, convert_to_binary=True
        )
        
        # Convert images to tensors
        batch_images = []
        for img in examples['image']:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            
            img = img.convert('RGB').resize((256, 256))
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1) / 255.0
            # Scale to [-1, 1] as FlowMo expects
            img_tensor = img_tensor * 2.0 - 1.0
            batch_images.append(img_tensor)
        
        # Stack images into a batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Process in batches to avoid OOM
        sub_batch_size = 32
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = batch_tensor[i:i+sub_batch_size]
            
            # Encode and decode with FlowMo
            with torch.no_grad():
                # Forward pass to get tokens
                x = sub_batch.cuda()
                prequantized_code = model.encode(x)[0].cuda()
                encoded_tokens, _, _ = model._quantize(prequantized_code)
                # Reshape encoded_tokens to [batch, code_length]
                # print(encoded_tokens[0][0])
                # print(encoded_tokens.shape)
                encoded_tokens = binary_to_decimal(encoded_tokens.to('cpu')).to(device)
                # print(encoded_tokens)
                # print(encoded_tokens.shape)
            
            batch_token_indices.extend(encoded_tokens.cpu().numpy().tolist())
        
        # Add new columns to examples while preserving original data
        result = {
            'token_indices': batch_token_indices
        }
        
        return result
    
    # Process the entire dataset, keeping all original columns
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=64,
        desc="Tokenizing and reconstructing images with FlowMo",
        remove_columns=None,
        load_from_cache_file=False,
    )
    
    return processed_dataset


if __name__ == '__main__':
    filtered_shapes3d = load_shapes3d_dataset()
    print(filtered_shapes3d)

    # Set up the FlowMo model
    # model_name = "flowmo_lfq_qwen_hi_all_sg_50xlr_bce_0.01_pretrain"
    model_name = "flowmo_lfq_qwen_hi_targets_sg_50xlr_bce_0.006_pretrain"
    # model_name = "flowmo_hi"

    # Model configuration
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
    model = train_utils.build_model(config)
    # Filter out keys starting with 'qwen_model'
    filtered_state_dict = {k: v for k, v in state_dict['model_ema_state_dict'].items() if not k.startswith('qwen_model')}
    # Load the filtered state dict, ignoring missing keys (like the qwen_model ones)
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    model.requires_grad_(False)

    # Process the filtered shapes3d dataset
    processed_shapes3d = tokenize_and_reconstruct(filtered_shapes3d)
    print(f"Processed dataset with FlowMo model {model_name}:", processed_shapes3d)

    # Save the processed dataset
    processed_shapes3d.save_to_disk(f"../infomec/outputs/processed_shapes3d_{model_name.replace('/', '_')}")
