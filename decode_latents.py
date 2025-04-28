#!/usr/bin/env python3
"""
This script decodes latent embeddings from FlowMo models into text tokens using Qwen's embedding layer.
It displays the original images alongside the decoded tokens for different model variants.
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from flowmo import train_utils
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_model(model_name, ckpt_path, device="cuda"):
    """Load a FlowMo model from a specific checkpoint."""
    config = OmegaConf.load('flowmo/configs/base.yaml')
    config.data.batch_size = 100 # Adjusted for potential memory constraints during eval
    config.data.num_workers = 1
    
    # Define the base model configurations (without specific checkpoint)
    zoo = {
        "0": {
            "context_dim": 896, 
            "quantization_type": "noop", 
            "code_length": 128, 
            "patch_size": 8, 
            "mup_width": 4,
            "base_ckpt_dir": "results/flowmo_noop_pretrain/checkpoints/"
        },
        "0.3": {
            "context_dim": 896, 
            "quantization_type": "qwen2.5-coder-0.5b_span_0.3", 
            "code_length": 128, 
            "patch_size": 8, 
            "mup_width": 4,
            "base_ckpt_dir": "results/flowmo_qwen2.5-coder-0.5b_span_0.3_pretrain/checkpoints/"
        },
        "0.6": {
            "context_dim": 896, 
            "quantization_type": "qwen2.5-coder-0.5b_span_0.6", 
            "code_length": 128, 
            "patch_size": 8, 
            "mup_width": 4,
            "base_ckpt_dir": "results/flowmo_qwen2.5-coder-0.5b_span_0.6_pretrain/checkpoints/"
        },
        "0.9": {
            "context_dim": 896, 
            "quantization_type": "qwen2.5-coder-0.5b_span_0.9", 
            "code_length": 128, 
            "patch_size": 8, 
            "mup_width": 4,
            "base_ckpt_dir": "results/flowmo_qwen2.5-coder-0.5b_span_0.9_pretrain/checkpoints/"
        }
    }
    
    if model_name not in zoo:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    model_config = zoo[model_name]
    
    # Configure model parameters from zoo
    config.model.context_dim = model_config['context_dim']
    if 'patch_size' in model_config:
        config.model.patch_size = model_config['patch_size']
    if 'mup_width' in model_config:
        config.model.mup_width = model_config['mup_width']
    if 'code_length' in model_config:
        config.model.code_length = model_config['code_length']
    if 'quantization_type' in model_config:
        config.model.quantization_type = model_config['quantization_type']
    
    config.model.codebook_size_for_entropy = 1  # Not needed at test time
    
    # Build the model
    model = train_utils.build_model(config)
    
    # Load checkpoint if path is valid
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        
        # Filter out keys starting with 'qwen_model' if they exist in the checkpoint
        # Check if 'model_ema_state_dict' exists, otherwise try 'model_state_dict' or the root
        if 'model_ema_state_dict' in state_dict:
            source_state_dict = state_dict['model_ema_state_dict']
        elif 'model_state_dict' in state_dict:
             source_state_dict = state_dict['model_state_dict']
        else:
             source_state_dict = state_dict # Assume root level if keys are missing

        filtered_state_dict = {k: v for k, v in source_state_dict.items() 
                             if not k.startswith('qwen_model')}
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys during state dict load: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys during state dict load: {unexpected_keys}")
            
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
    model.eval()
    model.to(device) # Ensure model is on the correct device
    return model, config

def decode_embeddings(qwen_model, tokenizer, embeddings, device="cuda"):
    """Decode embeddings to text tokens using Qwen's embedding layer."""
    # Get the embedding matrix from the model
    embedding_matrix = qwen_model.get_input_embeddings().weight

    print('decode_embeddings!!!')
    print(embeddings.shape)
    print(embedding_matrix.shape)
    
    # Move embeddings to the same device as the embedding matrix
    embeddings = embeddings.to(embedding_matrix.device).to(embedding_matrix.dtype)
    
    # Calculate cosine similarity between embeddings and all token embeddings
    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
    embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=-1, keepdim=True)
    
    # Calculate similarity
    similarity = torch.matmul(embeddings_norm, embedding_matrix_norm.t())
    
    # Get the most similar token for each embedding
    token_ids = similarity.argmax(dim=-1)
    
    # Convert token IDs to text
    decoded_texts = []
    for ids in token_ids:
        text = tokenizer.decode(ids.tolist())
        decoded_texts.append(text)
    
    return decoded_texts, token_ids

def display_image_and_tokens(image, decoded_tokens, embeddings, tokenizer, title):
    """Display an image alongside its decoded tokens."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    # Display the image
    ax1.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Display the decoded tokens
    token_text = tokenizer.decode(decoded_tokens)
    # Calculate mean of embedding values
    embedding_l1 = embeddings.norm(p=1)
    embedding_mean = embeddings.mean().item()
    embedding_std = embeddings.std().item()
    ax2.text(0.1, 0.5, f"{token_text}\n\nL1={embedding_l1}\n\nMean emb={embedding_mean:.6f}\n\nStd emb={embedding_std:.6f}", wrap=True, fontsize=10)
    ax2.set_title("Decoded Tokens")
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def main():
    # Set random seed
    set_seed(42)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation dataset
    config = OmegaConf.load('flowmo/configs/base.yaml')
    config.data.batch_size = 5  # Use a small batch size for visualization
    config.data.num_workers = 4  # Adjust based on system capability

    # Load a few images for visualization
    val_dataloader = train_utils.load_dataset(config, 'val', shuffle_val=True)  # Shuffle to get random images
    
    # Get a batch of images
    batch = next(train_utils.wrap_dataloader([next(iter(val_dataloader))]))
    images = batch['image']
    print(f"Loaded {images.size(0)} images for visualization.")

    # Define models and steps to test
    models_to_test = ["0.6", "0.3", "0", "0.9"]
    step = 90000  # Use a late checkpoint for better quality
    
    # Store base checkpoint directories
    zoo_base_paths = {
        "0": "results/flowmo_noop_pretrain/checkpoints/",
        "0.3": "results/flowmo_qwen2.5-coder-0.5b_span_0.3_pretrain/checkpoints/",
        "0.6": "results/flowmo_qwen2.5-coder-0.5b_span_0.6_pretrain/checkpoints/",
        "0.9": "results/flowmo_qwen2.5-coder-0.5b_span_0.9_pretrain/checkpoints/"
    }

    # Load the Qwen model and tokenizer
    print("Loading Qwen model and tokenizer...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    qwen_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True
    )

    # Create output directory for visualizations
    os.makedirs("decoded_visualizations", exist_ok=True)

    # Process each model variant
    for model_name in models_to_test:
        print(f"\nProcessing {model_name}...")
        ckpt_file = f"{step:08d}.pth"
        ckpt_path = os.path.join(zoo_base_paths[model_name], ckpt_file)
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping model {model_name}.")
            continue

        try:
            # Load FlowMo model
            model, _ = load_model(model_name, ckpt_path, device)
            
            # Process each image in the batch
            for i, image in enumerate(images):
                # Add batch dimension for model input
                image_batch = image.unsqueeze(0).to(device)
                
                # Get latent embeddings from the model
                with torch.no_grad():
                    latent_embeddings, _ = model.encode(image_batch)
                
                # Decode embeddings to tokens
                decoded_texts, token_ids = decode_embeddings(qwen_model, tokenizer, latent_embeddings[0], device)
                
                # Display and save the visualization
                fig = display_image_and_tokens(
                    image, 
                    token_ids,
                    latent_embeddings[0],
                    tokenizer, 
                    f"Model {model_name} - Image {i+1}"
                )

                # Save the latent embeddings to a text file with equal padding
                embedding_file_path = f"decoded_visualizations/model_{model_name}_image_{i+1}_embedding.txt"
                with open(embedding_file_path, 'w') as f:
                    # Convert tensor to numpy for easier handling
                    embedding_np = latent_embeddings[0].cpu().numpy()
                    
                    # Find the maximum width needed for padding
                    max_width = max(len(f"{value:.3f}") for value in embedding_np.flatten())
                    
                    # Write each embedding vector as a line with equal padding
                    for j in range(embedding_np.shape[0]):
                        line = " ".join([f"{value:{max_width}.3f}" for value in embedding_np[j]])
                        f.write(line + "\n")
                
                print(f"Saved embedding to {embedding_file_path}")
                
                # Save the figure
                fig.savefig(f"decoded_visualizations/model_{model_name}_image_{i+1}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Processed image {i+1} with model {model_name}")
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
    
    print("\nAll visualizations saved to 'decoded_visualizations' directory")

if __name__ == "__main__":
    main()
