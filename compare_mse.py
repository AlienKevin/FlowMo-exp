#!/usr/bin/env python3
"""
This script compares the Mean Squared Error (MSE) when predicting image tokens
with different FlowMo models and checkpoints on 100 random images from ImageNet.
It plots the MSE trend over training steps.
"""

import torch
import numpy as np
from omegaconf import OmegaConf
from flowmo import train_utils
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
import os
from tqdm import tqdm

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

def calculate_mse(model, qwen_model, images, device="cuda"):
    """Calculate MSE between output embeddings of the qwen model (shifted) and the input embeddings."""
    mse_values = []
    
    # Move images to device
    images = images.to(device)
    
    with torch.no_grad():
        # Get the input embeddings for all images
        input_embeddings, _ = model.encode(images)
        input_embeddings = input_embeddings.to(torch.bfloat16) # Match Qwen dtype
        
        # Forward pass through the qwen model
        # Ensure qwen_model is also on the correct device and dtype if not using device_map='auto' fully
        # qwen_model = qwen_model.to(device, dtype=torch.bfloat16) # Redundant if device_map='auto' works
        
        output_embeddings = qwen_model(inputs_embeds=input_embeddings, output_hidden_states=True, return_dict=True).hidden_states[-1]
        
        # Shift for causal prediction: Output[t] predicts Input[t+1]
        shifted_input_embeddings = input_embeddings[:, 1:, :]
        predicted_embeddings = output_embeddings[:, :-1, :] # Predictions up to the second-to-last token
        
        # Calculate MSE per image in the batch
        # Ensure dimensions match before calculating MSE
        if shifted_input_embeddings.shape == predicted_embeddings.shape:
            mse_per_token = (shifted_input_embeddings - predicted_embeddings) ** 2
            # Average MSE over tokens and sequence length for each image
            mse_per_image = torch.mean(mse_per_token, dim=[1, 2]) 
            mse_values.extend(mse_per_image.cpu().tolist()) # Collect MSE for each image
        else:
            print(f"Warning: Shape mismatch! Input: {shifted_input_embeddings.shape}, Output: {predicted_embeddings.shape}")
            # Fallback or error handling needed here
            # For now, calculate MSE on the overlapping part if possible, or skip
            min_len = min(shifted_input_embeddings.shape[1], predicted_embeddings.shape[1])
            mse_per_token = (shifted_input_embeddings[:, :min_len, :] - predicted_embeddings[:, :min_len, :]) ** 2
            mse_per_image = torch.mean(mse_per_token, dim=[1, 2])
            mse_values.extend(mse_per_image.cpu().tolist())


    return mse_values

def main():
    # Set random seed
    set_seed(42)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation dataset
    config = OmegaConf.load('flowmo/configs/base.yaml')
    config.data.batch_size = 10 # Use a reasonable batch size for evaluation
    config.data.num_workers = 4 # Adjust based on system capability

    # Load a fixed batch of 100 images for consistent comparison
    val_dataloader = train_utils.load_dataset(config, 'val', shuffle_val=False) # No shuffle for consistency
    
    # Accumulate 100 images (or fewer if dataset is smaller)
    images_list = []
    target_images = 100
    images_collected = 0
    for batch in val_dataloader:
        batch_images = next(train_utils.wrap_dataloader([batch]))['image']
        num_to_take = min(target_images - images_collected, batch_images.size(0))
        images_list.append(batch_images[:num_to_take])
        images_collected += num_to_take
        if images_collected >= target_images:
            break
    
    if not images_list:
        print("Error: Could not load any images from the validation set.")
        return
        
    images = torch.cat(images_list, dim=0)
    print(f"Loaded {images.size(0)} images for evaluation.")

    # Define models and steps to test
    models_to_test = ["0.6", "0.3", "0", "0.9"]
    steps = list(range(5000, 95001, 10000))
    
    # Store base checkpoint directories
    zoo_base_paths = {
        "0": "results/flowmo_noop_pretrain/checkpoints/",
        "0.3": "results/flowmo_qwen2.5-coder-0.5b_span_0.3_pretrain/checkpoints/",
        "0.6": "results/flowmo_qwen2.5-coder-0.5b_span_0.6_pretrain/checkpoints/",
        "0.9": "results/flowmo_qwen2.5-coder-0.5b_span_0.9_pretrain/checkpoints/"
    }

    # Initialize results structure
    results = {model_name: {'steps': [], 'mean_mse': [], 'std_mse': []} for model_name in models_to_test}

    # Load the Qwen model once
    print("Loading Qwen model...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        # "Qwen/Qwen2.5-Coder-0.5B",
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically distribute across available GPUs/CPU
        trust_remote_code=True,
    )
    qwen_model.eval() # Set Qwen model to evaluation mode

    # Evaluation loop
    for model_name in tqdm(models_to_test):
        print(f"\nEvaluating {model_name}...")
        base_path = zoo_base_paths[model_name]
        
        valid_steps = []
        mean_mses = []
        std_mses = []

        for step in tqdm(steps, desc=f"Steps for {model_name}"):
            ckpt_file = f"{step:08d}.pth"
            ckpt_path = os.path.join(base_path, ckpt_file)
            
            if not os.path.exists(ckpt_path):
                print(f"  Checkpoint not found: {ckpt_path}, skipping step {step}.")
                continue

            try:
                # Load FlowMo model for the current step
                model, _ = load_model(model_name, ckpt_path, device)
                
                # Calculate MSE for the batch of images
                mse_values = calculate_mse(model, qwen_model, images, device)
                
                if mse_values: # Ensure we got some results
                    mean_mse = np.mean(mse_values)
                    std_mse = np.std(mse_values)
                    
                    valid_steps.append(step)
                    mean_mses.append(mean_mse)
                    std_mses.append(std_mse)
                    
                    print(f"  Step {step:08d}: Mean MSE = {mean_mse:.6f} ± {std_mse:.6f}")
                else:
                     print(f"  Step {step:08d}: No MSE values calculated.")

                # Clean up GPU memory
                del model
                torch.cuda.empty_cache() 

            except FileNotFoundError:
                 # Already handled by the check above, but keep for safety
                 print(f"  Checkpoint file not found during load attempt: {ckpt_path}")
            except Exception as e:
                print(f"  Error processing step {step} for model {model_name}: {e}")
                # Optionally clean up memory here too if an error occurred mid-process
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
        
        # Store results for this model variant
        results[model_name]['steps'] = valid_steps
        results[model_name]['mean_mse'] = mean_mses
        results[model_name]['std_mse'] = std_mses

    # Plot results
    plt.figure(figsize=(12, 7))
    
    for model_name in models_to_test:
        if results[model_name]['steps']: # Check if there are results to plot
            steps_array = np.array(results[model_name]['steps'])
            mean_mse_array = np.array(results[model_name]['mean_mse'])
            std_mse_array = np.array(results[model_name]['std_mse'])
            
            # Plot mean MSE line
            plt.plot(steps_array, mean_mse_array, marker='o', linestyle='-', label=f"Model {model_name}")
            
            # Add shaded region for standard deviation
            plt.fill_between(steps_array, 
                             mean_mse_array - std_mse_array, 
                             mean_mse_array + std_mse_array, 
                             alpha=0.2, label=f"_±1 std dev {model_name}") # Use _ to hide from main legend if desired

    plt.title("MSE vs. Training Steps for FlowMo Models")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # Ensure x-axis ticks cover the intended range, even if some checkpoints were missing
    all_steps = sorted(list(set(step for model_data in results.values() for step in model_data['steps'])))
    if all_steps:
        plt.xticks(all_steps, rotation=45)
    else:
         plt.xticks(steps, rotation=45) # Fallback to original steps if no results
    plt.yscale('log') # Use log scale if MSE values vary widely
    plt.tight_layout() # Adjust layout
    
    plt.savefig('flowmo_mse_vs_steps.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to flowmo_mse_vs_steps.png")

if __name__ == "__main__":
    main()
