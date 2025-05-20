# -*- coding: utf-8 -*-
"""
Training script for Qwen on visual tokens from FlowMo encoder.
"""
from unsloth import FastLanguageModel, add_new_tokens
from flowmo import models, train_utils
import os
import torch
import torch.optim as optim
from omegaconf import OmegaConf, DictConfig
import argparse
from torch.utils.tensorboard import SummaryWriter

def get_visual_tokens(flowmo_model: models.FlowMo, images: torch.Tensor, device: torch.device):
    """
    Encodes images using FlowMo and returns quantized visual token indices.
    Images should be preprocessed (e.g., normalized to [-1, 1]).
    """
    with torch.no_grad():
        images = images.to(device)
        # Ensure FlowMo's internal config for quantization is correctly set via its constructor.
        # .encode calls ._quantize internally if configured for some models, but here we call explicitly after .encode
        codes, _encode_aux = flowmo_model.encode(images) 
        # codes: (batch, code_length, encoder_feature_dim)
        
        # _quantize uses flowmo_model.config.model.* settings
        _quantized_codes, indices, quant_aux = flowmo_model._quantize(codes)
        # Unflatten indices
        indices = indices.view(images.size(0), -1)

    return indices


def load_qwen_model_and_tokenizer(model_name: str, num_new_visual_tokens: int, device: torch.device, config: DictConfig):
    """Loads Qwen model and tokenizer, extends vocab for visual tokens."""
    print(f"Loading Qwen model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.qwen_train.qwen_max_seq_length, # Max sequence length for Qwen
        dtype=torch.bfloat16 if config.qwen_train.use_bfloat16 else torch.float32,
        load_in_4bit=False, # Full model for training
        load_in_8bit=False,
        full_finetuning=True,
        # token = "YOUR_HF_TOKEN" # if private model
    )
    
    print(f"Original Qwen vocab size: {len(tokenizer)}")
    new_tokens = [f"<VISUAL_{i}>" for i in range(num_new_visual_tokens)]
    add_new_tokens(model, tokenizer, new_tokens=new_tokens)
    print(f"Extended Qwen vocab size: {len(tokenizer)}")
    
    model.to(device)
    return model, tokenizer


def load_flowmo_from_config(config: DictConfig, device: torch.device):
    """Loads and returns a FlowMo model from the given configuration."""
    print(f"Loading FlowMo model from checkpoint: {config.qwen_train.flowmo_ckpt_path}")
    
    flowmo_model = train_utils.build_model(config)
    
    train_utils.restore_from_ckpt(flowmo_model, None, config.qwen_train.flowmo_ckpt_path)
    
    flowmo_model.to(device)
    
    return flowmo_model

def save_qwen_model_and_tokenizer(model, tokenizer, save_dir):
    """Save Qwen model and tokenizer to the specified directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Saving Qwen model and tokenizer to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved successfully to {save_dir}")

def train_qwen_on_visual_tokens(config: DictConfig):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(config.qwen_train.checkpoint_dir)

    if config.qwen_train.use_bfloat16 and not torch.cuda.is_bf16_supported():
        print("Warning: BFloat16 is requested but not supported on this device. Using Float32.")
        config.qwen_train.use_bfloat16 = False
    
    # 1. Load FlowMo encoder using the existing train_utils function
    flowmo_encoder = load_flowmo_from_config(config, device)
    flowmo_encoder.eval()
    for param in flowmo_encoder.parameters():
        param.requires_grad = False
    print("FlowMo model is frozen.")

    # 2. Load Qwen model and tokenizer
    num_visual_tokens = 2**config.model.codebook_size_for_entropy
    qwen_model, qwen_tokenizer = load_qwen_model_and_tokenizer(
        config.qwen_train.qwen_model_name,
        num_visual_tokens,
        device,
        config
    )
    # The ID of the first added visual token
    visual_token_start_id = len(qwen_tokenizer) - num_visual_tokens
    print(f"Visual tokens will be mapped to Qwen IDs starting from {visual_token_start_id}")

    # 3. Prepare ImageNet dataset and DataLoader using train_utils
    print("Loading dataset...")
    train_dataloader_from_utils = train_utils.load_dataset(config=config, split='train')

    dl_iter = iter(train_utils.wrap_dataloader(train_dataloader_from_utils)) # Wrap for permuting and moving to device

    # 4. Optimizer and Loss Criterion for Qwen
    optimizer = optim.AdamW(qwen_model.parameters(), lr=config.qwen_train.learning_rate)

    # 5. Training Loop
    print("Starting training...")
    total_steps = 0
    for epoch in range(config.qwen_train.epochs):
        qwen_model.train()
        total_lm_loss_epoch = 0.0
        processed_batches = 0

        # Calculate number of batches for epoch based on original dataset size if possible, or steps_per_epoch
        # This is an estimation if using an IterableDataset without a clear length for epochs.
        # If train_dataloader_from_utils.dataset has a len, use it.
        num_images_in_dataset = len(train_dataloader_from_utils.dataset)
        batches_per_epoch = (num_images_in_dataset + config.data.batch_size - 1) // config.data.batch_size
        print(f"Epoch {epoch+1}: Estimated batches per epoch: {batches_per_epoch}")

        for batch_idx in range(batches_per_epoch):
            batch = next(dl_iter)
            images = batch["image"]
            
            # A. Get visual tokens from FlowMo
            # visual_indices: (batch_size, num_tokens_per_image), values from 0 to (2^S - 1)
            visual_indices = get_visual_tokens(flowmo_encoder, images, device)
            
            # Map LFQ indices to Qwen's new visual token IDs
            # visual_token_ids: (batch_size, num_tokens_per_image)
            visual_token_ids = visual_indices.to(device) + visual_token_start_id

            # B. Prepare sequences for Qwen (LM input and target)
            # Since we assume each image fits within qwen_seq_len, we can simplify
            # Input: all tokens except the last one
            # Target: all tokens except the first one
            input_ids = visual_token_ids[:, :-1]  # (batch_size, num_tokens_per_image - 1)
            lm_labels = visual_token_ids[:, 1:]   # (batch_size, num_tokens_per_image - 1)

            # C. Qwen forward pass, loss calculation, and optimization
            optimizer.zero_grad()
            
            # Autocast for mixed precision if enabled
            autocast_dtype = torch.bfloat16 if config.qwen_train.use_bfloat16 else torch.float32
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=config.qwen_train.use_bfloat16):
                # Standard Hugging Face model call with labels for LM loss
                outputs = qwen_model(input_ids=input_ids, labels=lm_labels)
                lm_loss = outputs.loss

            # D. Backward pass and optimization
            lm_loss.backward()
            optimizer.step()
            
            total_lm_loss_epoch += lm_loss.item() * input_ids.shape[0] # Weighted by number of sequences
            processed_batches +=1
            total_steps += 1

            # Log metrics to TensorBoard
            writer.add_scalar('train/qwen_ce_loss', lm_loss.item(), total_steps)

        avg_epoch_lm_loss = total_lm_loss_epoch / (num_images_in_dataset * (visual_token_ids.shape[1] // config.qwen_train.qwen_seq_len if visual_token_ids.shape[1]>0 else 1) if num_images_in_dataset >0 and visual_token_ids.shape[1]>0 else 1) # also rough
        print(f"Epoch {epoch+1} finished. Average Qwen LM Loss for epoch: {avg_epoch_lm_loss:.4f}")
        writer.add_scalar('epoch/qwen_ce_loss', avg_epoch_lm_loss, epoch+1)

        # E. Save checkpoint
        if (epoch + 1) % config.qwen_train.save_interval == 0:
            hf_save_dir = os.path.join(config.qwen_train.checkpoint_dir, f"qwen_visual_hf_epoch_{epoch+1}")
            save_qwen_model_and_tokenizer(qwen_model, qwen_tokenizer, hf_save_dir)
    
    # Save final model and tokenizer
    final_hf_save_dir = os.path.join(config.qwen_train.checkpoint_dir, "qwen_visual_final")
    save_qwen_model_and_tokenizer(qwen_model, qwen_tokenizer, final_hf_save_dir)
    
    writer.close()


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train Qwen on FlowMo visual tokens.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a YAML configuration file compatible with FlowMo.")
    
    args = parser.parse_args()

    # --- Load Configuration ---
    print(f"Loading configuration from file: {args.config}")
    cfg = OmegaConf.load(args.config)
    print(cfg)

    train_qwen_on_visual_tokens(cfg)
