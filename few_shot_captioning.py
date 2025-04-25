import torch
import torch.nn as nn
from omegaconf import OmegaConf
from flowmo import train_utils
from transformers import AutoModelForCausalLM, AutoTokenizer
import einops

class FewShotImageCaptioner:
    """
    A class for performing k-shot image captioning using FlowMo model.
    It supports zero-shot or k-shot image captioning with in-context learning.
    """
    
    def __init__(self, model_name="flowmo_hi_qwen2.5-coder-0.5b_span_0.3", device="cuda"):
        """
        Initialize the image captioner with specified model.
        
        Args:
            model_name (str): Name of the FlowMo model to use
            device (str): Device to run the model on (cuda or cpu)
        """
        self.device = device
        self.model_name = model_name
        
        # Set up config
        self.config = OmegaConf.load('flowmo/configs/base.yaml')
        self.config.data.batch_size = 1
        self.config.data.num_workers = 0
        
        # Initialize model
        self._load_flowmo_model()
        self._load_qwen_model()
        
    def _load_flowmo_model(self):
        """Load the FlowMo model with appropriate configuration."""
        # The low BPP model has 18 bits per token, the high bitrate model has 56 bits per token.
        zoo = {
            "flowmo_lo": {"context_dim": 18, "ckpt_path": "flowmo_lo.pth"},
            "flowmo_hi": {"context_dim": 56, "ckpt_path": "flowmo_hi.pth"},
            "flowmo_hi_45000": {"context_dim": 56, "patch_size": 8, "mup_width": 4, "ckpt_path": "results/flowmo_hi_pretrain/checkpoints/00045000.pth"},
            "flowmo_hi_50000": {"context_dim": 56, "patch_size": 8, "mup_width": 4, "ckpt_path": "results/flowmo_hi_pretrain/checkpoints/00050000.pth"},
            "flowmo_hi_kl": {"context_dim": 768, "quantization_type": "kl", "code_length": 128, "patch_size": 8, "mup_width": 4, "ckpt_path": "results/flowmo_hi_kl_pretrain/checkpoints/00015000.pth"},
            "flowmo_hi_noop": {"context_dim": 768, "quantization_type": "noop", "code_length": 128, "patch_size": 8, "mup_width": 4, "ckpt_path": "results/flowmo_hi_noop_pretrain/checkpoints/00060000.pth"},
            "flowmo_hi_qwen2.5-coder-0.5b_span_0.3": {"context_dim": 896, "quantization_type": "qwen2.5-coder-0.5b_span_0.3", "code_length": 128, "patch_size": 8, "mup_width": 4, "ckpt_path": "results/flowmo_qwen2.5-coder-0.5b_span_0.3_pretrain/checkpoints/00095000.pth"},
            "flowmo_hi_qwen2.5-coder-0.5b_span_0.6": {"context_dim": 896, "quantization_type": "qwen2.5-coder-0.5b_span_0.6", "code_length": 128, "patch_size": 8, "mup_width": 4, "ckpt_path": "results/flowmo_qwen2.5-coder-0.5b_span_0.6_pretrain/checkpoints/00095000.pth"},
            "flowmo_hi_qwen2.5-coder-0.5b_span_0.9": {"context_dim": 896, "quantization_type": "qwen2.5-coder-0.5b_span_0.9", "code_length": 128, "patch_size": 8, "mup_width": 4, "ckpt_path": "results/flowmo_qwen2.5-coder-0.5b_span_0.9_pretrain/checkpoints/00095000.pth"},
        }
        
        model_config = zoo[self.model_name]
        
        if 'ckpt_path' in model_config:
            state_dict = torch.load(model_config['ckpt_path'], map_location=self.device)
            
        self.config.model.context_dim = model_config['context_dim']
        if 'patch_size' in model_config:
            self.config.model.patch_size = model_config['patch_size']
        if 'mup_width' in model_config:
            self.config.model.mup_width = model_config['mup_width']
        if 'code_length' in model_config:
            self.config.model.code_length = model_config['code_length']
        if 'quantization_type' in model_config:
            self.config.model.quantization_type = model_config['quantization_type']
            
        self.config.model.codebook_size_for_entropy = 1  # don't need this at test time.
        
        self.flowmo_model = train_utils.build_model(self.config)
        
        if 'ckpt_path' in model_config:
            # Filter out keys starting with 'qwen_model'
            filtered_state_dict = {k: v for k, v in state_dict['model_ema_state_dict'].items() if not k.startswith('qwen_model')}
            # Load the filtered state dict, ignoring missing keys (like the qwen_model ones)
            self.flowmo_model.load_state_dict(filtered_state_dict, strict=False)
            
        self.flowmo_model.eval()
        
    def _load_qwen_model(self):
        """Load the Qwen language model and tokenizer."""
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-coder-0.5B",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-coder-0.5B",
            trust_remote_code=True
        )
        
    def encode_image(self, image):
        """
        Encode an image using the FlowMo model.
        
        Args:
            image (torch.Tensor): Image tensor with shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Image token embeddings
        """
        with torch.no_grad():
            image_tokens, _ = self.flowmo_model.encode(image)
        return image_tokens
    
    def encode_text(self, text):
        """
        Encode text using the Qwen tokenizer and model.
        
        Args:
            text (str): Text to encode
            
        Returns:
            torch.Tensor: Text embeddings
        """
        text_tokens = self.qwen_tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        # Remove potential EOS token from text if tokenizer adds one automatically
        if self.qwen_tokenizer.eos_token_id is not None and text_tokens[0, -1] == self.qwen_tokenizer.eos_token_id:
            text_tokens = text_tokens[:, :-1]
            
        # Get text embeddings from the Qwen model's embedding layer
        text_embeds = self.qwen_model.get_input_embeddings()(text_tokens)
        
        return text_embeds
    
    def generate_caption(self, 
                         input_image, 
                         example_images=None, 
                         example_captions=None, 
                         prompt="is an image of",
                         max_new_tokens=None,
                         **generation_kwargs):
        """
        Generate a caption for an input image using k-shot learning if example images/captions are provided.
        
        Args:
            input_image (torch.Tensor): The image to caption, shape [1, C, H, W]
            example_images (list of torch.Tensor, optional): List of example images for in-context learning
            example_captions (list of str, optional): List of captions corresponding to example images
            prompt (str): Prompt to use for captioning
            max_new_tokens (int): Maximum number of tokens to generate
            **generation_kwargs: Additional keyword arguments for text generation
            
        Returns:
            str: Generated caption
        """
        # Validate inputs
        if example_images is not None and example_captions is not None:
            assert len(example_images) == len(example_captions), "Number of example images and captions must match"
        elif example_images is not None or example_captions is not None:
            raise ValueError("Both example_images and example_captions must be provided together for k-shot learning")
        
        # Encode the input image
        input_image_tokens = self.encode_image(input_image.unsqueeze(0))
        
        # Build the input embeddings sequence
        all_embeddings = []
        
        # Add in-context examples if provided (k-shot)
        if example_images is not None and example_captions is not None:
            for img, caption in zip(example_images, example_captions):
                # Encode example image
                example_img_tokens = self.encode_image(img.unsqueeze(0))
                
                # Encode caption
                example_caption_embeds = self.encode_text(f'{prompt} {caption}.')
                
                # Combine for this example
                all_embeddings.append(torch.cat([
                    example_img_tokens,
                    example_caption_embeds,
                ], dim=1))
        
        # Add the input image with prompt
        prompt_embeds = self.encode_text(prompt)
        all_embeddings.append(torch.cat([input_image_tokens, prompt_embeds], dim=1))
        
        # Combine all embeddings
        if len(all_embeddings) > 1:
            # Concatenate all examples and the input
            input_embeds = torch.cat(all_embeddings, dim=1).to(torch.bfloat16)
        else:
            # Just the input image + prompt
            input_embeds = all_embeddings[0].to(torch.bfloat16)
        
        # Create attention mask (1s for all positions)
        attention_mask = torch.ones(
            (input_embeds.shape[0], input_embeds.shape[1]),
            dtype=torch.long,
            device=self.device
        )
        
        # Generate caption using the Qwen model
        with torch.no_grad():
            if max_new_tokens is not None:
                generation_kwargs.setdefault('max_new_tokens', max_new_tokens)
            generated_ids = self.qwen_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.qwen_tokenizer.pad_token_id,
                **generation_kwargs,
            )
        
        # Decode the generated text using the Qwen tokenizer
        response = self.qwen_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return response.strip()
