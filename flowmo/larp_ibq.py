import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from flowmo import ibq

def init_qwen_weights(std, module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, Qwen3RMSNorm):
        module.weight.data.fill_(1.0)

def randomize_qwen(model):
    print('Randomizing Qwen weights...')
    for name, module in model.named_modules():
        init_qwen_weights(model.config.initializer_range, module)

def delete_embedding_layer(model):
    print('Deleting Qwen embedding layer...')
    del model.model.embed_tokens
    del model.lm_head
    model.model.embed_tokens = None
    model.lm_head = None

def freeze_embedding_layer(model):
    # Freeze embedding and lm_head layers
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False


# https://github.com/FoundationVision/LlamaGen/blob/ce98ec41803a74a90ce68c40ababa9eaeffeb4ec/autoregressive/models/gpt.py#L56
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # Ensure labels are on the same device as embedding table
        labels = labels.to(self.embedding_table.weight.device)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class LARPQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.quantizer = ibq.IndexPropagationQuantize(
                codebook_size=2**self.config.model.codebook_size_for_entropy,
                e_dim=self.config.model.codebook_size_for_entropy,
            )
        self.codebook_dim = self.quantizer.e_dim
        self.codebook_size = self.quantizer.codebook_size

        # Continuous Autoregressive Prior Model
        prior_config = config.prior

        from transformers import AutoModelForCausalLM

        model_id = f'Qwen/{prior_config.model_name}'

        self.prior_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
        )
        randomize_qwen(self.prior_model)
        # delete_embedding_layer(self.prior_model)
        freeze_embedding_layer(self.prior_model)

        self.prior_model_project_in = nn.Linear(self.config.model.codebook_size_for_entropy, self.prior_model.config.hidden_size)
        self.prior_model_project_out = nn.Linear(self.prior_model.config.hidden_size, self.config.model.codebook_size_for_entropy)

        self.prior_model_cls_embedding = LabelEmbedder(config.data.num_classes, self.prior_model.config.hidden_size, config.prior.class_dropout_prob)


    def forward(self, x, cond, prior_stop_grad=True):
        # x: [batch_size, feature_dim, sequence_length]
        batch_size, feature_dim, seq_len = x.shape
        
        quantized, (commit_loss, double_quant_loss, per_sample_entropy, codebook_entropy, entropy_aux_loss), indices = self.quantizer(x)

        # 2. De-quantize for AR Prior Input
        # self.quantizer.codebook is [codebook_size, codebook_dim]
        codebook = self.quantizer.embedding.weight.detach() if prior_stop_grad else self.quantizer.embedding.weight
        
        # 3. Autoregressive Prior Model
        # Prior input: quantized shifted by one (predict next token)
        # Prior target: actual next token indices
        prior_input = rearrange(quantized.detach() if prior_stop_grad else quantized, "b d t -> b t d")[:, :-1, :]
        prior_target_indices = (indices.detach() if prior_stop_grad else indices).reshape(batch_size, seq_len)

        # Get v_bar from prior model
        prior_input = self.prior_model_project_in(prior_input)
        cond_embeddings = self.prior_model_cls_embedding(cond, train=self.training)
        prior_input = torch.cat((cond_embeddings, prior_input), dim=1)
        prior_output = self.prior_model(inputs_embeds=prior_input, output_hidden_states=True).hidden_states[-1]
        # predicted: [B, seq_len-1, codebook_dim]
        predicted = self.prior_model_project_out(prior_output)
    
        # # 4. Calculate NLL loss (Lprior)
        logits = torch.einsum('btd,cd->btc', predicted, codebook)
        
        # Probabilities p = softmax(s)
        # NLL loss: F.cross_entropy expects logits [N, C] and targets [N]
        prior_loss = F.cross_entropy(
            logits.reshape(-1, self.codebook_size), # [B*(T-1), C]
            prior_target_indices.reshape(-1)          # [B*(T-1)]
        )

        return quantized, (prior_loss, commit_loss, double_quant_loss, per_sample_entropy, codebook_entropy, entropy_aux_loss), indices
