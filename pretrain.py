from torch.optim import AdamW
import torch
from tqdm import tqdm
import wandb
import os
from torch.utils.data import DataLoader
from datasets import Dataset
import json
import random
from omegaconf import OmegaConf
from flowmo import train_utils
import torchvision
from einops import rearrange
import numpy as np
from transformers import LogitsProcessor


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FilterLogitsProcessor(LogitsProcessor):
    def __init__(self, filter_vocab_size: int):
        self.filter_vocab_size = filter_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.filter_vocab_size:] = -float("Inf")
        return scores


class SFTTrainer:
    def __init__(self, entity='kevinxli', project_name='sft', batch_size=256, epochs=100, max_grad_norm=1.0):
        self.ckpt_dir = f"./ckpts/{project_name}"
        self.logger = wandb.init(entity=entity, project=project_name, name=f'batch_size_{batch_size}_epochs_{epochs}')
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        with open('encoded_tokens_dogs_flowmo_lo_c2i_larp_ibq_rand_sg_128x128_pretrain.json', 'r') as f:
            data = json.load(f)

        items = [{'image_name': k, 'tokens': v} for k, v in data.items()]
        
        class_data = {}
        for item in items:
            class_id = item['image_name'].split('_')[0]
            if class_id not in class_data:
                class_data[class_id] = []
            class_data[class_id].append(item)

        unique_classes = sorted(class_data.keys())
        self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        
        with open('imagenet_class_index.json', 'r') as f:
            imagenet_class_index = json.load(f)
        
        self.class_id_to_name = {}
        for _, info in imagenet_class_index.items():
            self.class_id_to_name[info[0]] = info[1]
        
        train_items = []
        val_items = []

        for class_id, class_items in class_data.items():
            random.shuffle(class_items)
            val_split_idx = int(round(len(class_items) * 0.01))
            
            val_items.extend(class_items[:val_split_idx])
            train_items.extend(class_items[val_split_idx:])

        print(f'Train items: {len(train_items)}')
        print(f'Validation items: {len(val_items)}')
        
        train_dataset = Dataset.from_list(train_items)
        val_dataset = Dataset.from_list(val_items)
        
        # Calculate number of unique visual tokens from both datasets
        all_tokens = set()
        for item in train_items + val_items:
            all_tokens.update(item['tokens'])
        
        all_tokens = sorted(list(all_tokens))
        num_visual_tokens = len(all_tokens)
        num_class_tokens = len(self.class_to_idx)
        
        print(f'Number of unique visual tokens: {num_visual_tokens}')
        print(f'Number of class tokens: {num_class_tokens}')
        num_visual_tokens = 2 ** 9
        print(f'Number of visual tokens set to: {num_visual_tokens}')

        n_ctx = 257

        print(f"Setting context length to {n_ctx}")

        self.model = init_gpt(vocab_size=num_visual_tokens+num_class_tokens, n_ctx=n_ctx)
        self.device = self.model.device

        def preprocess(batch):
            input_ids_list = []

            for i in range(len(batch['tokens'])):
                class_id = batch['image_name'][i].split('_')[0]
                class_token = num_visual_tokens + self.class_to_idx[class_id]
                token_ids = [class_token] + batch['tokens'][i]
                input_ids_list.append(token_ids)

            return {
                "input_ids": input_ids_list,
            }

        self.train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
        self.val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)
        self.train_dataset.set_format(type='torch', columns=['input_ids'])
        self.val_dataset.set_format(type='torch', columns=['input_ids'])

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(42)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        print(self.train_dataset)
        print(self.val_dataset)

        # Get optimizer
        lr = 1e-4
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # Load decoder model for generation
        decoder_model_name = "dogs_flowmo_lo_c2i_larp_ibq_rand_sg_128x128_pretrain"
        decoder_ckpth_iteration = 150000
        config_path = f'results/{decoder_model_name}/config.yaml'
        self.decoder_config = OmegaConf.load(config_path)
        checkpoint_path = f"results/{decoder_model_name}/checkpoints/{decoder_ckpth_iteration:08d}.pth"
        
        self.decoder_model = train_utils.build_model(self.decoder_config)
        state_dict = torch.load(checkpoint_path, map_location='cuda')
        self.decoder_model.load_state_dict(state_dict['model_ema_state_dict'], strict=False)
        self.decoder_model.eval()
        self.decoder_model.cuda()
        self.num_visual_tokens = num_visual_tokens

    def eval(self):
        seed_everything(42)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                labels = input_ids.clone()

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        metrics = {"val/loss": avg_loss}

        # Generation
        self.decoder_model.eval()

        num_samples_per_class = 4
        columns = ["class_id", "class_name"] + [f"sample_{i}" for i in range(num_samples_per_class)]
        table = wandb.Table(columns=columns)
        
        val_generation_class_ids = sorted(list(self.class_to_idx.keys()))

        logits_processor = [FilterLogitsProcessor(self.num_visual_tokens)]

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            class_tokens = [self.num_visual_tokens + self.class_to_idx[class_id] for class_id in val_generation_class_ids]

            for i in tqdm(range(0, len(class_tokens), self.batch_size), desc="Generating images"):
                batch_class_tokens = class_tokens[i:i+self.batch_size]
                batch_class_ids = val_generation_class_ids[i:i+self.batch_size]
                
                prompts = torch.tensor(batch_class_tokens, device=self.device).unsqueeze(1)
                attention_mask = torch.ones_like(prompts)
                
                generated_sequences = self.model.generate(
                    prompts,
                    attention_mask=attention_mask,
                    max_length=self.model.config.n_ctx,
                    num_return_sequences=num_samples_per_class,
                    do_sample=True,
                    top_k=0,
                    top_p=1.0,
                    temperature=1.0,
                    logits_processor=logits_processor,
                )
                
                visual_tokens = generated_sequences[:, 1:]

                code_length = self.decoder_config.model.code_length
                context_dim = self.decoder_config.model.context_dim
                codebook_size_for_entropy = self.decoder_config.model.codebook_size_for_entropy
                fh = context_dim // codebook_size_for_entropy
                seq_len = code_length * fh

                indices = visual_tokens
                total_images_in_batch = indices.shape[0]
                shape = (total_images_in_batch, seq_len, codebook_size_for_entropy)
                
                quantized = self.decoder_model.quantizer.quantizer.get_codebook_entry(indices, shape)
                code = rearrange(quantized, "b fg (t fh) -> b t (fg fh)", t=code_length, fh=fh)

                reconstructed_images = self.decoder_model.reconstruct(images=torch.zeros(total_images_in_batch, 3, self.decoder_config.data.image_size, self.decoder_config.data.image_size).cuda(), code=code)
                reconstructed_images = (reconstructed_images + 1.0) / 2.0
                reconstructed_images.clamp_(0, 1)

                img_idx = 0
                for class_id in batch_class_ids:
                    class_name = self.class_id_to_name[class_id]
                    row = [class_id, class_name]
                    for _ in range(num_samples_per_class):
                        image = torchvision.transforms.ToPILImage()(reconstructed_images[img_idx])
                        row.append(wandb.Image(image))
                        img_idx += 1
                    table.add_data(*row)
        
        metrics["val/generations"] = table
        return metrics

    def train(self):
        eval_metrics = self.eval()
        self.logger.log(eval_metrics)
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            self.optimizer.zero_grad()
            for idx, batch in enumerate(tqdm(self.train_dataloader)):
                input_ids = batch["input_ids"].to(self.device)
                labels = input_ids.clone()

                # print(f"input_ids max: {input_ids.max().item()}, min: {input_ids.min().item()}")

                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    return_dict=True,
                )

                # Compute loss
                loss = outputs.loss
                micro_batch_loss = loss.item()
                epoch_loss += micro_batch_loss

                # Backward step
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.log({"epoch": epoch, "train/loss": micro_batch_loss, "train/grad_norm": grad_norm.item()})

            if (epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch == self.epochs - 1):
                model_save_path = os.path.join(self.ckpt_dir, f'sft_epoch_{epoch}')
                save_model(self.model, model_save_path)

            eval_metrics = self.eval()
            eval_metrics.update({"epoch": epoch, "train/epoch_loss": epoch_loss})
            self.logger.log(eval_metrics)

        self.logger.finish()


def init_gpt(vocab_size, n_ctx=257):
    from transformers import GPT2LMHeadModel, AutoConfig
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=vocab_size,
        n_ctx=n_ctx,
    )
    model = GPT2LMHeadModel(config).cuda()
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    return model


def save_model(model, model_save_path) -> None:
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)


if __name__ == "__main__":
    seed_everything(42)
    sft_trainer = SFTTrainer(entity='kevinxli', project_name='dog_imagenet_gpt2', epochs=200, batch_size=256)
    sft_trainer.train()
