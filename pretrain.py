from torch.optim import AdamW
import torch
from tqdm import tqdm
import wandb
import os
from torch.utils.data import DataLoader
from datasets import Dataset
import json
import random


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

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True)
        print(self.train_dataset)
        print(self.val_dataset)

        # Get optimizer
        lr = 1e-4
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def eval(self):
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
        return {"val/loss": avg_loss}

    def train(self):
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

            if ((epoch + 1) % 10 == 0) or (epoch == self.epochs - 1):
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
    sft_trainer = SFTTrainer(entity='kevinxli', project_name='dog_imagenet_gpt2', epochs=100, batch_size=256)
    sft_trainer.train()
