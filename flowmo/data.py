import io
import json

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import webdataset as wds
from huggingface_hub import get_token
from torch.utils.data import IterableDataset


class IndexedTarDataset(Dataset):
    def __init__(
        self,
        imagenet_tar,
        imagenet_index,
        size=None,
        random_crop=False,
        aug_mode="default",
    ):
        self.size = size
        self.random_crop = random_crop

        self.aug_mode = aug_mode

        if aug_mode == "default":
            assert self.size is not None and self.size > 0
            self.rescaler = T.Resize(self.size)
            if not self.random_crop:
                self.cropper = T.CenterCrop((self.size, self.size))
            else:
                self.cropper = T.RandomCrop((self.size, self.size))
            self.preprocessor = T.Compose([self.rescaler, self.cropper])
        else:
            raise NotImplementedError

        # Tar setup
        self.imagenet_tar = imagenet_tar
        self.imagenet_index = imagenet_index
        with open(self.imagenet_index, "r") as fp:
            self.index = json.load(fp)
        self.index = sorted(self.index, key=lambda d: d["name"].split("/")[-1])
        self.id_to_handle = {}

        with open('imagenet_class_index.json', 'r') as f:
            imagenet_class_index = json.load(f)
        with open('imagenet-simple-labels.json', 'r') as f:
            imagenet_labels = json.load(f)
        self.labels = {}

        if self.index[0]['name'].startswith('ILSVRC2012_val_'):
            with open('imagenet_validation_ground_truth.txt', 'r') as f:
                for index, line in zip(self.index, f.readlines()):
                    self.labels[index['name'].split('.')[0]] = int(line) - 1
        else:
            # for values, label in zip(imagenet_class_index.values(), imagenet_labels):
            #     self.labels[values[0]] = label
            for class_idx, values in imagenet_class_index.items():
                self.labels[values[0]] = int(class_idx)
        
        self.captions = {}
        captions_df = pd.read_csv('imagenet_gemma3_12b.tsv', sep='\t', header=None, names=['file_name', 'caption'])
        for _, row in captions_df.iterrows():
            file_name, caption = row['file_name'], row['caption']
            self.captions[file_name] = caption

    def __len__(self):
        return len(self.index)

    def get_image(self, image_info):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if worker_id not in self.id_to_handle:
            self.id_to_handle[worker_id] = open(self.imagenet_tar, "rb")
        handle = self.id_to_handle[worker_id]

        handle.seek(image_info["offset"])
        img_bytes = handle.read(image_info["size"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image.load()
        label = self.labels[image_info["name"].split('.')[0]]
        if '.tar/' in image_info["name"]:
            caption = self.captions[image_info["name"].split('/')[-1]]
        else:
            caption = self.captions[image_info["name"]]
        return image, label, caption

    def preprocess_image(self, image_info):
        image, label, caption = self.get_image(image_info)
        image = self.preprocessor(image)
        image = np.array(image)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image, label, caption

    def __getitem__(self, i):
        example = dict()
        example["image"], example["label"], example["caption"] = self.preprocess_image(self.index[i])
        return example


class ImageNetWebDataset(IterableDataset):
    def __init__(
        self,
        imagenet_tar=None,
        imagenet_index=None,
        split="train",
        size=None,
        random_crop=False,
        aug_mode="default",
        buffer_size=1000,
    ):
        self.size = size
        self.random_crop = random_crop
        self.aug_mode = aug_mode

        if aug_mode == "default":
            assert self.size is not None and self.size > 0
            self.rescaler = T.Resize(self.size)
            if not self.random_crop:
                self.cropper = T.CenterCrop((self.size, self.size))
            else:
                self.cropper = T.RandomCrop((self.size, self.size))
            self.preprocessor = T.Compose([self.rescaler, self.cropper])
        else:
            raise NotImplementedError

        hf_token = get_token()
        if hf_token is None:
            raise ValueError(
                "Hugging Face token not found. Please login using `huggingface-cli login`."
            )

        if split == "train":
            self.url = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-train-{{0000..1023}}.tar"
            self.length = 1281167
        elif split == "val" or split == "validation":
            self.url = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-validation-{{0000..0063}}.tar"
            self.length = 50000
        else:
            raise ValueError(f"Unknown split: {split}")

        # Use curl to get around redirection issues
        self.url = f"pipe:curl -s -L '{self.url}' -H 'Authorization:Bearer {hf_token}'"

        self.captions = {}
        captions_df = pd.read_csv(
            "imagenet_gemma3_12b.tsv",
            sep="\t",
            header=None,
            names=["file_name", "caption"],
        )
        for _, row in captions_df.iterrows():
            file_name, caption = row["file_name"], row["caption"]
            self.captions[file_name] = caption

        self.dataset = (
            wds.WebDataset(self.url, shardshuffle=100)
            .shuffle(buffer_size)
            .decode("pil")
            .rename(image="jpg;jpeg", label="cls")
            .map(self.process_sample)
            .with_length(self.length)
        )

    def process_sample(self, sample):
        image = self.preprocessor(sample["image"])
        image = np.array(image)
        image = (image / 127.5 - 1.0).astype(np.float32)

        key = sample["__key__"]
        caption = self.captions.get(f"{key}.JPEG", "") # Assume .JPEG extension

        return {
            "image": image,
            "label": sample["label"],
            "caption": caption,
        }

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length
