# @Time    : 2026-01-13 16:01
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : finetune.py
# Finetune a StrokeNet checkpoint on my own collected data

from src.data.utils import build_img, build_char_map, char_to_folder_name, load_chars
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.ml.architectures.cnn import StrokeNet
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def main():
    ###############################
    # args
    ###############################
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=-1,
                        help="Number of trials per character to use for finetuning: \
                            default uses min trials of any character")
    parser.add_argument("--img-size", type=int, default=28, help="Downsample size of img (default 28x28)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--ckpt", type=str, required=True, 
                        help="wandb run folder name from emnist_pretrain.py to finetune from; uses best.ckpt")
    parser.add_argument("--epochs", type=int, default=100, help="max_epochs to train")
    parser.add_argument("--seed", type=int, default=42, help="seed to use")
    parser.add_argument("--debug", action="store_true", help="If set, show example images from dataset")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    ###############################
    # Finetuning
    ###############################
    L.seed_everything(args.seed, workers=True)
    # 1. Prepare a Lightning DataModule for finetuning (use preprocessing from
    #    main.py)
    data_module = MouseStrokeDataModule(
        cache_dir="raw_mouse_data",
        img_size=args.img_size,
        max_trials=args.n_trials,
        batch_size=args.batch_size,
        num_workers=0,
        augment=not args.no_augment,
    )
    data_module.setup()

    # 1.5 DEBUG: Show example images from dataset
    if args.debug:
        debug_image_display(data_module.train_dataloader(), data_module.id2label)

    # 2. Load a StrokeNet checkpoint (from emnist_pretrain.py) and replace head
    model = StrokeNet.load_from_checkpoint(
        f"checkpoints/emnist_pretrain/{args.ckpt}/best.ckpt",
        num_classes=27,  # match pretrained head size for loading
        finetune=True,
    )
    model.replace_classifier(num_classes=53) # replace with uninitialized 53-class head

    # 3. Finetune on my data
    print(
        f"Commencing finetuning with "
        f"{len(data_module.train_dataset)} train / "
        f"{len(data_module.val_dataset)} val / "
        f"{len(data_module.test_dataset)} test "
        f"(total {len(data_module.train_dataset)+len(data_module.val_dataset)+len(data_module.test_dataset)})"
    )
    tags = ["finetune", "uppercase"]
    if not args.no_augment:
        tags.append("augmented")
    wandblogger = WandbLogger(project="scribble", tags=tags)
    ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/mouse_finetune/{wandblogger.experiment.name}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    trainer = L.Trainer(
        min_epochs=5,
        max_epochs=args.epochs,
        accelerator="auto",
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=wandblogger,
        enable_checkpointing=True,
        callbacks=[ckpt],
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, datamodule=data_module)

    # 5. Eval and save the finetuned model
    trainer.test(
        ckpt_path="best", datamodule=data_module
    )  # auto selects test dataloader

    print("\nFinetuning complete!")
    print("----------------------\nINFO:\n")
    print("Checkpoint name: ", wandblogger.experiment.name)
    print("Pretrain checkpoint: ", args.ckpt)
    print("Total training samples: ", len(data_module.train_dataset))
    print("Batch size: ", args.batch_size)
    print("Epochs: ", args.epochs)
    print("Seed: ", args.seed)
    print("Data augmentation: ", "enabled" if not args.no_augment else "disabled")
    print(f"Final test accuracy: {trainer.callback_metrics['test_acc']*100:.1f}%")
    print(f"\nWeights saved to checkpoints/mouse_finetune/{wandblogger.experiment.name}")

class MouseStrokeDataModule(L.LightningDataModule):
    def __init__(self, cache_dir="raw_mouse_data", img_size=28, max_trials=-1, batch_size=128, num_workers=0, augment=True):
        super().__init__()
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.augment = augment

        # find character with min trials if max_trials <=0
        self.max_trials = max_trials
        min_trial_char = None
        if self.max_trials <= 0:
            self.max_trials = float('inf')
            for char in os.listdir(self.cache_dir):
                char_dir = os.path.join(self.cache_dir, char)
                num_trials = len(os.listdir(char_dir))
                if num_trials < self.max_trials:
                    self.max_trials = num_trials
                    min_trial_char = char
            print(f"Using min trials across characters: {self.max_trials} (from '{min_trial_char}')")

        self.batch_size = batch_size
        self.num_workers = num_workers

        # use shared char map from utils
        self.id2label = build_char_map()

        # data augmentation for training (applied on-the-fly)
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # up to 10% shift
                scale=(0.9, 1.1),      # up to 10% scale
                fill=0,
            ),
            transforms.RandomResizedCrop(
                size=28,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
        ]) if augment else None

    def setup(self, stage=None):
        """
        Prepare dataset:
        1. Load data I've labelled so far
        2. Preprocess data into (1, 28, 28) images
        3. Create TensorDatasets for train/val/test
        """
        # Load collected data by reading raw_mouse_data/<char>/*.csv files
        # 0->a, 1->b, ..., 25->z, 26->space
        
        idx_to_arrays = {}
        for char_idx, char_value in self.id2label.items():
            folder_name = char_to_folder_name(char_value)
            _, character_arrays = load_chars(f"{self.cache_dir}/{folder_name}", max_trials=self.max_trials, silent=True)
            idx_to_arrays[char_idx] = character_arrays

        # Preprocess character_arrays into images and create TensorDatasets
        images = []
        labels = []
        for char_idx, char_arrays in idx_to_arrays.items():
            for arr in char_arrays:
                x, y = arr
                x_tensor = torch.from_numpy(x).to(dtype=torch.float32)
                y_tensor = torch.from_numpy(y).to(dtype=torch.float32)
                img = build_img(x_tensor, y_tensor, invert_colors=False, downsample_size=self.img_size) # from main.py preprocessing
                images.append(img.unsqueeze(0)) # add channel dim
                labels.append(char_idx)

        images_tensor = torch.stack(images) # (N, 1, H, W)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        del idx_to_arrays, images, labels # some cleanup
        
        dataset = TensorDataset(images_tensor, labels_tensor)

        # Split into train/val/test (70/10/20)
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def _train_collate_fn(self, batch):
        """Collate function with augmentation for training batches"""
        images, labels = zip(*batch)
        if self.train_transform:
            images = [self.train_transform(img) for img in images]
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._train_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

def debug_image_display(loader : DataLoader, id2label):
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        tensor_img, label = loader.dataset[i]
        image = to_pil_image(tensor_img)
        label = id2label[label.item()]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
