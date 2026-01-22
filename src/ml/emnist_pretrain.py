# @Time    : 2026-01-13 08:04
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : pretrain.py

# load a public (lowercase!) EMNIST dataset & train our basic CNN on it
# this is the pretraining phase of scribble
#
# We'll pretrain with 27 output classes (lowercase a-z + space) to match EMNIST
# During finetuning, the classifier head will be replaced with a new 53-class head

from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from src.ml.architectures.cnn import StrokeNet
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.utils import build_img
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--img-size", type=int, default=28, help="Image size (default 28x28)")
    args = parser.parse_args()
    ###############################
    # Prepare EMNIST
    ###############################
    print("Preparing dataset...")
    data_module = EMNISTLettersDataModule(batch_size=args.batch_size, num_workers=0, downsample_size=args.img_size)
    data_module.prepare_data()
    data_module.setup()

    # dislay sample images from the dataset
    train_data = data_module.train_data
    id2label = data_module.id2label

    # debug displaying
    display_image(train_data, id2label)

    ###############################
    # Training CNN
    ###############################
    print(f"Commencing training with {len(data_module.train_data)} training samples...")
    L.seed_everything(42, workers=True)
    model = StrokeNet(num_classes=27, dropout_p=0.1, finetune=False)
    wandblogger = WandbLogger(project="scribble", tags=["pretrain", "lowercase"])

    ckpt = ModelCheckpoint(
        dirpath=f"checkpoints/emnist_pretrain/{wandblogger.experiment.name}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    trainer = L.Trainer(
        min_epochs=5,
        max_epochs=20,
        accelerator="auto",
        limit_train_batches=1.0,  # portion of train data to use
        limit_val_batches=1.0,  # portion of test data to use
        logger=wandblogger,
        enable_checkpointing=True,
        callbacks=[ckpt],
        check_val_every_n_epoch=1, # val & checkpoint this frequency
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    print("\nPretraining complete!")
    print(
        f"Pretrained model checkpoints saved to checkpoints/emnist_pretrain/{wandblogger.experiment.name}"
    )


def display_image(dataset, id2label):
    """
    Display 9 images from the dataset with their labels
    """
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        # images need to be rotated 90 deg clockwise and flipped horizontally
        image = dataset[i]["image"]
        image = image.rotate(-90, expand=True)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

        label = id2label[dataset[i]["label"]]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# note: letters are lowercase drawn, with uppercase labels
# for this first test (sanity checking the pretrain + finetune) we'll just go
# ahead with lowercase letter detection
class EMNISTLettersDataModule(L.LightningDataModule):
    def __init__(self, cache_dir="./emnist_data", batch_size=64, num_workers=0, downsample_size=64):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        # batch size tldr: 64 is 18s/epoch on m2 pro (72% usage GPU), 128 is 14s/epoch (86% usage)
        self.num_workers = num_workers
        self.downsample_size = downsample_size
        self.to_tensor = transforms.ToTensor()

        self.id2label = None
        self.train_data = None
        self.val_data = None

    def prepare_data(self):
        load_dataset("tanganke/emnist_letters", cache_dir=self.cache_dir)

    def setup(self, stage=None):
        dataset = load_dataset("tanganke/emnist_letters", cache_dir=self.cache_dir)
        self.train_data = dataset["train"]
        self.val_data = dataset["test"] # test used as val; no val set

        label_feature = dataset["train"].features["label"]
        self.id2label = label_feature.names

    @staticmethod
    def _fix_one(pil_img, downsample_size):
        pil_img = pil_img.resize((downsample_size, downsample_size)) 
        pil_img = pil_img.rotate(-90, expand=True)
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        return pil_img

    def collate_fn(self, batch):
        # batch: list of samples (each sample is a dict like {"x": tensor, "y": int})
        xs = []
        ys = []
        for ex in batch:
            img = self._fix_one(ex["image"], self.downsample_size)
            xs.append(self.to_tensor(img))
            ys.append(int(ex["label"]))

        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False, # bad on mps
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        # just using val for test
        return self.val_dataloader()


if __name__ == "__main__":
    main()
