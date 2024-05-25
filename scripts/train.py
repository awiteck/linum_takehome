"""
File: train.py
Description: Train the image restoration model.
"""

import argparse

import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from model.image_restoration_model import create_model
from data.loss_utils import lx, lm
from dataset import CorruptedImagesDataset
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./validation/",
        help="Directory with training data",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save model weights",
    )
    parser.add_argument(
        "--binary-mask-threshold",
        type=float,
        default=0.1,
        help="Threshold for converting the predicted mask to a binary mask. \
        Predicted values greater than or equal to this threshold will get rounded to 1.0 \
        and marked as a corrupted pixel when creating the reconstructed image.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(f"making dataset...")

    dataset = CorruptedImagesDataset(args.data_dir, transform=transform)

    total_samples = len(dataset)
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"dataloaders made...")

    # -------------------------------------------------------------------------------
    # Load the PyTorch Model
    # -------------------------------------------------------------------------------
    model = create_model()
    model.to(device)

    model.train()

    print(f"model made...")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    loss_history = {
        "train_total": [],
        "train_lx": [],
        "train_lm": [],
        "val_total": [],
        "val_lx": [],
        "val_lm": [],
    }

    # Training loop
    for epoch in range(args.epochs):
        print(f"epoch: {epoch}")
        model.train()
        total_train_loss = 0
        train_lx_losses = []
        train_lm_losses = []
        for corrupted_imgs, binary_masks, src_imgs in tqdm.tqdm(train_dataloader):
            corrupted_imgs = corrupted_imgs.to(device)
            binary_masks = binary_masks.to(device).unsqueeze(1)
            src_imgs = src_imgs.to(device)

            optimizer.zero_grad()
            predicted_imgs, predicted_masks = model(corrupted_imgs)

            lx_loss = lx(src_imgs, predicted_imgs, binary_masks)
            lm_loss = lm(predicted_masks, binary_masks)
            loss = 2 * lx_loss + lm_loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            loss_history["train_total"].append(loss.item())
            loss_history["train_lx"].append(lx_loss.item())
            loss_history["train_lm"].append(lm_loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {avg_train_loss}")

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for corrupted_imgs, binary_masks, src_imgs in tqdm.tqdm(val_dataloader):
                corrupted_imgs = corrupted_imgs.to(device)
                binary_masks = binary_masks.to(device).unsqueeze(1)
                src_imgs = src_imgs.to(device)

                predicted_imgs, predicted_masks = model(corrupted_imgs)

                lx_loss = lx(src_imgs, predicted_imgs, binary_masks)
                lm_loss = lm(predicted_masks, binary_masks)
                val_loss = 2 * lx_loss + lm_loss
                total_val_loss += val_loss.item()

                loss_history["val_total"].append(val_loss.item())
                loss_history["val_lx"].append(lx_loss.item())
                loss_history["val_lm"].append(lm_loss.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")

    # At the end of the training loop
    json_path = os.path.join(args.output_dir, "loss_history.json")
    with open(json_path, "w") as f:
        json.dump(loss_history, f)

    print(f"Loss history saved to {json_path}")
