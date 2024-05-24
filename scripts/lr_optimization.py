import argparse

import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
from torchvision import transforms
from model.image_restoration_model import create_model
from data.loss_utils import lx, lm
from dataset import CorruptedImagesDataset
import os
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./validation/",
        help="Directory with training data",
    )
    parser.add_argument(
        "--lr_find_epochs", type=int, default=5, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
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

    # Transform for converting numpy arrays to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    print(f"Making dataset...")
    # Set up dataset and dataloader
    dataset = CorruptedImagesDataset(args.data_dir, transform=transform)

    # Assuming train_dataset is your original large training dataset
    subset_indices = np.random.choice(
        len(dataset), size=int(len(dataset) * 0.1), replace=False
    )  # 10% of the dataset
    train_subset = Subset(dataset, subset_indices)

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"train_dataloader created...")

    # -------------------------------------------------------------------------------
    # Load the PyTorch Model
    # -------------------------------------------------------------------------------
    model = create_model()
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-7)

    print("model initialized...")

    lr_find_epochs = args.lr_find_epochs
    lr_increase = (1e-1 / 1e-7) ** (1 / (len(train_dataloader) * lr_find_epochs))

    lr_history = {"lrs": [], "losses": []}

    # Training loop
    for epoch in range(lr_find_epochs):
        # print(f"epoch: {epoch}")
        for corrupted_imgs, binary_masks, src_imgs in tqdm.tqdm(train_dataloader):
            # Move data to the correct device
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

            # Record the learning rate and loss
            lr = optimizer.param_groups[0]["lr"]
            lr_history["lrs"].append(lr)
            lr_history["losses"].append(loss.item())

            # print(f"lr: {lr}")
            # print(f"loss: {loss.item()}")

            # Update the learning rate
            optimizer.param_groups[0]["lr"] *= lr_increase

    # At the end of the training loop
    json_path = os.path.join(args.output_dir, "lr_history.json")
    with open(json_path, "w") as f:
        json.dump(lr_history, f)

    print(f"Learning rate history saved to {json_path}")
