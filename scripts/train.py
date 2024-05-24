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
from data.loss_utils import custom_loss
from dataset import CorruptedImagesDataset
import os


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
    device = torch.device(device)

    # Transform for converting numpy arrays to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Set up dataset and dataloader
    dataset = CorruptedImagesDataset(args.data_dir, transform=transform)

    total_samples = len(dataset)
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------------
    # Load the PyTorch Model
    # -------------------------------------------------------------------------------
    model = create_model()
    model.to(device)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(args.epochs):
        total_train_loss = 0
        for corrupted_imgs, binary_masks, src_imgs in tqdm.tqdm(train_dataloader):
            # Move data to the correct device
            corrupted_imgs = corrupted_imgs.to(device)
            binary_masks = binary_masks.to(device)
            binary_masks = binary_masks.unsqueeze(1)
            src_imgs = src_imgs.to(device)

            optimizer.zero_grad()
            predicted_imgs, predicted_masks = model(corrupted_imgs)

            loss = custom_loss(src_imgs, predicted_imgs, binary_masks, predicted_masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

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
                val_loss = custom_loss(
                    src_imgs, predicted_imgs, binary_masks, predicted_masks
                )
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")

    """
    # -------------------------------------------------------------------------------
    # Generate predictions for validation set
    # -------------------------------------------------------------------------------
    corrupted_img_paths = list(glob.glob(f"{args.data_dir}/corrupted_imgs/*.png"))

    predicted_imgs_out_dir = f"{args.output_dir}/predicted_imgs"
    reconstructed_imgs_out_dir = f"{args.output_dir}/reconstructed_imgs"
    binary_masks_out_dir = f"{args.output_dir}/binary_masks"

    os.makedirs(predicted_imgs_out_dir, exist_ok=True)
    os.makedirs(reconstructed_imgs_out_dir, exist_ok=True)
    os.makedirs(binary_masks_out_dir, exist_ok=True)

    with torch.inference_mode():
        for corrupted_img_path in tqdm.tqdm(corrupted_img_paths):
            # Read the image into memory and normalize values between -1 and 1,
            corrupted_img = read_rgba_img(img_path=corrupted_img_path)
            norm_corrupted_img = torch.from_numpy(normalize_rgb_img(img=corrupted_img))
            norm_corrupted_img = norm_corrupted_img.to("cuda")
            norm_corrupted_img = norm_corrupted_img.unsqueeze(
                0
            )  # The model expects a batch dimension

            # Generate a prediction
            predicted_img, predicted_mask = image_restoration_model(norm_corrupted_img)

            # Move to CPU and numpy
            norm_corrupted_img = (
                norm_corrupted_img.detach().cpu().numpy()
            )  # 1 x c x h x w
            predicted_img = predicted_img.detach().cpu().numpy()  # 1 x c x h x w
            predicted_mask = predicted_mask.detach().cpu().numpy()  # 1 x 1 x h x w

            # Convert the predicted mask to a binary mask, based on the threshold
            binary_mask = (predicted_mask >= args.binary_mask_threshold).astype(int)

            # Within the binary mask, the 1s indicate corrupted pixels and 0s indicate
            # uncorrupted pixels. For all the pixels with value 1, copy the "fixed" value
            # from the `predicted_img`. For all the pixels with value 0, copy the "original"
            # from the `norm_corrupted_img`.
            # Match dims of predicted_img (1 x 1 x h x w -> 1 x 3 x h x w)
            expanded_binary_mask = np.repeat(binary_mask, repeats=3, axis=1)
            reconstructed_img = (
                predicted_img * expanded_binary_mask
                + (1 - expanded_binary_mask) * norm_corrupted_img
            )

            # Let's un-normalize the images and masks
            reconstructed_pil_img = convert_img_ndarray_to_pil_img(
                norm_img=np.squeeze(reconstructed_img, axis=0)
            )
            predicted_pil_img = convert_img_ndarray_to_pil_img(
                norm_img=np.squeeze(predicted_img, axis=0)
            )
            binary_mask_pil_img = convert_mask_ndarray_to_pil_img(
                mask=np.squeeze(binary_mask, axis=(0, 1))
            )

            # Let's save the images and masks
            filename = os.path.basename(corrupted_img_path)
            reconstructed_pil_img.save(
                os.path.join(reconstructed_imgs_out_dir, filename)
            )
            predicted_pil_img.save(os.path.join(predicted_imgs_out_dir, filename))
            binary_mask_pil_img.save(os.path.join(binary_masks_out_dir, filename))
    """
