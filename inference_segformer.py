
base_path = "./"

import os
import cv2
import json
import numpy as np
import torch
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.optim import AdamW
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

import torch.backends.cudnn as cudnn
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
cudnn.deterministic = True
cudnn.benchmark = False

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def compute_iou(pred, target):
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    return (intersection ) / (union )

def compute_dice(pred, target):
    intersection = (pred * target).float().sum((1, 2))
    return (2. * intersection ) / (pred.float().sum((1, 2)) + target.float().sum((1, 2)) )

class PipeWallDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        #self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        ###
        ###
        image_path = self.image_paths[idx]
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"[ERROR] Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) // 255

        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        #mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=image)
            image= augmented['image']

        return image.float() / 255.0
        #return image / 255.0

transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

import os

val_img_path = os.path.join(base_path, "dataset_segment_blades/Blades_combined_dataset")
val_images = os.listdir(val_img_path)
for i in range(len(val_images)):
    val_images[i] = os.path.join(val_img_path, val_images[i])
val_images

import os

#val_masks_path = os.path.join(base_path, "test_masks")
#val_masks = os.listdir(val_masks_path)
#for i in range(len(val_masks)):
#    val_masks[i] = os.path.join(val_masks_path, val_masks[i])
#val_masks

val_images = sorted(val_images)
val_images

#val_masks = sorted(val_masks)
#val_masks[10] , val_masks[11] = val_masks[11] , val_masks[10]
#val_masks

test_dataset = PipeWallDataset(val_images, transform)
#test_dataset = PipeWallDataset(val_images)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b3")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

model_path = os.path.join(base_path, "GPU_trained_blade_model.pth")
#model_path

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

def overlay_predictions(model, loader, device, output_dir="./content/"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    unique_colors = [(255, 0, 0)]
    total_iou = 0
    total_dice = 0
    total_images = 0

    with torch.no_grad():
        for batch_idx, (images) in enumerate(tqdm(loader, desc="Generating Predictions")):
            images = images.to(device)
            #masks = masks.to(device)
            outputs = model(pixel_values=images).logits
            preds = outputs.argmax(dim=1)

            preds_resized = torch.nn.functional.interpolate(
                preds.unsqueeze(1).float(), size=(512, 512),
                mode='bilinear', align_corners=False
            ).squeeze(1).long()
            

            preds_np = preds_resized.cpu().numpy()
            #masks_np = masks.cpu().numpy()
            images_np = images.cpu().numpy()

            preds_resized_cv = [cv2.resize(pred.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) for pred in preds_np]

            for i, (image, pred_cv, pred_tensor) in enumerate(zip(images_np, preds_resized_cv, preds_resized)):
#                iou = compute_iou(pred_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)).item()
#                dice = compute_dice(pred_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)).item()
#                print(f"[Image {total_images + 1}] IOU: {iou:.4f}, Dice: {dice:.4f}")

#                total_iou += iou
#                total_dice += dice
#                total_images += 1

                image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                overlay = image.copy()

                class_idx = 1
                pred_mask = (pred_cv == class_idx).astype(np.uint8) * 255
                contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(overlay, contours, unique_colors[0])

                alpha = 0.5
                final_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0.0)

                output_path = os.path.join(output_dir, f"pred_{batch_idx}_{i}.png")
                #cv2.imwrite(output_path, final_image)

                #mask = (preds == target_class).astype(np.uint8)
                pred_mask = cv2.GaussianBlur(pred_mask, (5, 5), 0)
                masked_image = cv2.bitwise_and(image, image, mask=pred_mask)
                output_path = os.path.join(output_dir, final_image)
                cv2.imwrite(output_path, masked_image)

#                plt.figure(figsize=(15, 5))

#                plt.subplot(1, 3, 1)
#                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#                plt.title("Original Image")
#                plt.axis("off")

#                plt.subplot(1, 3, 2)
#                plt.imshow(mask_tensor.cpu().numpy(), cmap='gray')
#                plt.title("Ground Truth Mask")
#                plt.axis("off")

#                plt.subplot(1, 3, 3)
#                plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
#                plt.title("Segmented Output")
#                plt.axis("off")

#                plt.show()

#    avg_iou = total_iou / total_images
#    avg_dice = total_dice / total_images

#    print(f"\n[Overall] Average IOU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}")
    return 1,1


iou_avg, dice_avg = overlay_predictions(model, test_loader, device)

print(f"Avg IOU: {iou_avg}, Avg Dice coeff: {dice_avg}")


