
import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from tqdm import tqdm
import torchvision.transforms as T

# ==== Configuration ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "./dataset_segment_blades/Blades_combined_data"          # Directory with original annotated images
output_dir = "./masked_blades"        # Directory to save masked blade-only images
model_path = "./best_model.pth"           # Trained SegFormer model
target_class = 1                      # Assuming blade class = 1
image_size = (512, 512)

os.makedirs(output_dir, exist_ok=True)

# ==== Load Model ====
#model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
#feature_extractor = SegformerFeatureExtractor.from_pretrained(model_path)
#model.eval()
from transformers import SegformerForSemanticSegmentation

base_model = "nvidia/segformer-b0-finetuned-ade-512-512"  # or whatever backbone you used

model = SegformerForSemanticSegmentation.from_pretrained(base_model)
model.load_state_dict(torch.load("./best_model.pth", map_location="cpu"))
model.to(device)
model.eval()
# ==== Preprocessing ====
preprocess = T.Compose([
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# ==== Postprocess: Smooth the mask ====
def smooth_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ==== Inference Loop ====
for img_name in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    orig_np = np.array(image)

    # Preprocess and forward pass
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    # Generate binary mask for the target class (blade)
    mask = (preds == target_class).astype(np.uint8) * 255
    mask = smooth_mask(mask)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(orig_np, orig_np, mask=mask)

    # Save result
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, masked_image)

print(f"✅ Done. Masked images saved in: {output_dir}")
