# Import necessary libraries
import cv2
from PIL import Image
import torch
import numpy as np
from torchvision import transforms as TF
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# Load the trained model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')

# Replace with the actual number of classes
model.config.num_labels = 2  

# Load the state from the fine-tuned model and set to model.eval() mode
model.load_state_dict(torch.load('Segformer/best_model.pth'))
model.to(device)
model.eval()

# Load an image
image_path = 'Segformer/img and vdos/final road.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Error: Image could not be read. Check the path.")
    exit()

# Perform transformations
data_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.Resize((360, 640)),
    TF.ToTensor(),
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess the image
input_tensor = data_transforms(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    outputs = model(pixel_values=input_tensor, return_dict=True)
    outputs = F.interpolate(outputs["logits"], size=(360, 640), mode="bilinear", align_corners=False)
    
    preds = torch.argmax(outputs, dim=1)
    preds = torch.unsqueeze(preds, dim=1)
    predicted_mask = (torch.sigmoid(preds) > 0.5).float()

# Create an RGB version of the mask to overlay on the original image
mask_np = predicted_mask.cpu().squeeze().numpy()
mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

mask_rgb = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)
mask_rgb[:, :, 1] = (mask_resized * 255).astype(np.uint8)  # Set only the green channel

# Post-processing for mask smoothening
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(mask_rgb, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Overlay the mask on the image
blended = cv2.addWeighted(image, 0.65, closing, 0.6, 0)

# Save the processed image
cv2.imwrite('Segformer/outputs/output_final_road.jpg', blended)
