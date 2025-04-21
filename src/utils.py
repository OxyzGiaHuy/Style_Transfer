import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import io

# --- Configuration ---
IMSIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transforms = transforms.Compose([
    transforms.Resize((IMSIZE, IMSIZE)),
    transforms.ToTensor(),
    # VGG Normalization is applied within the model forward pass in vgg_helpers
])

unloader = transforms.ToPILImage() # convert tensor to PIL

def image_loader(image_bytes):
    """Loads an image from bytes, transforms, and sends to device."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = img_transforms(image).unsqueeze(0) # Add batch dimension
        return image.to(DEVICE, torch.float)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def tensor_to_pil(tensor):
    """Converts a tensor image to a PIL image."""
    if tensor is None:
        return None
    image = tensor.cpu().clone()
    if image.dim() == 4: # Handle batch dimension if present
        image = image.squeeze(0)
    # Ensure tensor is in CHW format before unloader
    if image.dim() != 3:
         st.warning(f"Tensor shape unexpected for conversion to PIL: {image.shape}")
         return None # Or handle differently
    try:
        image = unloader(image)
        return image
    except Exception as e:
        st.error(f"Error converting tensor to PIL: {e}, Tensor shape: {tensor.shape}")
        return None