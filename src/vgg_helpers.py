import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import streamlit as st
from .utils import DEVICE

CONTENT_WEIGHT_VGG = 1
STYLE_WEIGHT_VGG = 1e6

CONTENT_LAYERS_VGG = ['conv_4']
STYLE_LAYERS_VGG = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Define VGG normalization here as it's specific to the pre-trained model
vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)

vgg_normalization_transform = transforms.Normalize(vgg_normalization_mean, vgg_normalization_std)

@st.cache_resource(show_spinner="Loading VGG Model...")
def load_vgg_model():
    """Loads the pretrained VGG19 Model"""
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg

# Get the single instance of the loaded model
vgg_model = load_vgg_model()

def get_features(image):
    """Extracts features from specified VGG layers using the cached model."""
    model = vgg_model
    normalization = vgg_normalization_transform

    layers = {
        '0': 'conv_1',  # layer 1
        '5': 'conv_2',  # layer 6
        '10': 'conv_3', # layer 11
        '19': 'conv_4', # layer 20
        '28': 'conv_5'  # layer 29
    }
    features = {}
    x = image
    # Apply normalization expected by VGG *before* passing to the model
    x = normalization(x)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            # Stop if we've extracted the last needed style layer
            if layers[name] == STYLE_LAYERS_VGG[-1] and layers[name] in STYLE_LAYERS_VGG:
                break
            # Also check if the single content layer is found if it's later
            if layers[name] == CONTENT_LAYERS_VGG[0] and layers[name] not in STYLE_LAYERS_VGG:
                # This condition might need refinement based on exact layer order needs
                pass # Continue if only content layer found but style layers still needed

    # Ensure all required layers were found (optional check)
    required_layers = set(CONTENT_LAYERS_VGG + STYLE_LAYERS_VGG)
    if not required_layers.issubset(features.keys()):
        st.warning(f"Could not extract all required VGG layers. Found: {list(features.keys())}")

    return features

def gram_matrix(tensor):
    """Calculates the Gram Matrix."""
    if tensor is None or tensor.dim() != 4:
        st.error(f"Invalid input for Gram matrix: shape {tensor.shape if tensor is not None else 'None'}")
        return None
    B, C, H, W = tensor.size()
    tensor = tensor.view(B * C, H * W)
    G = torch.mm(tensor, tensor.t())
    return G.div(B * C * H * W)

# Define Loss instances here
ContentLoss = nn.MSELoss().to(DEVICE)
StyleLoss = nn.MSELoss().to(DEVICE)