import torch
import torch.optim as optim
import streamlit as st
from .vgg_helpers import (
    get_features, gram_matrix, ContentLoss, StyleLoss,
    CONTENT_LAYERS_VGG, STYLE_LAYERS_VGG,
    CONTENT_WEIGHT_VGG, STYLE_WEIGHT_VGG
)
from .utils import DEVICE

def get_dual_style_features(style_image1, style_image2):
    """Combines features from two style images."""
    style_features1 = get_features(style_image1)
    style_features2 = get_features(style_image2)
    final_style_features = {} 
    missing_layers = []
    for layer in STYLE_LAYERS_VGG:
        if layer not in style_features1 or layer not in style_features2:
            missing_layers.append(layer)
            continue # Skip if a layer is missing in either style image

        sf1 = style_features1[layer]
        sf2 = style_features2[layer]
        sf1_size = sf1.size(1) // 4
        
        sf1_part = sf1[:, :sf1_size, :, :] # extract channels
        sf2_part = sf2[:, sf1_size:, :, :]
        sf = torch.cat((sf1_part, sf2_part), dim=1)
        final_style_features[layer] = sf
    
    if missing_layers:
        st.warning(f"Could not extract dual style features for layers: {missing_layers}")
    return final_style_features

def get_rot_style_features(style_image):
    """Modifies style features using rotation."""
    style_features = get_features(style_image)
    final_rot_style_features = {}
    missing_layers = []
    for layer in STYLE_LAYERS_VGG:
        if layer not in style_features:
            missing_layers.append(layer)
            continue
        sf = style_features[layer]
        # Ensure gradients are not tracked for these operations on the *style* features
        with torch.no_grad():
            sf_rot90 = torch.rot90(sf, k=1, dims=(2, 3))
            sf_rot180 = torch.rot90(sf, k=2, dims=(2, 3))
            # Calculate final feature: sf + (sf_rot90 - sf_rot180)
            final_rot = sf + (sf_rot90 - sf_rot180)
        final_rot_style_features[layer] = final_rot

    if missing_layers:
        st.warning(f"Could not extract rotated style features for layers: {missing_layers}")
    return final_rot_style_features

def run_vgg_optimization(target_img_initial, content_features, style_features, steps, progress_bar):
    """Runs the VGG optimization loop."""
    target_img = target_img_initial.clone().requires_grad_(True).to(DEVICE)
    optimizer = optim.Adam([target_img], lr=0.02, weight_decay=1e-4)  # Added weight_decay

    for step in range(steps):
        optimizer.zero_grad()

        # Clamp target image values to [0, 1] before processing
        with torch.no_grad():
            target_img.clamp_(0, 1)

        target_features = get_features(target_img)  # Extract features from current target

        # Calculate content and style losses
        content_loss, missing_content_layers = calculate_content_loss(target_features, content_features)
        style_loss, missing_style_layers = calculate_style_loss(target_features, style_features)

        handle_missing_layers(step, missing_content_layers, missing_style_layers)

        # Backpropagate and Optimize
        total_loss = (content_loss * CONTENT_WEIGHT_VGG) + (style_loss * STYLE_WEIGHT_VGG)
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            st.error(f"Stopping VGG optimization at step {step+1} due to NaN/Inf loss.")
            break  # Stop optimization if loss becomes invalid
        total_loss.backward()
        optimizer.step()

        # Update progress bar
        if progress_bar:
            progress_bar.progress((step + 1) / steps)

    # Final clamp
    with torch.no_grad():
        target_img.clamp_(0, 1)

    return target_img.detach()  # Return the final image tensor


def calculate_content_loss(target_features, content_features):
    """Calculate content loss."""
    content_loss = 0
    missing_content_layers = []
    for layer in CONTENT_LAYERS_VGG:
        if layer not in target_features or layer not in content_features:
            missing_content_layers.append(layer)
            continue
        content_loss += ContentLoss(target_features[layer], content_features[layer])
    return content_loss, missing_content_layers


def calculate_style_loss(target_features, style_features):
    """Calculate style loss."""
    style_loss = 0
    missing_style_layers = []
    for layer in STYLE_LAYERS_VGG:
        if layer not in target_features or layer not in style_features:
            missing_style_layers.append(layer)
            continue
        target_gram = gram_matrix(target_features[layer])
        with torch.no_grad():
            style_gram = gram_matrix(style_features[layer])
        if target_gram is not None and style_gram is not None:
            style_loss += StyleLoss(target_gram, style_gram)
        else:
            missing_style_layers.append(f"{layer} (Gram calc failed)")
    return style_loss, missing_style_layers


def handle_missing_layers(step, missing_content_layers, missing_style_layers):
    """Handle missing layers by logging warnings."""
    if missing_content_layers or missing_style_layers:
        st.warning(f"Step {step+1}: Missing layers - Content: {missing_content_layers}, Style: {missing_style_layers}")