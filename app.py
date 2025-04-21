import streamlit as st
from PIL import Image
import torch # Keep torch import if used directly for checks like isnan

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🎨 Neural Style Transfer Showcase")

# Import functions from our modules
from src.utils import image_loader, tensor_to_pil, IMSIZE, DEVICE
from src.vgg_helpers import get_features as get_vgg_features # Rename for clarity
from src.vgg_styles import (
    get_dual_style_features, get_rot_style_features, run_vgg_optimization
)
from src.stytr_helpers import setup_stytr, run_stytr_inference

# --- Configuration ---
DEFAULT_VGG_STEPS = 100 # Reduced steps for faster demo



# --- Run Setup ---
# Attempt StyTR setup early. If it fails, the app might stop here or handle errors later.
setup_ok = setup_stytr()
if not setup_ok:
    st.error("StyTR setup failed. Cannot proceed with Transformer style transfer.")
    # Optionally allow VGG styles to continue if VGG model loaded ok
    # For simplicity here, we might just stop or disable the button


# --- Sidebar for Uploads and Controls ---
st.sidebar.header("Upload Images")
content_file = st.sidebar.file_uploader("1. Content Image", type=["jpg", "jpeg", "png"])
style_file1 = st.sidebar.file_uploader("2. Style Image 1", type=["jpg", "jpeg", "png"])
style_file2 = st.sidebar.file_uploader("3. Style Image 2 (for Dual Style)", type=["jpg", "jpeg", "png"])

st.sidebar.header("VGG Optimization")
vgg_steps = st.sidebar.slider(
    "Number of VGG Steps (Dual & Rot)",
    min_value=10, max_value=500, value=DEFAULT_VGG_STEPS, step=10,
    help="Higher steps improve quality but take longer."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Using Device: {DEVICE}")
st.sidebar.markdown("Built with Streamlit and PyTorch.")


# --- Display Uploaded Images ---
col_orig1, col_orig2, col_orig3 = st.columns(3)
content_img_display = None
style1_img_display = None
style2_img_display = None

if content_file:
    try:
        content_img_display = Image.open(content_file).convert('RGB')
        with col_orig1:
            st.image(content_img_display, caption='Content Image', use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying content image: {e}")
if style_file1:
     try:
        style1_img_display = Image.open(style_file1).convert('RGB')
        with col_orig2:
            st.image(style1_img_display, caption='Style Image 1', use_container_width=True)
     except Exception as e:
        st.error(f"Error displaying style 1 image: {e}")
if style_file2:
    try:
        style2_img_display = Image.open(style_file2).convert('RGB')
        with col_orig3:
            st.image(style2_img_display, caption='Style Image 2', use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying style 2 image: {e}")
else:
     with col_orig3:
        st.markdown("(Optional for Dual Style)")


st.divider()

# --- Generate Button and Processing Logic ---
if st.button("✨ Generate Styled Images", type="primary", disabled=not (content_file and style_file1)):

    # --- Load Images ---
    content_img = image_loader(content_file.getvalue()) if content_file else None
    style_img1 = image_loader(style_file1.getvalue()) if style_file1 else None

    if content_img is None or style_img1 is None:
        st.error("Failed to load Content and/or Style Image 1. Please check the files and try again.")
        st.stop() # Stop execution if essential images failed

    # --- Calculate Fixed VGG Content Features (once) ---
    content_features_vgg = None
    with st.spinner("Calculating VGG content features..."):
        try:
            content_features_vgg = get_vgg_features(content_img)
            if not content_features_vgg: # Check if feature extraction failed
                 raise ValueError("VGG Content feature extraction returned empty.")
        except Exception as e:
            st.error(f"Failed to extract VGG features from content image: {e}")
            st.stop()

    # --- 1. Dual Style Transfer (VGG) ---
    st.subheader("1. Dual Style Transfer (VGG-based)")
    if style_file2:
        style_img2 = image_loader(style_file2.getvalue())
        if style_img2 is None:
            st.warning("Could not load Style Image 2, skipping Dual Style.")
        else:
            with st.spinner(f"Running Dual Style transfer for {vgg_steps} steps..."):
                dual_prog_bar = st.progress(0, text="Dual Style Progress")
                try:
                    # Calculate combined style features
                    final_dual_style_features = get_dual_style_features(style_img1, style_img2)

                    if final_dual_style_features: # Proceed only if features were extracted
                         # Run optimization
                         output_tensor_dual = run_vgg_optimization(
                             content_img, # Initial image
                             content_features_vgg,
                             final_dual_style_features,
                             vgg_steps,
                             dual_prog_bar
                         )
                         # Convert and display
                         dual_style_output_img = tensor_to_pil(output_tensor_dual)
                         dual_prog_bar.empty()
                         if dual_style_output_img:
                             st.image(dual_style_output_img, caption=f'Dual Style Output ({vgg_steps} steps)', use_container_width=True)
                         else:
                             st.error("Dual style processing completed but failed to generate output image.")
                    else:
                         st.error("Failed to extract features needed for Dual Style.")
                         dual_prog_bar.empty()

                except Exception as e:
                     st.error(f"Error during Dual Style processing: {e}")
                     import traceback
                     st.error(traceback.format_exc())
                     dual_prog_bar.empty()
    else:
        st.info("Upload Style Image 2 to enable Dual Style Transfer.")

    st.divider()

    # --- 2. Rotated Style Transfer (VGG) ---
    st.subheader("2. Rotated Style Transfer (VGG-based)")
    with st.spinner(f"Running Rotated Style transfer for {vgg_steps} steps..."):
        rot_prog_bar = st.progress(0, text="Rotated Style Progress")
        try:
            # Calculate rotated style features
            final_rot_style_features = get_rot_style_features(style_img1)

            if final_rot_style_features: # Proceed only if features were extracted
                # Run optimization
                output_tensor_rot = run_vgg_optimization(
                    content_img, # Initial image
                    content_features_vgg,
                    final_rot_style_features,
                    vgg_steps,
                    rot_prog_bar
                )
                # Convert and display
                rot_style_output_img = tensor_to_pil(output_tensor_rot)
                rot_prog_bar.empty()
                if rot_style_output_img:
                    st.image(rot_style_output_img, caption=f'Rotated Style Output ({vgg_steps} steps)', use_container_width=True)
                else:
                    st.error("Rotated style processing completed but failed to generate output image.")
            else:
                st.error("Failed to extract features needed for Rotated Style.")
                rot_prog_bar.empty()

        except Exception as e:
            st.error(f"Error during Rotated Style processing: {e}")
            import traceback
            st.error(traceback.format_exc())
            rot_prog_bar.empty()


    st.divider()

    # --- 3. Transformer Style Transfer (StyTR) ---
    st.subheader("3. Transformer Style Transfer (StyTR)")
    if not setup_ok:
        st.warning("Skipping Transformer Style Transfer because setup failed.")
    else:
        with st.spinner("Running Transformer Style transfer (feed-forward)..."):
            try:
                # Run inference using the helper function
                transformer_output_img = run_stytr_inference(
                    content_file.getvalue(),
                    style_file1.getvalue()
                )

                if transformer_output_img:
                    st.image(transformer_output_img, caption='Transformer (StyTR) Output', use_container_width=True)
                else:
                    # Error message should have been shown within run_stytr_inference
                    st.error("Transformer style transfer failed to produce an image.")

            except Exception as e:
                st.error(f"An unexpected error occurred when calling StyTR inference: {e}")
                import traceback
                st.error(traceback.format_exc())